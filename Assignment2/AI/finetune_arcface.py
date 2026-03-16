#!/usr/bin/env python3
"""
Fine-tune ArcFace (iResNet100) on a custom face dataset.

Pipeline
--------
Step 1 — Preprocessing (detect + align):
  Scans  <input_dir>/<identity>/<image>  using YOLOv7-Face and FaceAligner
  (the same logic as build_database.py) and writes 112×112 aligned face chips to
  <output_dir>/aligned_faces/<identity>/<stem>.jpg
  Images where no face is detected or alignment fails are skipped and logged.
  Re-run with --skip-preprocessing to skip this step (chips already exist).

Step 2 — Training:
  Fine-tunes IResNet100 + ArcFaceLoss on the aligned chips.

Outputs written to <output_dir>/:
    aligned_faces/             — pre-processed face chips (step 1)
    checkpoint_epoch_NNNN.pt   — periodic checkpoints
    best_model.pt              — lowest validation-loss checkpoint
    final_model.pt             — weights after the last epoch
    final_model.onnx           — ONNX export (requires --export-onnx)
    finetune.log               — full training log (INFO→stdout, DEBUG→file)

Pre-trained weight loading
--------------------------
  .pt / .pth  — PyTorch checkpoint (dict with key 'backbone', or bare state dict)
  .onnx       — InsightFace ONNX model; requires  pip install onnx2torch
                Tensors matched by name + shape; mismatches silently skipped.

Example
-------
    python finetune_arcface.py \\
        --input-dir   data/faces \\
        --output-dir  runs/exp01 \\
        --detector-weights  models/yolov7-face.pt \\
        --detector-mode     pytorch \\
        --yolov7-dir        models/yolov7-face \\
        --pretrained        models/arcface_r100.onnx \\
        --device cuda \\
        --epochs 30 --lr 1e-3 --weight-decay 5e-4 \\
        --aug-flip --aug-color-jitter \\
        --export-onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from modules.aligner  import FaceAligner
from modules.detector import YOLOv7FaceDetector
from modules.iresnet  import ArcFaceLoss, IResNet100

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Preprocessing: detect → align → save chips
# ──────────────────────────────────────────────────────────────────────────────

def prepare_aligned_faces(
    input_dir: Path,
    staging_dir: Path,
    detector: YOLOv7FaceDetector,
    aligner: FaceAligner,
    logger: logging.Logger,
    debug_dir: Path | None = None,
) -> int:
    """Detect and align all faces in *input_dir*, save chips to *staging_dir*.

    Layout mirrors the source::

        input_dir/<identity>/<image>  →  staging_dir/<identity>/<stem>.jpg

    Uses the most-confident detection when multiple faces appear in one photo,
    matching the behaviour of build_database.py.

    Returns the number of chips successfully saved.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug aligned chips will be saved to: {debug_dir}")

    identity_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    )
    if not identity_dirs:
        raise ValueError(f"No identity sub-directories found in {input_dir}")

    total_saved   = 0
    total_skipped = 0

    for identity_dir in tqdm(identity_dirs, desc="Preprocessing identities"):
        out_identity = staging_dir / identity_dir.name
        out_identity.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in identity_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not image_paths:
            logger.warning(f"No images in {identity_dir}, skipping identity")
            continue

        for img_path in image_paths:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                logger.warning(f"Cannot read {img_path}, skipping")
                total_skipped += 1
                continue

            detections = detector.detect(bgr)
            if not detections:
                logger.warning(f"No face detected in {img_path.name}, skipping")
                total_skipped += 1
                continue

            # Mirror build_database.py: use the most-confident detection
            best    = max(detections, key=lambda d: d['confidence'])
            aligned = aligner.align(bgr, best['landmarks'])
            if aligned is None:
                logger.warning(f"Alignment failed for {img_path.name}, skipping")
                total_skipped += 1
                continue

            out_path = out_identity / (img_path.stem + '.jpg')
            cv2.imwrite(str(out_path), aligned)
            total_saved += 1
            logger.debug(f"Saved chip: {out_path}")

            if debug_dir is not None:
                dbg_identity = debug_dir / identity_dir.name
                dbg_identity.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dbg_identity / (img_path.stem + '.jpg')), aligned)

    logger.info(
        f"Preprocessing complete — saved: {total_saved}, skipped: {total_skipped}"
    )
    return total_saved


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Dataset (operates on already-aligned 112×112 chips)
# ──────────────────────────────────────────────────────────────────────────────

class AlignedFaceDataset(Dataset):
    """Classification dataset over pre-aligned 112×112 face chips.

    Layout: ``<root>/<identity>/<chip>.jpg``
    """

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        identities = sorted(
            d for d in root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        )
        if not identities:
            raise ValueError(f"No identity sub-directories found in {root}")

        self.classes      = [d.name for d in identities]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for d in identities:
            idx = self.class_to_idx[d.name]
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in _IMG_EXTS:
                    self.samples.append((p, idx))

        if not self.samples:
            raise ValueError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        # Chips are BGR (saved by cv2); open as RGB for torchvision
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_transform(
    *,
    aug_flip: bool,
    aug_color_jitter: bool,
    aug_random_erasing: bool,
    aug_random_crop: bool,
    is_train: bool,
) -> transforms.Compose:
    ops: list = []

    if is_train and aug_random_crop:
        # Mild crop — chips are already tightly framed
        ops.append(transforms.RandomResizedCrop(112, scale=(0.85, 1.0), ratio=(0.95, 1.05)))
    else:
        ops.append(transforms.Resize((112, 112)))

    if is_train and aug_flip:
        ops.append(transforms.RandomHorizontalFlip())
    if is_train and aug_color_jitter:
        ops.append(
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
        )

    # ArcFace standard normalisation: mean=0.5, std=0.5 → range [-1, 1]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if is_train and aug_random_erasing:
        ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)))

    return transforms.Compose(ops)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    """Logger that writes INFO+ to stdout and DEBUG+ to *log_file*."""
    logger = logging.getLogger('arcface_finetune')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_pretrained(
    backbone: IResNet100,
    path: Path,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Initialise *backbone* from a .pt/.pth or .onnx checkpoint.

    For ONNX, ``onnx2torch`` converts the graph and tensors are matched by
    name + shape; mismatches are skipped so the call is safe even when the
    checkpoint was trained on a different number of identities.
    """
    suffix = path.suffix.lower()

    if suffix in ('.pt', '.pth'):
        ckpt  = torch.load(str(path), map_location=device)
        state = ckpt.get('backbone', ckpt)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        logger.info(
            f"PyTorch weights loaded from {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return

    if suffix == '.onnx':
        try:
            import onnx
            from onnx2torch import convert
        except ImportError:
            logger.warning(
                "onnx2torch not installed — ONNX weight transfer skipped. "
                "Install with:  pip install onnx2torch"
            )
            return
        try:
            logger.info(f"Converting ONNX → PyTorch for weight transfer: {path}")
            onnx_model  = onnx.load(str(path))
            torch_model = convert(onnx_model)
            onnx_sd     = torch_model.state_dict()
            own_sd      = backbone.state_dict()
            matched, skipped = 0, 0
            new_sd: dict = {}
            for k, v in own_sd.items():
                if k in onnx_sd and onnx_sd[k].shape == v.shape:
                    new_sd[k] = onnx_sd[k]
                    matched += 1
                else:
                    new_sd[k] = v
                    skipped += 1
            backbone.load_state_dict(new_sd)
            logger.info(
                f"ONNX weight transfer: {matched} matched, "
                f"{skipped} skipped (shape mismatch or name not found)"
            )
        except Exception as exc:
            logger.warning(f"ONNX weight transfer failed ({exc}) — using random init")
        return

    logger.warning(f"Unknown pretrained file extension '{suffix}' — skipping")


def export_onnx(backbone: IResNet100, output_path: Path, logger: logging.Logger) -> None:
    """Export *backbone* to ONNX with a dynamic batch axis."""
    backbone.eval().cpu()
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        backbone,
        dummy,
        str(output_path),
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=11,
        do_constant_folding=True,
    )
    logger.info(f"ONNX export saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / 'finetune.log')
    logger.info("=" * 70)
    logger.info("ArcFace Fine-tuning  (iResNet100 backbone)")
    logger.info("=" * 70)
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k:28s} = {v}")

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
    logger.info(f"Training device: {device}")

    # ── Step 1: preprocessing ────────────────────────────────────────────────
    staging_dir = output_dir / 'aligned_faces'

    if args.skip_preprocessing:
        logger.info(f"Skipping preprocessing — using existing chips in {staging_dir}")
        if not staging_dir.exists():
            raise FileNotFoundError(
                f"--skip-preprocessing set but {staging_dir} does not exist"
            )
    else:
        logger.info("Step 1 — Loading detector and aligner …")
        detector = YOLOv7FaceDetector(
            weights    = args.detector_weights,
            mode       = args.detector_mode,
            yolov7_dir = args.yolov7_dir,
            img_size   = args.img_size,
            conf_thres = args.conf_thres,
            iou_thres  = args.iou_thres,
        )
        aligner = FaceAligner(output_size=112)

        debug_dir = Path(args.debug_aligned_dir) if args.debug_aligned_dir else None
        logger.info(f"Step 1 — Detecting and aligning faces: {args.input_dir} → {staging_dir}")
        n_chips = prepare_aligned_faces(
            Path(args.input_dir), staging_dir, detector, aligner, logger, debug_dir
        )
        if n_chips == 0:
            raise RuntimeError(
                "No aligned face chips were produced. "
                "Check detector weights, --conf-thres, and your input images."
            )

    # ── Step 2: training ─────────────────────────────────────────────────────
    logger.info("Step 2 — Setting up training …")

    # Class metadata always derived from the staging directory
    meta_ds      = AlignedFaceDataset(staging_dir, transform=None)
    num_classes  = len(meta_ds.classes)
    class_to_idx = meta_ds.class_to_idx
    logger.info(f"Classes ({num_classes}): {meta_ds.classes}")

    train_tf = _build_transform(
        aug_flip=args.aug_flip,
        aug_color_jitter=args.aug_color_jitter,
        aug_random_erasing=args.aug_random_erasing,
        aug_random_crop=args.aug_random_crop,
        is_train=True,
    )
    val_tf = _build_transform(
        aug_flip=False, aug_color_jitter=False,
        aug_random_erasing=False, aug_random_crop=False,
        is_train=False,
    )

    if args.val_dir:
        train_dataset: Dataset = AlignedFaceDataset(staging_dir, transform=train_tf)
        val_dataset:   Dataset = AlignedFaceDataset(Path(args.val_dir), transform=val_tf)
    else:
        n       = len(meta_ds)
        n_val   = max(1, int(n * args.val_split))
        n_train = n - n_val
        perm    = torch.randperm(n).tolist()
        train_dataset = Subset(AlignedFaceDataset(staging_dir, transform=train_tf), perm[:n_train])
        val_dataset   = Subset(AlignedFaceDataset(staging_dir, transform=val_tf),   perm[n_train:])

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    backbone = IResNet100(embedding_size=args.embedding_size, dropout=args.dropout).to(device)
    head     = ArcFaceLoss(
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        scale=args.scale,
        margin=args.margin,
    ).to(device)

    logger.info(
        f"Backbone params: {sum(p.numel() for p in backbone.parameters()):,} | "
        f"ArcFace head params: {sum(p.numel() for p in head.parameters()):,}"
    )

    if args.pretrained:
        load_pretrained(backbone, Path(args.pretrained), device, logger)

    # ── Optimiser ────────────────────────────────────────────────────────────
    param_groups = [
        {'params': backbone.parameters()},
        {'params': head.parameters()},
    ]
    if args.optimizer == 'sgd':
        optimizer: optim.Optimizer = optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:  # adamw
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    scheduler: optim.lr_scheduler.LRScheduler | None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        milestones = [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        scheduler  = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
        logger.info(f"MultiStepLR milestones: {milestones}")
    else:
        scheduler = None

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt.get('scheduler'):
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from {args.resume} (next epoch: {start_epoch})")

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info(f"Starting training: epochs {start_epoch}–{args.epochs}")

    val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):

        # -- train ----------------------------------------------------------
        backbone.train()
        head.train()
        epoch_train_loss = 0.0
        n_train          = 0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            emb  = F.normalize(backbone(imgs), p=2, dim=1)
            loss = head(emb, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            bs               = imgs.size(0)
            epoch_train_loss += loss.item() * bs
            n_train          += bs

            if batch_idx % args.log_interval == 0:
                logger.debug(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )

        train_loss = epoch_train_loss / max(n_train, 1)

        # -- validate -------------------------------------------------------
        backbone.eval()
        head.eval()
        epoch_val_loss = 0.0
        n_val          = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                emb    = F.normalize(backbone(imgs), p=2, dim=1)
                loss   = head(emb, labels)
                bs     = imgs.size(0)
                epoch_val_loss += loss.item() * bs
                n_val          += bs

        val_loss = epoch_val_loss / max(n_val, 1)
        elapsed  = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"time {elapsed:.1f}s"
        )

        # -- checkpoint -----------------------------------------------------
        ckpt_payload = {
            'epoch':         epoch,
            'backbone':      backbone.state_dict(),
            'head':          head.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict() if scheduler else None,
            'train_loss':    train_loss,
            'val_loss':      val_loss,
            'best_val_loss': best_val_loss,
            'class_to_idx':  class_to_idx,
            'args':          vars(args),
        }

        if epoch % args.save_interval == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            torch.save(ckpt_payload, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss                 = val_loss
            ckpt_payload['best_val_loss'] = best_val_loss
            torch.save(ckpt_payload, output_dir / 'best_model.pt')
            logger.info(f"Best model updated (val_loss={val_loss:.4f})")

    # ── Final save ───────────────────────────────────────────────────────────
    torch.save(
        {
            'epoch':         args.epochs,
            'backbone':      backbone.state_dict(),
            'head':          head.state_dict(),
            'val_loss':      val_loss,
            'best_val_loss': best_val_loss,
            'class_to_idx':  class_to_idx,
            'args':          vars(args),
        },
        output_dir / 'final_model.pt',
    )
    logger.info(f"Final model saved: {output_dir / 'final_model.pt'}")

    if args.export_onnx:
        export_onnx(backbone, output_dir / 'final_model.onnx', logger)

    logger.info("=" * 70)
    logger.info("Training complete")
    logger.info(f"  Best val loss : {best_val_loss:.4f}")
    logger.info(f"  Output dir    : {output_dir.resolve()}")
    logger.info(f"  Log file      : {(output_dir / 'finetune.log').resolve()}")
    logger.info("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    P = argparse.ArgumentParser(
        description='Fine-tune ArcFace (iResNet100) on a custom face dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O ──────────────────────────────────────────────────────────────────────
    io = P.add_argument_group('I/O')
    io.add_argument(
        '--input-dir', required=True, metavar='DIR',
        help='Raw image root — layout: <dir>/<identity>/<image>',
    )
    io.add_argument(
        '--output-dir', required=True, metavar='DIR',
        help='Destination for aligned chips, checkpoints, ONNX export, and log',
    )
    io.add_argument(
        '--val-dir', default=None, metavar='DIR',
        help='Separate validation directory of pre-aligned chips (optional). '
             'If omitted, --val-split fraction of the training chips is held out.',
    )
    io.add_argument(
        '--pretrained', default=None, metavar='FILE',
        help='Pre-trained backbone weights: .onnx (InsightFace), .pt, or .pth',
    )
    io.add_argument(
        '--resume', default=None, metavar='CKPT',
        help='Checkpoint .pt to resume training from',
    )
    io.add_argument(
        '--export-onnx', action='store_true',
        help='Export the final backbone to ONNX after training',
    )
    io.add_argument(
        '--skip-preprocessing', action='store_true',
        help='Skip face detection/alignment and use existing chips in '
             '<output-dir>/aligned_faces/',
    )
    io.add_argument(
        '--debug-aligned-dir', default='data/aligned_debug', metavar='DIR',
        help='Directory under data/ where aligned chips are also saved for debugging '
             '(set to empty string to disable)',
    )

    # Detector (preprocessing) ─────────────────────────────────────────────────
    det = P.add_argument_group('Detector (preprocessing)')
    det.add_argument(
        '--detector-weights', default='models/yolov7-face.pt', metavar='FILE',
        help='YOLOv7-Face weights (.pt or .onnx)',
    )
    det.add_argument(
        '--detector-mode', default=None, choices=['pytorch', 'onnx'],
        help='Detector loading mode (inferred from extension if omitted)',
    )
    det.add_argument(
        '--yolov7-dir', default=None, metavar='DIR',
        help='yolov7-face repo root added to sys.path (pytorch mode only)',
    )
    det.add_argument('--img-size',   type=int,   default=640,  help='Detector input resolution')
    det.add_argument('--conf-thres', type=float, default=0.25, help='Detection confidence threshold')
    det.add_argument('--iou-thres',  type=float, default=0.45, help='NMS IoU threshold')

    # Device ───────────────────────────────────────────────────────────────────
    P.add_argument(
        '--device', choices=['cuda', 'cpu'], default='cuda',
        help='Training device (falls back to CPU if CUDA unavailable)',
    )

    # Hyperparameters ──────────────────────────────────────────────────────────
    hp = P.add_argument_group('Hyperparameters')
    hp.add_argument('--epochs',         type=int,   default=30,   help='Training epochs')
    hp.add_argument('--batch-size',     type=int,   default=32,   help='Mini-batch size')
    hp.add_argument('--lr',             type=float, default=1e-3, help='Initial learning rate')
    hp.add_argument(
        '--weight-decay', type=float, default=5e-4,
        help='L2 regularisation coefficient (weight decay)',
    )
    hp.add_argument('--momentum',       type=float, default=0.9,  help='SGD momentum (ignored for AdamW)')
    hp.add_argument('--optimizer',      choices=['sgd', 'adamw'], default='sgd', help='Optimiser')
    hp.add_argument(
        '--scheduler', choices=['cosine', 'step', 'none'], default='cosine',
        help='LR scheduler — cosine annealing | multi-step (×0.1 at 60%% and 80%% of epochs) | none',
    )
    hp.add_argument('--embedding-size', type=int,   default=512,  help='Output embedding dimension')
    hp.add_argument('--dropout',        type=float, default=0.4,  help='Dropout in backbone output layer')
    hp.add_argument('--scale',          type=float, default=64.0, help='ArcFace logit scale s')
    hp.add_argument(
        '--margin', type=float, default=0.5,
        help='ArcFace angular margin m in radians (0.5 ≈ 28.6°)',
    )
    hp.add_argument(
        '--val-split', type=float, default=0.1,
        help='Fraction of chips held out for validation when --val-dir is not set',
    )

    # Data augmentation ────────────────────────────────────────────────────────
    aug = P.add_argument_group('Data augmentation')
    aug.add_argument('--aug-flip', action='store_true', default=True,
                     help='Random horizontal flip (on by default)')
    aug.add_argument('--no-aug-flip', dest='aug_flip', action='store_false',
                     help='Disable random horizontal flip')
    aug.add_argument('--aug-color-jitter', action='store_true', default=True,
                     help='Color jitter — brightness/contrast/saturation/hue (on by default)')
    aug.add_argument('--no-aug-color-jitter', dest='aug_color_jitter', action='store_false',
                     help='Disable color jitter')
    aug.add_argument('--aug-random-erasing', action='store_true', default=False,
                     help='Random erasing to simulate partial occlusion')
    aug.add_argument('--aug-random-crop', action='store_true', default=False,
                     help='RandomResizedCrop (scale 0.85–1.0) instead of plain Resize')

    # Runtime ──────────────────────────────────────────────────────────────────
    rt = P.add_argument_group('Runtime')
    rt.add_argument('--num-workers',   type=int, default=4,  help='DataLoader worker threads')
    rt.add_argument('--log-interval',  type=int, default=10, help='Write a DEBUG entry every N batches')
    rt.add_argument('--save-interval', type=int, default=5,  help='Save a checkpoint every N epochs')

    return P.parse_args()


if __name__ == '__main__':
    run(_parse_args())
