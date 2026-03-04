"""
build_database.py — Gallery embedding database builder.

Scans  data/faces/<PersonName>/<image>.{jpg,jpeg,png}
Detects, aligns, and embeds each face image, then averages all embeddings
per identity and saves the result to  database/embeddings.npz.

Usage:
    python build_database.py \
        --detector-weights  models/yolov7-face.pt \
        --recognizer-weights models/arcface_r100.onnx \
        --faces-dir         data/faces \
        --output            database/embeddings.npz \
        [--detector-mode    pytorch|onnx] \
        [--yolov7-dir       path/to/yolov7-face] \
        [--img-size         640] \
        [--conf-thres       0.25] \
        [--metric           cosine]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from modules.detector   import YOLOv7FaceDetector
from modules.aligner    import FaceAligner
from modules.recognizer import ArcFaceRecognizer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build face embedding gallery database.")
    p.add_argument("--detector-weights",  required=True,           help="YOLOv7-Face .pt or .onnx weights")
    p.add_argument("--recognizer-weights",required=True,           help="ArcFace ONNX model path")
    p.add_argument("--faces-dir",         default="data/faces",    help="Root directory of identity folders")
    p.add_argument("--output",            default="database/embeddings.npz", help="Output .npz path")
    p.add_argument("--detector-mode",     default=None,            choices=["pytorch", "onnx"])
    p.add_argument("--yolov7-dir",        default=None,            help="yolov7-face repo root (pytorch mode only)")
    p.add_argument("--img-size",          type=int, default=640)
    p.add_argument("--conf-thres",        type=float, default=0.25)
    p.add_argument("--iou-thres",         type=float, default=0.45)
    p.add_argument("--force",             action="store_true",     help="Rebuild even if output already exists")
    return p.parse_args()


def collect_image_paths(faces_dir: Path) -> dict[str, list[Path]]:
    """Return {identity_name: [image_path, ...]} for all identities."""
    gallery: dict[str, list[Path]] = defaultdict(list)
    for identity_dir in sorted(faces_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                gallery[identity_dir.name].append(img_path)
    return dict(gallery)


def embed_identity(
    images: list[Path],
    detector: YOLOv7FaceDetector,
    aligner:  FaceAligner,
    recognizer: ArcFaceRecognizer,
    identity: str,
) -> np.ndarray | None:
    """
    Process all images for one identity.

    Returns the averaged, L2-renormalized 512-D embedding, or None if no
    valid face was found in any image.
    """
    accum: list[np.ndarray] = []

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [WARN] Cannot read {img_path}, skipping.", file=sys.stderr)
            continue

        detections = detector.detect(bgr)
        if not detections:
            print(f"  [WARN] No face in {img_path.name}, skipping.", file=sys.stderr)
            continue

        # Use the most-confident detection when multiple faces are present
        best = max(detections, key=lambda d: d["confidence"])

        aligned = aligner.align(bgr, best["landmarks"])
        if aligned is None:
            print(f"  [WARN] Alignment failed for {img_path.name}, skipping.", file=sys.stderr)
            continue

        embedding = recognizer.get_embedding(aligned)
        accum.append(embedding)

    if not accum:
        return None

    # Average all embeddings and re-normalize
    mean_emb = np.mean(np.stack(accum, axis=0), axis=0)
    norm     = np.linalg.norm(mean_emb)
    return mean_emb / (norm + 1e-6)


def build_database(args: argparse.Namespace) -> None:
    faces_dir  = Path(args.faces_dir)
    output_path = Path(args.output)

    if not faces_dir.exists():
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")

    if output_path.exists() and not args.force:
        print(f"Database already exists at {output_path}. Use --force to rebuild.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
    print("Loading detector …")
    detector = YOLOv7FaceDetector(
        weights     = args.detector_weights,
        mode        = args.detector_mode,
        yolov7_dir  = args.yolov7_dir,
        img_size    = args.img_size,
        conf_thres  = args.conf_thres,
        iou_thres   = args.iou_thres,
    )

    print("Loading recognizer …")
    recognizer = ArcFaceRecognizer(model_path=args.recognizer_weights)
    aligner    = FaceAligner()

    # ------------------------------------------------------------------ #
    # Collect images
    # ------------------------------------------------------------------ #
    gallery = collect_image_paths(faces_dir)
    if not gallery:
        raise RuntimeError(
            f"No identity folders found in {faces_dir}. "
            "Expected structure: data/faces/<PersonName>/<image>.jpg"
        )

    print(f"Found {len(gallery)} identities:")
    for name, paths in gallery.items():
        print(f"  {name}: {len(paths)} image(s)")

    # ------------------------------------------------------------------ #
    # Embed
    # ------------------------------------------------------------------ #
    all_names      : list[str]       = []
    all_embeddings : list[np.ndarray] = []

    for identity, images in tqdm(gallery.items(), desc="Embedding identities"):
        emb = embed_identity(images, detector, aligner, recognizer, identity)
        if emb is None:
            print(f"  [ERROR] No valid embedding for '{identity}', skipping entirely.", file=sys.stderr)
            continue
        all_names.append(identity)
        all_embeddings.append(emb)

    if not all_embeddings:
        raise RuntimeError("No embeddings were generated. Check model weights and face images.")

    embeddings_matrix = np.stack(all_embeddings, axis=0)  # (N, 512)
    names_array       = np.array(all_names)

    np.savez(str(output_path), names=names_array, embeddings=embeddings_matrix)

    print(
        f"\nDatabase saved to {output_path}\n"
        f"  Identities : {len(all_names)}\n"
        f"  Embedding dim: {embeddings_matrix.shape[1]}"
    )


if __name__ == "__main__":
    build_database(parse_args())
