"""
YOLOv7-Face detection wrapper.

Supports two loading modes:
  - 'pytorch': Load from a .pt checkpoint. Requires the yolov7-face repository
    to be on sys.path (or installed).  See:
    https://github.com/derronqi/yolov7-face
  - 'onnx': Load from an exported .onnx file. Requires onnxruntime only.
    Export from PyTorch with:
      python export.py --weights yolov7-face.pt --img-size 640 --include onnx

Output per detection:
  {
    'bbox':       [x1, y1, x2, y2],     # int, original image coordinates
    'confidence': float,                 # objectness × class confidence
    'landmarks':  [[x, y], ...] × 5,   # float, original image coordinates
  }
Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def letterbox(
    im: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Resize image with unchanged aspect ratio using padding."""
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (W, H)

    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw, dh = dw % stride / 2, dh % stride / 2  # modulo padding

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    ratio = (r, r)
    pad   = (dw, dh)
    return im, ratio, pad


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _scale_back(
    coords: torch.Tensor,
    pad_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> torch.Tensor:
    """Scale (x1,y1,x2,y2) back from padded image to original image."""
    gain = min(pad_shape[0] / orig_shape[0], pad_shape[1] / orig_shape[1])
    pad_w = (pad_shape[1] - orig_shape[1] * gain) / 2
    pad_h = (pad_shape[0] - orig_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad_w
    coords[:, [1, 3]] -= pad_h
    coords[:, :4]      = (coords[:, :4] / gain).clamp(min=0)
    return coords


def _scale_landmarks_back(
    landmarks: torch.Tensor,
    pad_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> torch.Tensor:
    """Scale 10 landmark values back from padded image to original image."""
    gain = min(pad_shape[0] / orig_shape[0], pad_shape[1] / orig_shape[1])
    pad_w = (pad_shape[1] - orig_shape[1] * gain) / 2
    pad_h = (pad_shape[0] - orig_shape[0] * gain) / 2
    landmarks[:, [0, 2, 4, 6, 8]] -= pad_w
    landmarks[:, [1, 3, 5, 7, 9]] -= pad_h
    landmarks[:, :10] /= gain
    return landmarks


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def non_max_suppression_face(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float  = 0.45,
) -> List[torch.Tensor]:
    """
    NMS for YOLOv7-Face raw output.

    Args:
        prediction: [batch, anchors, 16]
            16 = 4 (cx,cy,w,h) + 1 (obj) + 1 (cls) + 10 (5 landmarks × xy)
        conf_thres: minimum objectness × class confidence
        iou_thres:  IoU threshold for NMS

    Returns:
        List of tensors [N, 16], one per image in batch.
        Columns: x1,y1,x2,y2, confidence, class_id, lm1x,lm1y,...,lm5y
    """
    batch_size = prediction.shape[0]
    output = [torch.zeros((0, 16), device=prediction.device)] * batch_size

    for i, x in enumerate(prediction):
        # Filter by objectness
        x = x[x[:, 4] > conf_thres]
        if not len(x):
            continue

        # Combined confidence = objectness × class score
        x[:, 5] *= x[:, 4]

        # Convert bbox to (x1, y1, x2, y2)
        box        = xywh2xyxy(x[:, :4])
        conf       = x[:, 5:6]
        class_ids  = torch.zeros_like(conf)  # single class: face = 0
        landmarks  = x[:, 6:]               # (N, 10)

        det = torch.cat([box, conf, class_ids, landmarks], dim=1)
        det = det[conf.view(-1) > conf_thres]
        if not len(det):
            continue

        # NMS
        keep = torchvision.ops.nms(det[:, :4], det[:, 4], iou_thres)
        output[i] = det[keep]

    return output


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YOLOv7FaceDetector:
    """
    YOLOv7-Face inference wrapper.

    Args:
        weights:    Path to model file (.pt for PyTorch, .onnx for ONNX).
        mode:       'pytorch' or 'onnx'. Inferred from extension if not given.
        yolov7_dir: Path to yolov7-face repo root (added to sys.path for
                    PyTorch mode). Required when mode='pytorch'.
        img_size:   Input resolution (square). Default 640.
        conf_thres: Confidence threshold. Default 0.25.
        iou_thres:  NMS IoU threshold. Default 0.45.
        device:     'cuda' | 'cpu' | None (auto-detect).
    """

    def __init__(
        self,
        weights: str,
        mode: str | None = None,
        yolov7_dir: str | None = None,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float  = 0.45,
        device: str | None = None,
    ) -> None:
        self.img_size   = img_size
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        weights_path = Path(weights)
        self.mode    = mode or ("onnx" if weights_path.suffix == ".onnx" else "pytorch")

        if self.mode == "pytorch":
            self._load_pytorch(weights_path, yolov7_dir)
        elif self.mode == "onnx":
            self._load_onnx(weights_path)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'pytorch' or 'onnx'.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_pytorch(self, weights: Path, yolov7_dir: str | None) -> None:
        if yolov7_dir:
            yolov7_path = str(Path(yolov7_dir).resolve())
            if yolov7_path not in sys.path:
                sys.path.insert(0, yolov7_path)

        ckpt = torch.load(str(weights), weights_only=False, map_location=self.device)
        model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
        self.model = model.float().eval().to(self.device)

        # Fuse Conv + BN layers for faster inference
        if hasattr(self.model, "fuse"):
            self.model.fuse()

    def _load_onnx(self, weights: Path) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(weights), providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------
    # Pre/post processing helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, Tuple]:
        """Return (tensor[1,3,H,W], padded_img, orig_shape)."""
        orig_shape = image.shape[:2]  # (H, W)
        padded, _, _ = letterbox(image, self.img_size, stride=32)
        rgb = padded[:, :, ::-1]  # BGR → RGB
        chw = rgb.transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).float() / 255.0
        return tensor.unsqueeze(0).to(self.device), padded, orig_shape

    def _preprocess_onnx(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        orig_shape = image.shape[:2]
        padded, _, _ = letterbox(image, self.img_size, stride=32)
        rgb = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(rgb, 0)
        return blob, padded, orig_shape

    def _postprocess(
        self,
        raw: torch.Tensor,
        pad_shape: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> List[dict]:
        dets = non_max_suppression_face(raw, self.conf_thres, self.iou_thres)
        return self._decode_detections(dets[0], pad_shape, orig_shape)

    def _decode_detections(
        self,
        det: torch.Tensor,
        pad_shape: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> List[dict]:
        if det is None or len(det) == 0:
            return []

        det[:, :4] = _scale_back(det[:, :4], pad_shape[:2], orig_shape)
        det[:, 6:] = _scale_landmarks_back(det[:, 6:], pad_shape[:2], orig_shape)

        results = []
        for d in det.cpu().numpy():
            x1, y1, x2, y2 = d[:4].astype(int)
            conf            = float(d[4])
            lm              = d[6:16].reshape(5, 2).tolist()
            results.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "landmarks": lm})

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in a BGR image.

        Args:
            image: numpy array (H, W, 3), uint8, BGR.

        Returns:
            List of dicts:
              {
                'bbox':       [x1, y1, x2, y2],   int, original-image coords
                'confidence': float,
                'landmarks':  [[x,y], ...] × 5,   original-image coords
              }
        """
        if self.mode == "pytorch":
            tensor, padded, orig = self._preprocess(image)
            with torch.no_grad():
                out = self.model(tensor)
            raw = out[0] if isinstance(out, (list, tuple)) else out
            return self._postprocess(raw, padded.shape[:2], orig)

        # ONNX
        blob, padded, orig = self._preprocess_onnx(image)
        raw_np = self.session.run([self.output_name], {self.input_name: blob})[0]
        raw    = torch.from_numpy(raw_np).to(self.device)
        return self._postprocess(raw, padded.shape[:2], orig)
