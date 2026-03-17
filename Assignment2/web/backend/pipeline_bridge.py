"""
pipeline_bridge.py — Initialises the AI pipeline once and exposes a simple
process_frame() helper for use by FastAPI routes.

Must be run from the Assignment2/ root directory so that the AI/ package is
on the path.
"""

from __future__ import annotations

import sys
import os
import base64
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Add AI/ to sys.path so the pipeline modules can be imported
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]   # Assignment2/
sys.path.insert(0, str(_ROOT / "AI"))
sys.path.insert(0, str(_ROOT / "AI" / "models" / "yolov7-face"))

from modules.detector   import YOLOv7FaceDetector
from modules.aligner    import FaceAligner
from modules.recognizer import ArcFaceRecognizer
from modules.matcher    import FaceMatcher
from pipeline           import FaceRecognitionPipeline, draw_results

# ---------------------------------------------------------------------------
# Config — paths relative to Assignment2/ root
# ---------------------------------------------------------------------------
_DETECTOR_WEIGHTS   = str(_ROOT / "AI" / "models" / "yolov7-tiny-face.pt")
_YOLOV7_DIR         = str(_ROOT / "AI" / "models" / "yolov7-face")
_RECOGNIZER_WEIGHTS = str(_ROOT / "AI" / "models" / "arcface_r100.onnx")
_DATABASE           = str(_ROOT / "AI" / "database" / "embeddings.npz")

# ---------------------------------------------------------------------------
# Pipeline singleton — built lazily on first use to survive import errors
# ---------------------------------------------------------------------------
_pipeline: FaceRecognitionPipeline | None = None


def get_pipeline() -> FaceRecognitionPipeline:
    global _pipeline
    if _pipeline is None:
        print("[bridge] Loading detector …")
        detector = YOLOv7FaceDetector(
            weights    = _DETECTOR_WEIGHTS,
            mode       = "pytorch",
            yolov7_dir = _YOLOV7_DIR,
            img_size   = 640,
            conf_thres = 0.25,
            iou_thres  = 0.45,
        )
        print("[bridge] Loading recognizer …")
        recognizer = ArcFaceRecognizer(model_path=_RECOGNIZER_WEIGHTS)

        print("[bridge] Loading gallery …")
        matcher = FaceMatcher.from_npz(
            path      = _DATABASE,
            threshold = 0.40,
            metric    = "cosine",
        )
        aligner   = FaceAligner()
        _pipeline = FaceRecognitionPipeline(detector, aligner, recognizer, matcher)
        print("[bridge] Pipeline ready.")
    return _pipeline


def process_frame(bgr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Run end-to-end inference on one BGR frame.

    Returns:
        annotated_bgr  — frame with bboxes, landmarks, labels drawn on it
        faces          — list of dicts ready to serialise as JSON:
                         {identity, score, angle_deg, result, bbox}
    """
    pipeline = get_pipeline()
    detections, match_results = pipeline.run(bgr)
    annotated = draw_results(bgr, detections, match_results)

    faces = [
        {
            "identity":  res.identity,
            "score":     round(float(res.score), 4),
            "threshold": round(float(res.threshold), 2),
            "result":    "MATCH" if res.matched else "NO MATCH",
            "bbox":      [int(v) for v in det["bbox"]],
        }
        for det, res in zip(detections, match_results)
    ]
    return annotated, faces


def bgr_to_base64_jpeg(bgr: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 JPEG string."""
    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpeg).decode()
