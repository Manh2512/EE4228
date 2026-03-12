"""
pipeline.py — End-to-end face detection and recognition inference.

Modes:
  --mode image : Process a single image file and save annotated result.
  --mode video : Process a video file (or webcam index) in real-time.

Result semantics
  - MATCH   (res.matched=True):  cos(θ) >= threshold → identity = gallery name (e.g. "John_Doe")
  - NO MATCH (res.matched=False): cos(θ) < threshold  → identity = "Unknown"

Usage examples:
  # Single image
  python pipeline.py \
      --mode image \
      --input photo.jpg \
      --output result.jpg \
      --detector-weights  models/yolov7-face.pt \
      --detector-mode pytorch \
      --yolov7-dir models/yolov7-face \
      --recognizer-weights models/arcface_r100.onnx \
      --database          database/embeddings.npz

  # Webcam (device 0)
  python pipeline.py \
      --mode video \
      --input 0 \
      --detector-weights  models/yolov7-face.pt \
      --detector-mode pytorch \
      --yolov7-dir models/yolov7-face \
      --recognizer-weights models/arcface_r100.onnx \
      --database          database/embeddings.npz

  # Video file
  python pipeline.py \
      --mode video \
      --input clip.mp4 \
      --output annotated.mp4 \
      --detector-weights  models/yolov7-face.pt \
      --detector-mode pytorch \
      --yolov7-dir models/yolov7-face \
      --recognizer-weights models/arcface_r100.onnx \
      --database          database/embeddings.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from modules.detector   import YOLOv7FaceDetector
from modules.aligner    import FaceAligner
from modules.recognizer import ArcFaceRecognizer
from modules.matcher    import FaceMatcher, MatchResult


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_COLOR_MATCH    = (0, 200, 0)   # green  — identity confirmed
_COLOR_NO_MATCH = (0, 0, 220)   # red    — below threshold / "Unknown"

# 5-point landmark colors: left eye, right eye, nose, left mouth, right mouth
_LANDMARK_COLORS = [
    (255,   0, 255),  # left eye       — magenta
    (  0, 128, 255),  # right eye      — orange
    (  0, 255, 255),  # nose tip       — yellow
    (  0, 255,   0),  # left mouth     — green
    (255, 255,   0),  # right mouth    — cyan
]


def draw_results(
    frame: np.ndarray,
    detections: list[dict],
    results: list[MatchResult],
) -> np.ndarray:
    """
    Annotate frame with bounding boxes, landmarks, and identity labels.

    Label format: "<identity> (<score>)"
      - Green box  → MATCH   (res.matched=True,  identity = known name)
      - Red box    → NO MATCH (res.matched=False, identity = "Unknown")
    """
    out = frame.copy()
    for det, res in zip(detections, results):
        x1, y1, x2, y2 = det["bbox"]
        color = _COLOR_MATCH if res.matched else _COLOR_NO_MATCH

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Facial landmarks (5-point: left eye, right eye, nose, left mouth, right mouth)
        for i, (lx, ly) in enumerate(det["landmarks"]):
            color_lm = _LANDMARK_COLORS[i]
            cv2.circle(out, (int(lx), int(ly)), 4, color_lm, -1)

        # Identity label with background rectangle for readability
        label = f"{res.identity} ({res.score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            out, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class FaceRecognitionPipeline:
    """
    Orchestrates Detector → Aligner → Recognizer → Matcher.

    Args:
        detector:   YOLOv7FaceDetector instance.
        aligner:    FaceAligner instance.
        recognizer: ArcFaceRecognizer instance.
        matcher:    FaceMatcher instance (pre-loaded gallery).
    """

    def __init__(
        self,
        detector:   YOLOv7FaceDetector,
        aligner:    FaceAligner,
        recognizer: ArcFaceRecognizer,
        matcher:    FaceMatcher,
    ) -> None:
        self.detector   = detector
        self.aligner    = aligner
        self.recognizer = recognizer
        self.matcher    = matcher

    def run(self, frame: np.ndarray) -> tuple[list[dict], list[MatchResult]]:
        """
        Process one BGR frame end-to-end.

        Steps per detected face:
          1. Align  → 112×112 crop
          2. Embed  → 512-D L2-normalized vector
          3. Match  → MatchResult

        Returns:
            (detections, match_results)
            detections    — list of dicts from YOLOv7FaceDetector
            match_results — one MatchResult per detection, same order
        """
        detections = self.detector.detect(frame)
        if not detections:
            return [], []

        match_results: list[MatchResult] = []

        for det in detections:
            aligned = self.aligner.align(frame, det["landmarks"])

            if aligned is None:
                # Alignment degenerated (e.g. collinear landmarks) → Unknown
                match_results.append(
                    MatchResult("Unknown", 0.0, float(np.degrees(np.arccos(0.0))), self.matcher.threshold, False)
                )
                continue

            embedding = self.recognizer.get_embedding(aligned)
            result    = self.matcher.match(embedding)
            match_results.append(result)

            print(result)

        return detections, match_results


# ---------------------------------------------------------------------------
# Image / video runners
# ---------------------------------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> FaceRecognitionPipeline:
    print("Loading detector …")
    detector = YOLOv7FaceDetector(
        weights    = args.detector_weights,
        mode       = args.detector_mode,
        yolov7_dir = args.yolov7_dir,
        img_size   = args.img_size,
        conf_thres = args.conf_thres,
        iou_thres  = args.iou_thres,
    )

    print("Loading recognizer …")
    recognizer = ArcFaceRecognizer(model_path=args.recognizer_weights)

    print("Loading gallery …")
    matcher = FaceMatcher.from_npz(
        path      = args.database,
        threshold = args.threshold,
        metric    = args.metric,
    )

    return FaceRecognitionPipeline(detector, FaceAligner(), recognizer, matcher)


def run_image(args: argparse.Namespace) -> None:
    frame = cv2.imread(args.input)
    if frame is None:
        sys.exit(f"Cannot read image: {args.input}")

    pipeline = build_pipeline(args)

    t0 = time.perf_counter()
    detections, results = pipeline.run(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\nDetected {len(detections)} face(s) in {elapsed_ms:.1f} ms")

    annotated = draw_results(frame, detections, results)

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Saved annotated image → {args.output}")
    else:
        cv2.imshow("Face Recognition", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_folder(args: argparse.Namespace) -> None:
    folder = Path(args.input)
    if not folder.is_dir():
        sys.exit(f"Not a directory: {args.input}")

    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    )
    if not image_paths:
        sys.exit(f"No images found in: {args.input}")

    out_dir: Path | None = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args)

    total_faces = 0
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue

        t0 = time.perf_counter()
        detections, results = pipeline.run(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_faces += len(detections)
        print(f"{img_path.name}: {len(detections)} face(s) in {elapsed_ms:.1f} ms")

        annotated = draw_results(frame, detections, results)

        if out_dir:
            cv2.imwrite(str(out_dir / img_path.name), annotated)
        else:
            cv2.imshow(f"Face Recognition — {img_path.name}", annotated)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    print(f"\nProcessed {len(image_paths)} image(s), {total_faces} face(s) total.")
    if out_dir:
        print(f"Annotated images saved → {out_dir}")


def run_video(args: argparse.Namespace) -> None:
    source = int(args.input) if args.input.isdigit() else args.input
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open video source: {args.input}")

    pipeline = build_pipeline(args)

    writer: cv2.VideoWriter | None = None
    max_fps = 20 # keep fps low to accelerate processing step
    
    if args.output:
        fps    = cap.get(cv2.CAP_PROP_FPS) or max_fps
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            detections, results = pipeline.run(frame)
            annotated = draw_results(frame, detections, results)

            if writer:
                writer.write(annotated)
            else:
                cv2.imshow("Face Recognition — press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Video saved → {args.output}")
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frame(s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face Detection and Recognition Pipeline")
    p.add_argument("--mode",                 required=True, choices=["image", "video", "folder"])
    p.add_argument("--input",                required=True, help="Image/video path or webcam index (0, 1 …)")
    p.add_argument("--output",               default=None,  help="Output path (optional; shows window if omitted)")
    p.add_argument("--detector-weights",     required=True, help=".pt or .onnx YOLOv7-Face weights")
    p.add_argument("--recognizer-weights",   required=True, help="ArcFace ONNX model path")
    p.add_argument("--database",             default="database/embeddings.npz")
    p.add_argument("--detector-mode",        default=None,  choices=["pytorch", "onnx"])
    p.add_argument("--yolov7-dir",           default=None,  help="yolov7-face repo root (pytorch mode only)")
    p.add_argument("--img-size",             type=int,   default=640)
    p.add_argument("--conf-thres",           type=float, default=0.25)
    p.add_argument("--iou-thres",            type=float, default=0.45)
    p.add_argument("--threshold",            type=float, default=0.40,
                                             help="ArcFace angular similarity threshold: cos(θ) ≥ threshold → MATCH "
                                                  "(default 0.40 ≈ θ ≤ 66°)")
    p.add_argument("--metric",               default="arcface", choices=["arcface"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "image":
        run_image(args)
    elif args.mode == "folder":
        run_folder(args)
    else:
        run_video(args)
