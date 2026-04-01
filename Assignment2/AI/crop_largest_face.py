"""
crop_largest_face.py — Crop a 640×640 frame centred on the largest detected face.

Usage:
    python crop_largest_face.py --input-dir <folder> [--output-dir <folder>]
                                [--detector-weights models/yolov7-face.pt]
                                [--detector-mode pytorch]
                                [--yolov7-dir models/yolov7-face]

The script iterates over images whose filenames start with 'Photo' inside
--input-dir, detects faces with YOLOv7, picks the largest bounding box, and
saves a 640×640 crop centred on that face to --output-dir.
"""

from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make sure AI/ modules are importable when running from any CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.detector import YOLOv7FaceDetector


CROP_SIZE = 640


def crop_640_around_bbox(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Return a 640×640 crop centred on *bbox* (x1,y1,x2,y2), clamped to image borders."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    half = CROP_SIZE // 2
    left   = cx - half
    top    = cy - half
    right  = left + CROP_SIZE
    bottom = top  + CROP_SIZE

    # Shift window if it goes out of bounds.
    if left < 0:
        left, right = 0, CROP_SIZE
    if top < 0:
        top, bottom = 0, CROP_SIZE
    if right > w:
        right, left = w, w - CROP_SIZE
    if bottom > h:
        bottom, top = h, h - CROP_SIZE

    # Clamp to valid range (image smaller than 640).
    left   = max(0, left)
    top    = max(0, top)
    right  = min(w, right)
    bottom = min(h, bottom)

    return image[top:bottom, left:right]


def process_folder(
    input_dir: Path,
    output_dir: Path,
    detector: YOLOv7FaceDetector,
) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = sorted(
        p for p in input_dir.iterdir()
        if p.name.startswith("Photo") and p.suffix.lower() in exts
    )

    if not photos:
        print(f"No 'Photo*' images found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    iter = 13

    for photo_path in photos:
        frame = cv2.imread(str(photo_path))
        if frame is None:
            print(f"  [SKIP] Cannot read {photo_path.name}")
            continue

        detections = detector.detect(frame)

        if not detections:
            print(f"  [SKIP] No face detected in {photo_path.name}")
            continue

        # Largest face by bounding-box area.
        largest = max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )

        crop = crop_640_around_bbox(frame, largest["bbox"])

        # Pad to 640×640 if image was smaller than 640 in either dimension.
        ch, cw = crop.shape[:2]
        if ch < CROP_SIZE or cw < CROP_SIZE:
            padded = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=frame.dtype)
            padded[:ch, :cw] = crop
            crop = padded

        out_path_name = str(output_dir) + f"/{iter}.png"
        cv2.imwrite(out_path_name, crop)
        x1, y1, x2, y2 = largest["bbox"]
        print(f"  [OK]   {photo_path.name} → {out_path_name}  (face bbox {x1},{y1},{x2},{y2})")
        
        iter += 1
        # delete the original photo
        if os.path.exists(photo_path):
            os.remove(photo_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop 640×640 around largest face in Photo* images.")
    parser.add_argument("--input-dir",         required=True,  type=Path, help="Folder containing Photo* images")
    parser.add_argument("--output-dir",        type=Path,      default=None, help="Where to save crops (default: <input-dir>/cropped)")
    parser.add_argument("--detector-weights",  default="models/yolov7-face.pt")
    parser.add_argument("--detector-mode",     default="pytorch", choices=["pytorch", "onnx"])
    parser.add_argument("--yolov7-dir",        default="models/yolov7-face")
    parser.add_argument("--conf-thres",        type=float, default=0.3)
    parser.add_argument("--iou-thres",         type=float, default=0.5)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    print("Loading detector …")
    detector = YOLOv7FaceDetector(
        weights=args.detector_weights,
        mode=args.detector_mode,
        yolov7_dir=args.yolov7_dir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )

    print(f"Processing images in: {args.input_dir}")
    process_folder(args.input_dir, output_dir, detector)
    print(f"\nDone. Crops saved to: {output_dir}")


if __name__ == "__main__":
    main()
