"""
Face alignment using 5-point landmarks from YOLOv7-Face.

Computes a similarity transform from the detected landmarks to canonical
ArcFace 112×112 reference positions and warps the face chip with
cv2.warpAffine. No Dlib required.

Landmark order expected (as produced by YOLOv7FaceDetector):
  left_eye, right_eye, nose, left_mouth, right_mouth
"""

from __future__ import annotations

import cv2
import numpy as np


# Canonical 5-point landmark positions for a 112×112 ArcFace chip.
_ARCFACE_REF_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


class FaceAligner:
    """
    Align a detected face to a canonical square crop using the 5 landmarks
    provided by YOLOv7-Face.

    Args:
        output_size: Side length (px) of the square output. Default 112
                     (ArcFace standard).
    """

    def __init__(self, output_size: int = 112) -> None:
        self.output_size = output_size
        scale = output_size / 112.0
        self._ref = (_ARCFACE_REF_112 * scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        image: np.ndarray,
        landmarks: list[list[float]],
    ) -> np.ndarray | None:
        """
        Crop and align a face using YOLO 5-point landmarks.

        Args:
            image:     BGR numpy array (H, W, 3), uint8.
            landmarks: [[x, y], ...] × 5 in original-image pixel coordinates,
                       as returned by YOLOv7FaceDetector.detect().

        Returns:
            Aligned face as (output_size, output_size, 3) uint8 BGR array,
            or None if transform estimation fails.
        """
        try:
            src = np.array(landmarks, dtype=np.float32)  # (5, 2)
            M, _ = cv2.estimateAffinePartial2D(src, self._ref, method=cv2.LMEDS)
            if M is None:
                return None

            return cv2.warpAffine(
                image, M, (self.output_size, self.output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        except Exception:
            return None

    def align_batch(
        self,
        image: np.ndarray,
        detections: list[dict],
    ) -> list[np.ndarray | None]:
        """
        Align all detected faces in one call.

        Args:
            image:      BGR source image.
            detections: List of dicts from YOLOv7FaceDetector.detect().

        Returns:
            List of aligned crops (same order as detections).
        """
        return [self.align(image, d["landmarks"]) for d in detections]
