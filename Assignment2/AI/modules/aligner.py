"""
Face alignment module.

Warps a detected face to a canonical 112×112 crop using a 5-point
similarity transform (ArcFace standard reference landmarks).

Landmark order expected from the detector:
  0 – left eye center
  1 – right eye center
  2 – nose tip
  3 – left mouth corner
  4 – right mouth corner
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import transform as sk_transform


# ArcFace reference landmarks for a 112×112 output image.
# Source: InsightFace / arcface_torch standard.
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7318, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


class FaceAligner:
    """
    Align a detected face to the ArcFace-standard 112×112 crop.

    Args:
        output_size: (width, height) of the aligned crop. Default (112, 112).
    """

    def __init__(self, output_size: tuple[int, int] = (112, 112)) -> None:
        self.output_size = output_size
        scale = output_size[0] / 112.0
        self._dst = _ARCFACE_DST * scale  # rescale reference if needed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        image: np.ndarray,
        landmarks: list[list[float]] | np.ndarray,
    ) -> np.ndarray | None:
        """
        Crop and align a face from ``image`` using 5 facial landmarks.

        Args:
            image:     BGR numpy array (H, W, 3), uint8.
            landmarks: 5×2 array or list [[x,y], ...] in original-image coords.
                       Order: left_eye, right_eye, nose, left_mouth, right_mouth.

        Returns:
            Aligned face as (output_size[1], output_size[0], 3) uint8 BGR
            array, or None if the transform could not be estimated.
        """
        src = np.array(landmarks, dtype=np.float32)
        if src.shape != (5, 2):
            raise ValueError(f"Expected landmarks shape (5, 2), got {src.shape}")

        M = self._estimate_transform(src)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            image,
            M,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_transform(self, src: np.ndarray) -> np.ndarray | None:
        """
        Estimate a 2×3 affine matrix mapping ``src`` → ``self._dst``.

        Uses a similarity transform (translation + rotation + uniform scale).
        Returns None on degenerate input.
        """
        try:
            tform = sk_transform.SimilarityTransform()
            ok = tform.estimate(src, self._dst)
            if not ok:
                return None
            return tform.params[:2]  # 2×3 matrix
        except Exception:
            return None
