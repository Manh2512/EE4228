"""
ArcFace recognition module (ONNX Runtime backend).

Accepts a 112×112 aligned BGR face crop and returns a 512-D
L2-normalized embedding.

Compatible models (download from InsightFace model zoo):
  - buffalo_l/w600k_r50.onnx  (iResNet50, WF600K, recommended)
  - arcface_r100_v1.onnx      (iResNet100, MS1MV2)

Download helper:
  python -c "
  import insightface
  from insightface.app import FaceAnalysis
  app = FaceAnalysis(name='buffalo_l', allowed_modules=['recognition'])
  app.prepare(ctx_id=-1)
  "
  # Model will be cached to ~/.insightface/models/buffalo_l/w600k_r50.onnx
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ArcFaceRecognizer:
    """
    ArcFace embedding extractor.

    Args:
        model_path: Path to the ArcFace ONNX model file.
        use_gpu:    Use CUDA execution provider if available. Defaults to
                    auto-detect via ``torch.cuda.is_available()``.
    """

    # ArcFace preprocessing constants
    _MEAN = 127.5
    _STD  = 127.5
    _INPUT_SIZE = (112, 112)

    def __init__(self, model_path: str, use_gpu: bool | None = None) -> None:
        import onnxruntime as ort
        import torch

        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        self._session    = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._emb_size   = self._session.get_outputs()[0].shape[-1]  # typically 512

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embedding_size(self) -> int:
        return self._emb_size

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalized 512-D embedding.

        Args:
            aligned_face_bgr: 112×112 BGR uint8 numpy array (output of
                              FaceAligner.align).

        Returns:
            np.ndarray of shape (512,), float32, L2-normalized.
        """
        blob = self._preprocess(aligned_face_bgr)
        result = self._session.run(None, {self._input_name: blob})[0]
        embedding = result[0]           # (512,)
        return self._l2_normalize(embedding)

    def get_embeddings_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> np.ndarray:
        """
        Extract embeddings for a list of aligned faces.

        Args:
            aligned_faces: List of 112×112 BGR uint8 arrays.

        Returns:
            np.ndarray of shape (N, 512), float32, each row L2-normalized.
        """
        if not aligned_faces:
            return np.zeros((0, self._emb_size), dtype=np.float32)

        blobs = np.concatenate([self._preprocess(f) for f in aligned_faces], axis=0)
        results = self._session.run(None, {self._input_name: blobs})[0]  # (N, 512)
        norms   = np.linalg.norm(results, axis=1, keepdims=True) + 1e-6
        return results / norms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return a [1, 3, 112, 112] float32 NCHW blob in range [-1, 1]."""
        if face_bgr.shape[:2] != self._INPUT_SIZE:
            face_bgr = cv2.resize(face_bgr, self._INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        blob = (face_rgb.astype(np.float32) - self._MEAN) / self._STD  # [-1, 1]
        blob = blob.transpose(2, 0, 1)          # HWC → CHW
        blob = np.expand_dims(blob, axis=0)     # → [1, 3, 112, 112]
        return blob

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6)
