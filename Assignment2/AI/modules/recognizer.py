"""
ArcFace recognition module.

Supports two backends selected automatically by file extension:

  .onnx          — ONNX Runtime (original behaviour, InsightFace models)
  .pt / .pth     — PyTorch IResNet100 checkpoint produced by finetune_arcface.py
                   Checkpoint format: dict with key 'backbone' (state dict),
                   or a bare state dict.

Accepts a 112×112 aligned BGR face crop and returns a 512-D
L2-normalized embedding.

Compatible ONNX models (download from InsightFace model zoo):
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
        model_path:      Path to the ArcFace model (.onnx, .pt, or .pth).
        use_gpu:         Use CUDA if available. Defaults to auto-detect.
        embedding_size:  Output dimension (PyTorch backend only; ignored for ONNX).
        dropout:         Dropout rate (PyTorch backend only; ignored for ONNX).
    """

    # ArcFace preprocessing constants
    _MEAN = 127.5
    _STD  = 127.5
    _INPUT_SIZE = (112, 112)

    def __init__(
        self,
        model_path: str,
        use_gpu: bool | None = None,
        embedding_size: int = 512,
        dropout: float = 0.4,
    ) -> None:
        import torch

        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        self._device = torch.device("cuda" if use_gpu else "cpu")
        suffix = Path(model_path).suffix.lower()

        if suffix == ".onnx":
            self._backend = "onnx"
            self._init_onnx(model_path, use_gpu)
        elif suffix in (".pt", ".pth"):
            self._backend = "pytorch"
            self._init_pytorch(model_path, embedding_size, dropout)
        else:
            raise ValueError(
                f"Unsupported model extension '{suffix}'. "
                "Expected .onnx, .pt, or .pth."
            )

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_onnx(self, model_path: str, use_gpu: bool) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self._session    = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._emb_size   = self._session.get_outputs()[0].shape[-1]

    def _init_pytorch(
        self, model_path: str, embedding_size: int, dropout: float
    ) -> None:
        import torch
        from modules.iresnet import IResNet100

        backbone = IResNet100(embedding_size=embedding_size, dropout=dropout)
        ckpt = torch.load(str(model_path), map_location=self._device)
        state = ckpt.get("backbone", ckpt) if isinstance(ckpt, dict) else ckpt
        backbone.load_state_dict(state, strict=False)
        backbone.to(self._device).eval()
        self._backbone  = backbone
        self._emb_size  = embedding_size

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
        if self._backend == "onnx":
            result = self._session.run(None, {self._input_name: blob})[0]
            embedding = result[0]
        else:
            embedding = self._forward_pytorch(blob)[0]
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

        if self._backend == "onnx":
            results = self._session.run(None, {self._input_name: blobs})[0]
        else:
            results = self._forward_pytorch(blobs)

        norms = np.linalg.norm(results, axis=1, keepdims=True) + 1e-6
        return results / norms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_pytorch(self, blob: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            t = torch.from_numpy(blob).to(self._device)
            out = F.normalize(self._backbone(t), p=2, dim=1)
        return out.cpu().numpy()

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
