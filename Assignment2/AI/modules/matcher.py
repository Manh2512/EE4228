"""
Embedding matching module.

Compares a query embedding against a gallery of named embeddings and
returns the best-matching identity, score, and MATCH / NO MATCH decision.

Metrics:
  - 'cosine' (default): Cosine similarity (ArcFace verification metric).

        score = cos(θ) = e_q · e_db          (both vectors L2-normalized)

    Range: [-1, 1]. Higher score → more similar. MATCH when score ≥ threshold.
    Default threshold: 0.40 (≈ θ ≤ 66°).

  - 'euclidean': L2 distance between embeddings.

        distance = ‖e_q − e_db‖₂

    Range: [0, 2] for L2-normalized unit vectors. Lower distance → more similar.
    MATCH when distance ≤ threshold. Default threshold: 1.10 (≈ cos θ = 0.40).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class MatchResult:
    identity: str
    score: float      # cosine: cos(θ) ∈ [-1, 1]; euclidean: L2 distance ∈ [0, 2]
    threshold: float  # decision boundary (metric-dependent)
    matched: bool
    metric: str = field(default="cosine")

    def __str__(self) -> str:
        result = "MATCH" if self.matched else "NO MATCH"
        if self.metric == "cosine":
            return (
                f"Identity: {self.identity} | "
                f"cos(θ): {self.score:.4f} | "
                f"Threshold: {self.threshold:.4f} | "
                f"Result: {result}"
            )
        else:  # euclidean
            return (
                f"Identity: {self.identity} | "
                f"L2 dist: {self.score:.4f} | "
                f"Threshold: {self.threshold:.4f} | "
                f"Result: {result}"
            )


class FaceMatcher:
    """
    Match a query 512-D embedding against a pre-loaded gallery.

    Supported metrics:

      'cosine'    — Cosine similarity: score = e_q · e_db (L2-normalized dot product).
                    MATCH when score ≥ threshold. Default threshold: 0.40.

      'euclidean' — L2 distance: distance = ‖e_q − e_db‖₂.
                    MATCH when distance ≤ threshold. Default threshold: 1.10.

    Args:
        names:      Array of identity names, shape (N,).
        embeddings: L2-normalized embedding matrix, shape (N, 512), float32.
        threshold:  Decision boundary. Pass None to use the metric-specific default.
        metric:     'cosine' or 'euclidean'.
    """

    _DEFAULTS = {"cosine": 0.40, "euclidean": 1.10}

    def __init__(
        self,
        names: np.ndarray | list[str],
        embeddings: np.ndarray,
        threshold: float | None = None,
        metric: Literal["cosine", "euclidean"] = "cosine",
    ) -> None:
        if metric not in self._DEFAULTS:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                "Choose 'cosine' or 'euclidean'."
            )
        self.names      = np.asarray(names)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.metric     = metric
        self.threshold  = threshold if threshold is not None else self._DEFAULTS[metric]

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (N, D)")
        if len(self.names) != len(self.embeddings):
            raise ValueError("names and embeddings must have the same length")
        if len(self.names) == 0:
            raise ValueError(
                "Gallery is empty. Run build_database.py first to populate "
                "database/embeddings.npz."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_npz(
        cls,
        path: str,
        threshold: float | None = None,
        metric: str = "cosine",
    ) -> "FaceMatcher":
        """Load gallery from an .npz file produced by build_database.py."""
        data = np.load(path, allow_pickle=True)
        return cls(
            names=data["names"],
            embeddings=data["embeddings"],
            threshold=threshold,
            metric=metric,
        )

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Find the best-matching identity for a single query embedding.

        Args:
            query_embedding: L2-normalized float32 array of shape (512,).

        Returns:
            MatchResult with identity, score, threshold, and matched flag.
        """
        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        internal_scores = self._internal_scores(query)   # higher = better
        best_idx        = int(np.argmax(internal_scores))
        best_score      = self._natural_score(internal_scores[best_idx])

        matched  = self._is_match(best_score)
        identity = str(self.names[best_idx]) if matched else "Unknown"

        return MatchResult(
            identity=identity,
            score=best_score,
            threshold=self.threshold,
            matched=matched,
            metric=self.metric,
        )

    def match_batch(self, query_embeddings: np.ndarray) -> list[MatchResult]:
        """Match multiple query embeddings in one call."""
        return [self.match(emb) for emb in query_embeddings]

    def top_k(self, query_embedding: np.ndarray, k: int = 3) -> list[MatchResult]:
        """
        Return the top-k best-matching identities (regardless of threshold).
        """
        query  = np.asarray(query_embedding, dtype=np.float32).ravel()
        scores = self._internal_scores(query)
        k      = min(k, len(scores))
        idxs   = np.argsort(scores)[::-1][:k]

        return [
            MatchResult(
                identity=str(self.names[i]),
                score=self._natural_score(scores[i]),
                threshold=self.threshold,
                matched=self._is_match(self._natural_score(scores[i])),
                metric=self.metric,
            )
            for i in idxs
        ]

    def unknown_result(self) -> MatchResult:
        """Return a sentinel MatchResult for cases where matching cannot proceed."""
        worst = -1.0 if self.metric == "cosine" else 2.0
        return MatchResult("Unknown", worst, self.threshold, False, self.metric)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def all_scores(self, query_embedding: np.ndarray) -> list[tuple[str, float]]:
        """
        Return (name, score) for every gallery entry, sorted best-first.

        Score semantics match the chosen metric:
          cosine    → cos(θ), higher = more similar
          euclidean → L2 distance, lower = more similar
        """
        query  = np.asarray(query_embedding, dtype=np.float32).ravel()
        scores = self._internal_scores(query)
        order  = np.argsort(scores)[::-1]   # best first for both metrics
        return [
            (str(self.names[i]), self._natural_score(scores[i]))
            for i in order
        ]

    def _internal_scores(self, query: np.ndarray) -> np.ndarray:
        """Compute scores where higher always means more similar (for argmax)."""
        if self.metric == "cosine":
            return self.embeddings @ query
        else:  # euclidean — return negative distance so argmax picks closest
            diff = self.embeddings - query
            return -np.linalg.norm(diff, axis=1)

    def _natural_score(self, internal: float) -> float:
        """Convert internal score to the natural metric value shown to users."""
        if self.metric == "cosine":
            return float(internal)
        else:  # euclidean: internal is -distance
            return float(-internal)

    def _is_match(self, natural_score: float) -> bool:
        if self.metric == "cosine":
            return natural_score >= self.threshold
        else:  # euclidean: match when distance is small enough
            return natural_score <= self.threshold
