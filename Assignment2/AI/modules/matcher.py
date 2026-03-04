"""
Embedding matching module.

Compares a query embedding against a gallery of named embeddings and
returns the best-matching identity, score, and MATCH / NO MATCH decision.

Metric:
  - 'arcface' (default): Additive Angular Margin similarity as described in
    the ArcFace paper (Deng et al., CVPR 2019).

    For L2-normalized embeddings e_q and e_db, the similarity score is:

        score = cos(θ) = e_q · e_db

    where θ = arccos(e_q · e_db) is the geodesic angle between the two
    feature vectors on the unit hypersphere.  Higher score → smaller angle →
    more similar.  Range: [-1, 1].

    This is the exact verification metric used in the ArcFace paper
    (Section 4, LFW/YTF evaluation): the learned angular margin during
    training makes the decision boundary in angle space tight, so
    thresholding on cos(θ) at inference is the natural choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class MatchResult:
    identity: str
    score: float          # cos(θ) ∈ [-1, 1]; higher = more similar
    angle_deg: float      # θ in degrees; lower = more similar
    threshold: float      # cos(θ) threshold for MATCH decision
    matched: bool

    def __str__(self) -> str:
        result = "MATCH" if self.matched else "NO MATCH"
        return (
            f"Identity: {self.identity} | "
            f"cos(θ): {self.score:.4f} | "
            f"θ: {self.angle_deg:.2f}° | "
            f"Threshold: {self.threshold:.4f} | "
            f"Result: {result}"
        )


class FaceMatcher:
    """
    Match a query 512-D embedding against a pre-loaded gallery using the
    ArcFace additive angular margin similarity metric.

    Scoring formula (ArcFace paper, CVPR 2019):

        score = cos(θ) = e_q · e_db          (both vectors L2-normalized)
        θ     = arccos(score)                 (angle between embeddings)

    A face pair is declared MATCH when cos(θ) ≥ threshold, i.e. the angular
    distance θ is below the corresponding angle arccos(threshold).

    Args:
        names:      Array of identity names, shape (N,).
        embeddings: L2-normalized embedding matrix, shape (N, 512), float32.
        threshold:  cos(θ) decision boundary. Default 0.40 — a widely used
                    operating point for ArcFace R100 on real-world data
                    (corresponds to θ ≈ 66°).  Raise toward 1.0 for higher
                    precision; lower toward 0.0 for higher recall.
        metric:     Must be 'arcface' (the only supported metric).
    """

    def __init__(
        self,
        names: np.ndarray | list[str],
        embeddings: np.ndarray,
        threshold: float = 0.40,
        metric: Literal["arcface"] = "arcface",
    ) -> None:
        self.names      = np.asarray(names)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.threshold  = threshold
        self.metric     = metric

        if self.metric != "arcface":
            raise ValueError(
                f"Unknown metric: {self.metric!r}. "
                "Only 'arcface' (additive angular margin cosine similarity) is supported."
            )
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
        threshold: float = 0.40,
        metric: str = "arcface",
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
            MatchResult with identity, cos(θ) score, angle in degrees,
            threshold, and matched flag.
        """
        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        scores = self._compute_scores(query)          # cos(θ), shape (N,)
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_angle = float(np.degrees(np.arccos(np.clip(best_score, -1.0, 1.0))))

        if best_score >= self.threshold:
            identity = str(self.names[best_idx])
            matched  = True
        else:
            identity = "Unknown"
            matched  = False

        return MatchResult(
            identity=identity,
            score=best_score,
            angle_deg=best_angle,
            threshold=self.threshold,
            matched=matched,
        )

    def match_batch(self, query_embeddings: np.ndarray) -> list[MatchResult]:
        """
        Match multiple query embeddings in one vectorized call.

        Args:
            query_embeddings: float32 array of shape (M, 512), L2-normalized.

        Returns:
            List of MatchResult, one per row.
        """
        results = []
        for emb in query_embeddings:
            results.append(self.match(emb))
        return results

    def top_k(self, query_embedding: np.ndarray, k: int = 3) -> list[MatchResult]:
        """
        Return the top-k best-matching identities (regardless of threshold).

        Useful for debugging or when you need more than the single best match.
        """
        query  = np.asarray(query_embedding, dtype=np.float32).ravel()
        scores = self._compute_scores(query)
        k      = min(k, len(scores))
        idxs   = np.argsort(scores)[::-1][:k]

        return [
            MatchResult(
                identity=str(self.names[i]),
                score=float(scores[i]),
                angle_deg=float(np.degrees(np.arccos(np.clip(float(scores[i]), -1.0, 1.0)))),
                threshold=self.threshold,
                matched=float(scores[i]) >= self.threshold,
            )
            for i in idxs
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_scores(self, query: np.ndarray) -> np.ndarray:
        """
        Compute ArcFace angular similarity scores for all gallery entries.

        Returns cos(θ) = e_q · e_db for each gallery embedding, shape (N,).
        Both query and gallery embeddings must be L2-normalized so that the
        dot product equals the cosine of the angle between them.
        """
        # Dot product of unit vectors = cos(θ), the ArcFace similarity metric
        return self.embeddings @ query
