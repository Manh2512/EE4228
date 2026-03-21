"""
visualize_embeddings.py — 2-D PCA scatter of gallery embeddings.

Usage (run from AI/):
    python visualize_embeddings.py --database database/embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="database/embeddings.npz")
    p.add_argument("--output", default=None, help="Save figure to file instead of showing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db = np.load(args.database)
    names: np.ndarray = db["names"]          # (N,)
    embeddings: np.ndarray = db["embeddings"]  # (N, 512)

    n = len(names)
    print(f"Loaded {n} identities: {list(names)}")

    # Project 512-D → 2-D with PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)   # (N, 2)
    var = pca.explained_variance_ratio_ * 100

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_names = list(dict.fromkeys(names))   # ordered, deduplicated
    cmap = plt.get_cmap("tab10")
    name_to_color = {name: cmap(i % 10) for i, name in enumerate(unique_names)}

    labeled = set()
    for name, (x, y) in zip(names, coords):
        color = name_to_color[name]
        ax.scatter(
            x, y,
            color=color, s=50, zorder=3,
            edgecolors="white", linewidths=0.8,
            label=name if name not in labeled else "_nolegend_",
        )
        labeled.add(name)

    if len(unique_names) <= 20:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Identity", fontsize=9, title_fontsize=10)

    ax.set_xlabel(f"PC 1  ({var[0]:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC 2  ({var[1]:.1f}% var)", fontsize=11)
    ax.set_title("ArcFace Embedding Space — Gallery Identities (PCA 2-D)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
