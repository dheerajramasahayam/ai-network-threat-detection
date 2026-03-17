from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parents[1]


def add_box(ax, x: float, y: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        1.8,
        0.62,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=color,
        facecolor="#f8fafc",
    )
    ax.add_patch(box)
    ax.text(x + 0.9, y + 0.31, text, ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#334155")
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    add_box(ax, 0.3, 2.95, "UNSW-NB15 Official", "#0f766e")
    add_box(ax, 2.55, 2.95, "NSL-KDD Official", "#1d4ed8")
    add_box(ax, 4.8, 2.95, "CICIDS2017 Holdout", "#b45309")
    add_box(ax, 3.15, 1.95, "Canonical 41 Features", "#475569")

    add_box(ax, 0.35, 0.85, "Signature IDS", "#991b1b")
    add_box(ax, 2.55, 0.85, "Random Forest", "#0f766e")
    add_box(ax, 4.75, 0.85, "LSTM", "#1d4ed8")
    add_box(ax, 6.95, 0.85, "Transformer", "#b45309")
    add_box(ax, 9.15, 0.85, "Drift-Adaptive Hybrid", "#7c3aed")
    add_box(ax, 11.35, 0.85, "Security Alert", "#be123c")

    add_arrow(ax, 2.1, 3.26, 3.6, 2.57)
    add_arrow(ax, 4.35, 3.26, 4.05, 2.57)
    add_arrow(ax, 6.6, 3.26, 4.5, 2.57)

    add_arrow(ax, 4.05, 1.95, 1.2, 1.47)
    add_arrow(ax, 4.05, 1.95, 3.4, 1.47)
    add_arrow(ax, 4.95, 1.95, 5.6, 1.47)
    add_arrow(ax, 5.35, 1.95, 7.8, 1.47)
    add_arrow(ax, 2.1, 1.16, 9.15, 1.16)
    add_arrow(ax, 4.3, 1.16, 9.15, 1.16)
    add_arrow(ax, 6.5, 1.16, 9.15, 1.16)
    add_arrow(ax, 8.7, 1.16, 9.15, 1.16)
    add_arrow(ax, 10.95, 1.16, 11.35, 1.16)

    ax.text(
        6.75,
        3.95,
        "Drift-Adaptive Intrusion Detection for Enterprise Networks",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        6.75,
        0.22,
        "Evaluation layers: official-split benchmarking, cross-dataset transfer, latency under load, and explainability by ablation",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#475569",
    )

    plt.tight_layout()
    plt.savefig(BASE_DIR / "architecture.png", dpi=180, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
