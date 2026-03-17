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
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 9.8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    add_box(ax, 0.3, 0.95, "UNSW-NB15 Traffic", "#0f766e")
    add_box(ax, 2.45, 0.95, "Feature Engineering", "#1d4ed8")
    add_box(ax, 4.6, 0.95, "RF / LSTM / Transformer", "#b45309")
    add_box(ax, 6.75, 0.95, "Threat Classification", "#7c3aed")
    add_box(ax, 8.9, 0.95, "Security Alert", "#be123c")

    add_arrow(ax, 2.1, 1.26, 2.45, 1.26)
    add_arrow(ax, 4.25, 1.26, 4.6, 1.26)
    add_arrow(ax, 6.4, 1.26, 6.75, 1.26)
    add_arrow(ax, 8.55, 1.26, 8.9, 1.26)

    ax.text(
        4.9,
        2.08,
        "AI-Based Network Threat Detection Using Deep Learning for Enterprise Infrastructure",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        4.9,
        0.42,
        "Flow duration, packet rates, byte volume, TTL, TCP timing, and connection-behavior features",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#475569",
    )

    plt.tight_layout()
    plt.savefig(BASE_DIR / "architecture.png", dpi=180, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
