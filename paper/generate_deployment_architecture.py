from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "paper" / "deployment_architecture.png"


def _box(ax, xy, width, height, text, facecolor):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#1f2937",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        wrap=True,
    )


def _arrow(ax, start, end, text=""):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#374151")
    ax.add_patch(arrow)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, text, ha="center", va="center", fontsize=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.03, 0.30), 0.13, 0.36, "Enterprise\nTraffic\nMirrors", "#dbeafe")
    _box(ax, (0.21, 0.30), 0.14, 0.36, "Flow Collector\nZeek / NetFlow", "#dcfce7")
    _box(ax, (0.40, 0.30), 0.15, 0.36, "Kafka Topic\nor File Stream", "#fef3c7")
    _box(ax, (0.60, 0.30), 0.15, 0.36, "Canonical Feature\nExtraction", "#fee2e2")
    _box(ax, (0.80, 0.34), 0.16, 0.28, "Drift-Adaptive IDS\n0.263 ms/flow\nfull-stream avg.", "#e9d5ff")

    _arrow(ax, (0.16, 0.48), (0.21, 0.48), "mirrored flows")
    _arrow(ax, (0.35, 0.48), (0.40, 0.48), "events")
    _arrow(ax, (0.55, 0.48), (0.60, 0.48), "stream batches")
    _arrow(ax, (0.75, 0.48), (0.80, 0.48), "41 features")

    _box(ax, (0.58, 0.03), 0.16, 0.15, "Security Alert\nthreshold + label", "#fde68a")
    _box(ax, (0.79, 0.03), 0.17, 0.15, "SIEM / SOC\ntriage and response", "#bfdbfe")
    _arrow(ax, (0.88, 0.34), (0.66, 0.18), "attack probability")
    _arrow(ax, (0.74, 0.105), (0.79, 0.105), "alert event")

    ax.text(
        0.50,
        0.88,
        "Production Deployment Scenario for the Drift-Adaptive IDS",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.50,
        0.82,
        "Streaming ingress, canonical feature alignment, low-latency inference, and SOC handoff.",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
