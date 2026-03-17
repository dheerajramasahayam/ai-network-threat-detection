from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def _paper_page_one(pdf: PdfPages, results_df: pd.DataFrame, dataset_summary: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, "AI-Based Network Threat Detection Using Deep Learning for Enterprise Infrastructure", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.935, "Dheeraj Ramasahayam", fontsize=11)

    abstract = (
        "Abstract. This paper evaluates three AI models for enterprise network-threat detection "
        "on the UNSW-NB15 benchmark: Random Forest, LSTM, and Transformer. The workflow uses "
        "engineered flow-level features derived from packet and session metadata, then compares "
        "classification quality and inference latency on an official train/test split."
    )
    fig.text(0.08, 0.88, abstract, fontsize=10.5, wrap=True)

    dataset_lines = [
        f"Dataset: {dataset_summary['dataset_name']}",
        f"Train rows: {dataset_summary['train_rows']:,}",
        f"Test rows: {dataset_summary['test_rows']:,}",
        f"Classes: {dataset_summary['num_classes']}",
    ]
    fig.text(0.08, 0.80, "\n".join(dataset_lines), fontsize=10.5)

    fig.text(0.08, 0.73, "Research Questions", fontsize=12.5, fontweight="bold")
    questions = [
        "1. Can AI-driven models detect enterprise network threats with strong accuracy and practical inference speed?",
        "2. Which model family performs best on labeled network attack detection: classical ensemble, recurrent deep learning, or attention-based deep learning?",
    ]
    fig.text(0.09, 0.67, "\n".join(questions), fontsize=10.5)

    fig.text(0.08, 0.58, "Experimental Results", fontsize=12.5, fontweight="bold")
    ax = fig.add_axes([0.08, 0.28, 0.84, 0.26])
    ax.axis("off")
    table = ax.table(
        cellText=results_df.values.tolist(),
        colLabels=list(results_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    fig.text(
        0.08,
        0.18,
        "Finding. The best model by weighted F1 score is reported as the primary architecture for the repository demo and paper artifacts.",
        fontsize=10.5,
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _paper_page_two(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, "Methodology and Deployment Notes", fontsize=15, fontweight="bold")

    methodology = (
        "Feature Engineering. The pipeline derives flow duration, total bytes, total packets, packet-size estimates, "
        "byte-rate features, TCP timing signals, TTL values, loss counts, and connection-count indicators. "
        "Protocol, service, and state are encoded as categorical context features.\n\n"
        "Models. Random Forest serves as the classical ML baseline. LSTM models sequential dependencies across "
        "the engineered feature vector. Transformer applies self-attention over the same feature sequence to "
        "capture non-local interactions among traffic indicators.\n\n"
        "Comparison with Traditional IDS. Signature-driven IDS remains valuable for deterministic known patterns, "
        "but its coverage is constrained by rules. The AI models in this repository provide broader supervised "
        "classification across multiple attack categories while remaining fast enough for batch flow scoring."
    )
    fig.text(0.08, 0.84, methodology, fontsize=10.5, wrap=True)

    architecture_path = BASE_DIR / "architecture.png"
    if architecture_path.exists():
        image = plt.imread(architecture_path)
        ax = fig.add_axes([0.08, 0.46, 0.84, 0.25])
        ax.imshow(image)
        ax.axis("off")

    figure_pairs = [
        ("ROC Curve", RESULTS_DIR / "roc_curve.png", [0.08, 0.12, 0.38, 0.24]),
        ("Feature Importance", RESULTS_DIR / "feature_importance.png", [0.54, 0.12, 0.38, 0.24]),
    ]
    for title, path, rect in figure_pairs:
        if path.exists():
            ax = fig.add_axes(rect)
            ax.imshow(plt.imread(path))
            ax.axis("off")
            fig.text(rect[0], rect[1] + rect[3] + 0.01, title, fontsize=10.5, fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    results_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    with open(RESULTS_DIR / "experiment_summary.json") as handle:
        summary = json.load(handle)

    with PdfPages(BASE_DIR / "research_paper.pdf") as pdf:
        _paper_page_one(pdf, results_df, summary["dataset_summary"])
        _paper_page_two(pdf)


if __name__ == "__main__":
    main()
