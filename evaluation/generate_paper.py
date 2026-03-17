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
ADVANCED_SUMMARY = RESULTS_DIR / "advanced_experiment_summary.json"


def _add_table(fig, rect: list[float], df: pd.DataFrame, title: str, font_size: float = 8.5) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    fig.text(rect[0], rect[1] + rect[3] + 0.015, title, fontsize=11.5, fontweight="bold")
    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.35)


def _add_image(fig, rect: list[float], path: Path, title: str) -> None:
    if not path.exists():
        return
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(path))
    ax.axis("off")
    fig.text(rect[0], rect[1] + rect[3] + 0.01, title, fontsize=10.5, fontweight="bold")


def _load_table(prefix: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{prefix}_model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


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
    _add_table(fig, [0.08, 0.28, 0.84, 0.26], results_df, "UNSW-NB15 Benchmark Results", font_size=9.0)

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
        _add_image(fig, rect, path, title)

    pdf.savefig(fig)
    plt.close(fig)


def _advanced_page_one(pdf: PdfPages, summary: dict, unsw_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, "AI-Based Network Threat Detection Using Deep Learning for Enterprise Infrastructure", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.935, "Advanced Multi-Dataset Research Suite", fontsize=11)

    abstract = (
        "Abstract. This upgraded study benchmarks a flow-signature IDS baseline, Random Forest, LSTM, "
        "Transformer, and a proposed Drift-Aware Hybrid detector across full official UNSW-NB15 and NSL-KDD "
        "splits, then measures cross-dataset transfer on external CICIDS2017 traffic. The paper extends the "
        "original benchmark with a transparent traditional baseline, richer evaluation on diverse datasets, "
        "latency-under-load measurements, and explainability validated by feature ablation."
    )
    fig.text(0.08, 0.885, abstract, fontsize=10.3, wrap=True)

    primary = summary["primary_official_split"]
    secondary = summary["secondary_official_split"]
    transfer = summary["external_transfer_split"]
    dataset_lines = [
        f"UNSW-NB15 official split: {primary['train_rows']:,} train / {primary['test_rows']:,} test",
        f"NSL-KDD official split: {secondary['train_rows']:,} train / {secondary['test_rows']:,} test",
        f"Transfer setup: {transfer['train_rows']:,} train from UNSW+NSL, {transfer['test_rows']:,} external CICIDS2017 rows",
        f"Canonical feature space: {primary['feature_count']} flow features",
        f"Best models: UNSW={summary['best_models']['official_unsw']}, NSL-KDD={summary['best_models']['official_nsl_kdd']}, Transfer={summary['best_models']['transfer_unsw_nsl_to_cicids']}",
    ]
    fig.text(0.08, 0.775, "\n".join(dataset_lines), fontsize=10.2)

    fig.text(0.08, 0.67, "Research Questions", fontsize=12.5, fontweight="bold")
    questions = [
        "1. How does a rule-based IDS baseline compare with classical ML, deep learning, and the proposed hybrid detector?",
        "2. Which model retains the best performance on full official splits and on external cross-dataset transfer?",
        "3. Can the strongest models remain practical under online load while still being explainable through ablation?",
    ]
    fig.text(0.09, 0.60, "\n".join(questions), fontsize=10.2)

    contributions = [
        "Contributions",
        "- Flow-signature IDS baseline over the same canonical features as the learning models",
        "- Proposed Drift-Aware Hybrid that fuses signatures, RF, LSTM, Transformer, and drift scoring",
        "- Full official-split evaluation on UNSW-NB15 and NSL-KDD plus external CICIDS2017 transfer",
        "- Online latency-under-load benchmarking and explainability validated by ablation",
    ]
    fig.text(0.08, 0.46, "\n".join(contributions), fontsize=10.2)

    _add_table(fig, [0.08, 0.11, 0.84, 0.24], unsw_df, "Official UNSW-NB15 Results")
    pdf.savefig(fig)
    plt.close(fig)


def _advanced_page_two(pdf: PdfPages, nsl_df: pd.DataFrame | None, transfer_df: pd.DataFrame | None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, "Extended Experiments", fontsize=15, fontweight="bold")

    methodology = (
        "Methodology. All datasets are aligned into a canonical 41-feature flow representation. The traditional "
        "baseline is implemented as an explicit signature engine over those features, allowing a fair rule-based "
        "comparison on public flow datasets that do not ship packet payloads for direct Snort or Suricata replay. "
        "The proposed Drift-Aware Hybrid learns when to trust signatures, tree ensembles, and deep models under "
        "distribution shift, using an Isolation Forest drift signal as an additional decision feature."
    )
    fig.text(0.08, 0.87, methodology, fontsize=10.3, wrap=True)

    architecture_path = BASE_DIR / "architecture.png"
    _add_image(fig, [0.08, 0.57, 0.84, 0.20], architecture_path, "System Architecture")

    if nsl_df is not None:
        _add_table(fig, [0.08, 0.29, 0.84, 0.20], nsl_df, "Official NSL-KDD Results", font_size=8.2)
    if transfer_df is not None:
        _add_table(fig, [0.08, 0.05, 0.84, 0.18], transfer_df, "Cross-Dataset Transfer Results", font_size=8.0)

    pdf.savefig(fig)
    plt.close(fig)


def _advanced_page_three(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, "Operational Analysis", fontsize=15, fontweight="bold")

    notes = (
        "Operational Evaluation. The repository includes three post-accuracy analyses aimed at deployment realism: "
        "latency under varying batch sizes, explainability validated through top-feature ablation, and artifact "
        "export for deterministic demo replay. These assets make the project more defensible as a reproducible "
        "benchmark and stronger as a systems-oriented security paper."
    )
    fig.text(0.08, 0.90, notes, fontsize=10.3, wrap=True)

    _add_image(fig, [0.08, 0.54, 0.38, 0.26], RESULTS_DIR / "official_unsw_roc_curve.png", "Official UNSW ROC")
    _add_image(fig, [0.54, 0.54, 0.38, 0.26], RESULTS_DIR / "official_unsw_latency_under_load.png", "Latency Under Load")
    _add_image(fig, [0.08, 0.18, 0.38, 0.26], RESULTS_DIR / "official_unsw_feature_importance.png", "Feature Importance")
    _add_image(fig, [0.54, 0.18, 0.38, 0.26], RESULTS_DIR / "official_unsw_explainability_ablation.png", "Explainability Ablation")

    pdf.savefig(fig)
    plt.close(fig)


def _generate_advanced_paper() -> None:
    summary = json.loads(ADVANCED_SUMMARY.read_text())
    unsw_df = _load_table("official_unsw")
    nsl_df = _load_table("official_nsl_kdd")
    transfer_df = _load_table("transfer_unsw_nsl_to_cicids")
    if unsw_df is None:
        raise FileNotFoundError("Advanced paper generation requires results/official_unsw_model_comparison.csv")

    with PdfPages(BASE_DIR / "research_paper.pdf") as pdf:
        _advanced_page_one(pdf, summary, unsw_df)
        _advanced_page_two(pdf, nsl_df, transfer_df)
        _advanced_page_three(pdf)


def _generate_legacy_paper() -> None:
    results_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    with open(RESULTS_DIR / "experiment_summary.json") as handle:
        summary = json.load(handle)

    with PdfPages(BASE_DIR / "research_paper.pdf") as pdf:
        _paper_page_one(pdf, results_df, summary["dataset_summary"])
        _paper_page_two(pdf)


def main() -> None:
    if ADVANCED_SUMMARY.exists():
        _generate_advanced_paper()
        return
    _generate_legacy_paper()


if __name__ == "__main__":
    main()
