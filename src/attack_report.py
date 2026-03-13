"""
attack_report.py
----------------
Per-Attack-Type Precision/Recall Breakdown for AI-NIDS.

Research gap addressed
----------------------
Published papers report 99%+ overall accuracy but hide that many models
completely fail on rare attack classes (e.g. Heartbleed: 11 samples,
Infiltration: 36 samples). This report exposes per-class true performance.

Produces
--------
  results/attack_report.md         — markdown table with per-attack metrics
  results/attack_precision_recall.png — bar chart per attack type

Usage
-----
    python3 src/attack_report.py --model models/random_forest_cicids2017.joblib
"""

from __future__ import annotations

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.preprocessing import FEATURE_COLUMNS, _CICIDS_COL_MAP


def _load_cicids(path: str) -> pd.DataFrame:
    logger.info(f"Loading CICIDS2017 with raw labels ({path}) …")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Rename to canonical
    rename = {k: v for k, v in _CICIDS_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

    # Keep the raw attack label (not binarized)
    label_col = "Label" if "Label" in df.columns else "label"
    df["_raw_label"] = df[label_col].astype(str).str.strip()
    return df


def generate_attack_report(
    dataset_path: str,
    model_path: str,
    out_md: str,
    out_png: str,
    threshold: float = 0.5,
):
    from src.model import IntrusionDetectionModel

    df = _load_cicids(dataset_path)
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feats].values
    raw_labels = df["_raw_label"].values

    # Binary ground truth
    y_true = (raw_labels != "BENIGN").astype(int)

    # Scale
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Load & predict
    logger.info(f"Loading model: {model_path}")
    model = IntrusionDetectionModel.load(model_path)
    y_prob = model.predict_proba(X_s)
    y_pred = (y_prob >= threshold).astype(int)

    # Per-attack-type breakdown
    attack_types = sorted(set(raw_labels))
    rows = []
    for attack in attack_types:
        mask = raw_labels == attack
        n = mask.sum()
        is_attack = (attack != "BENIGN")
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask]

        tp = int(((y_p == 1) & (y_t == 1)).sum())
        fp = int(((y_p == 1) & (y_t == 0)).sum())
        fn = int(((y_p == 0) & (y_t == 1)).sum())
        tn = int(((y_p == 0) & (y_t == 0)).sum())

        if is_attack:
            # For attacks: correct detections = TP
            detection_rate = round(100 * tp / max(n, 1), 2)
            false_neg_rate = round(100 * fn / max(n, 1), 2)
        else:
            # For benign: correct = TN
            detection_rate = round(100 * tn / max(n, 1), 2)
            false_neg_rate = round(100 * fp / max(n, 1), 2)

        prec = round(100 * precision_score(y_t, y_p, zero_division=0), 2)
        rec  = round(100 * recall_score(y_t, y_p, zero_division=0), 2)
        f1   = round(100 * f1_score(y_t, y_p, zero_division=0), 2)

        rows.append({
            "attack_type":     attack,
            "samples":         int(n),
            "is_attack":       is_attack,
            "detection_rate":  detection_rate,
            "miss_rate":       false_neg_rate,
            "precision":       prec,
            "recall":          rec,
            "f1":              f1,
        })
        logger.info(
            f"  {attack:<40s}  n={n:>6,}  "
            f"detect={detection_rate:>6.2f}%  miss={false_neg_rate:>6.2f}%  "
            f"f1={f1:.2f}%"
        )

    df_report = pd.DataFrame(rows).sort_values(
        ["is_attack", "samples"], ascending=[False, True]
    )

    # Markdown report
    _write_md(df_report, out_md)

    # Chart
    _write_chart(df_report, out_png)

    return df_report


def _write_md(df: pd.DataFrame, path: str):
    lines = [
        "# Per-Attack-Type Detection Report\n\n",
        "This report exposes per-class performance — the metric that 99%+ overall\n",
        "accuracy hides. Rare attacks often have detection rates << overall accuracy.\n\n",
        "| Attack Type | Samples | Detection Rate | Miss Rate | Precision | Recall | F1 |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for _, row in df.iterrows():
        flag = "🔴" if row["is_attack"] and row["miss_rate"] > 20 else (
               "🟡" if row["is_attack"] and row["miss_rate"] > 5 else "🟢")
        lines.append(
            f"| {flag} **{row['attack_type']}** "
            f"| {row['samples']:,} "
            f"| {row['detection_rate']:.2f}% "
            f"| {row['miss_rate']:.2f}% "
            f"| {row['precision']:.2f}% "
            f"| {row['recall']:.2f}% "
            f"| {row['f1']:.2f}% |\n"
        )
    lines += [
        "\n**Legend:** 🟢 Miss rate ≤ 5%  🟡 5–20%  🔴 > 20%\n\n",
        "## Key findings\n\n",
        "Models that achieve 99%+ overall accuracy often miss rare attack categories.\n",
        "This breakdown is essential for real-world deployment decisions.\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)
    logger.info(f"Attack report saved → {path}")


def _write_chart(df: pd.DataFrame, path: str):
    attacks_only = df[df["is_attack"]].sort_values("f1")
    if attacks_only.empty:
        logger.warning("No attack rows to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(attacks_only) * 0.5 + 2)))

    # Detection rate bar
    colors_det = ["#dc2626" if r < 80 else "#f59e0b" if r < 95 else "#16a34a"
                  for r in attacks_only["detection_rate"]]
    axes[0].barh(attacks_only["attack_type"], attacks_only["detection_rate"],
                 color=colors_det, edgecolor="none")
    axes[0].axvline(95, color="#64748b", linestyle="--", alpha=0.7, label="95% threshold")
    axes[0].set_xlabel("Detection Rate (%)", fontsize=11)
    axes[0].set_title("Detection Rate per Attack Type\n(🔴 < 80%  🟡 80–95%  🟢 ≥ 95%)",
                       fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, 105)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)

    # F1 bar
    colors_f1 = ["#dc2626" if r < 80 else "#f59e0b" if r < 95 else "#16a34a"
                 for r in attacks_only["f1"]]
    axes[1].barh(attacks_only["attack_type"], attacks_only["f1"],
                 color=colors_f1, edgecolor="none")
    axes[1].axvline(95, color="#64748b", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("F1 Score (%)", fontsize=11)
    axes[1].set_title("F1 Score per Attack Type", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, 105)
    axes[1].grid(axis="x", alpha=0.3)

    # Sample count annotation
    for ax, col in zip(axes, ["detection_rate", "f1"]):
        for i, (_, row) in enumerate(attacks_only.iterrows()):
            ax.text(2, i, f"n={row['samples']:,}", va="center",
                    fontsize=8, color="#475569")

    plt.suptitle("Per-Attack-Type Performance Breakdown\n"
                 "The metric that overall accuracy hides",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-attack chart saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=os.path.join(BASE_DIR, "dataset", "cicids2017.csv"))
    parser.add_argument("--model",
                        default=os.path.join(BASE_DIR, "models", "random_forest_cicids2017.joblib"),
                        help="Path to trained .joblib model")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    generate_attack_report(
        dataset_path=args.dataset,
        model_path=args.model,
        out_md=os.path.join(RESULTS_DIR, "attack_report.md"),
        out_png=os.path.join(RESULTS_DIR, "attack_precision_recall.png"),
        threshold=args.threshold,
    )
    print("✅  Per-attack report complete — see results/attack_report.md")
