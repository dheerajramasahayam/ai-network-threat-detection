"""
benchmark.py
------------
Cross-Dataset Generalization Benchmark for AI-NIDS.

Research gap addressed
----------------------
Existing papers evaluate models on multiple datasets SEPARATELY.
Nobody trains on a merged corpus (A+B+C) and holds out a 4th dataset
as the true unseen test. This measures real-world generalizability —
how well the model detects attacks it has never seen in training.

Protocol
--------
For each source dataset in {CICIDS2017, UNSW-NB15, NSL-KDD, NF-ToN-IoT}:
  1. Train on the OTHER 3 sources combined  (leave-one-out)
  2. Test on the held-out source
  3. Report accuracy, precision, recall, F1, AUC

Produces results/cross_dataset_benchmark.md and
         results/cross_dataset_heatmap.png

Usage
-----
    python3 src/benchmark.py --combined dataset/combined.csv
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.preprocessing import FEATURE_COLUMNS

SOURCES = ["cicids2017", "unsw_nb15", "nsl_kdd", "nf_ton_iot"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_combined(path: str) -> pd.DataFrame:
    logger.info(f"Loading combined.csv ({path}) …")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    return df


def _get_available_features(df: pd.DataFrame) -> list[str]:
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return available


def _balance(X: np.ndarray, y: np.ndarray, max_per_class: int = 150_000):
    """Undersample each class to max_per_class for speed."""
    df = pd.DataFrame(X)
    df["_y"] = y
    parts = []
    for label in [0, 1]:
        subset = df[df["_y"] == label]
        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=42)
        parts.append(subset)
    balanced = pd.concat(parts).sample(frac=1, random_state=42)
    return balanced.drop("_y", axis=1).values, balanced["_y"].values


def _evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc":       round(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5, 4),
    }


# ── main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(combined_path: str) -> pd.DataFrame:
    df = _load_combined(combined_path)
    feats = _get_available_features(df)
    logger.info(f"Using {len(feats)} features.")

    if "_source" not in df.columns:
        raise ValueError("combined.csv must contain '_source' column.")
    if "label" not in df.columns and "Label" not in df.columns:
        raise ValueError("combined.csv must contain 'label' or 'Label' column.")
    label_col = "label" if "label" in df.columns else "Label"

    # Encode labels
    col = df[label_col]
    df["_y"] = (col if pd.api.types.is_numeric_dtype(col)
                else col.astype(str).str.upper() != "BENIGN").astype(int)

    df = df.dropna(subset=feats + ["_y", "_source"])

    results = []

    for hold_out_src in SOURCES:
        mask_test  = df["_source"] == hold_out_src
        mask_train = ~mask_test

        if mask_test.sum() == 0:
            logger.warning(f"No rows for source '{hold_out_src}' — skipping.")
            continue

        X_train_df = df.loc[mask_train, feats]
        y_train     = df.loc[mask_train, "_y"].values
        X_test_df   = df.loc[mask_test,  feats]
        y_test      = df.loc[mask_test,  "_y"].values

        # Fill NaNs with column median from train
        medians = X_train_df.median()
        X_train_df = X_train_df.fillna(medians)
        X_test_df  = X_test_df.fillna(medians)

        logger.info(
            f"\n{'='*55}\n"
            f"Hold-out: {hold_out_src}  ({mask_test.sum():,} test rows)\n"
            f"Training: {mask_train.sum():,} rows"
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_df.values)
        X_test_s  = scaler.transform(X_test_df.values)

        X_train_b, y_train_b = _balance(X_train_s, y_train)
        logger.info(f"Balanced train: {len(X_train_b):,} samples")

        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X_train_b, y_train_b)
        train_sec = round(time.time() - t0, 1)
        logger.info(f"Trained in {train_sec}s")

        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        metrics = _evaluate(y_test, y_pred, y_prob)
        metrics["hold_out"] = hold_out_src
        metrics["train_sec"] = train_sec
        metrics["test_rows"] = int(mask_test.sum())
        results.append(metrics)

        logger.info(
            f"  acc={metrics['accuracy']}%  "
            f"prec={metrics['precision']}%  "
            f"rec={metrics['recall']}%  "
            f"f1={metrics['f1']}%  "
            f"auc={metrics['auc']}"
        )

    df_results = pd.DataFrame(results).set_index("hold_out")
    return df_results


def save_markdown_report(df_results: pd.DataFrame, out_path: str):
    lines = [
        "# Cross-Dataset Generalization Benchmark\n\n",
        "**Protocol**: Leave-one-out — trained on 3 sources, tested on unseen 4th source.\n",
        "This tests true generalization to novel network environments.\n\n",
        "| Hold-Out Dataset | Test Rows | Accuracy | Precision | Recall | F1 | AUC | Train (s) |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for src, row in df_results.iterrows():
        lines.append(
            f"| **{src}** | {row['test_rows']:,} "
            f"| {row['accuracy']}% | {row['precision']}% "
            f"| {row['recall']}% | {row['f1']}% "
            f"| {row['auc']} | {row['train_sec']}s |\n"
        )
    avg = df_results[["accuracy","precision","recall","f1","auc"]].mean()
    lines += [
        f"\n**Average across all hold-outs**: "
        f"acc={avg['accuracy']:.2f}%  "
        f"prec={avg['precision']:.2f}%  "
        f"rec={avg['recall']:.2f}%  "
        f"f1={avg['f1']:.2f}%  "
        f"auc={avg['auc']:.4f}\n\n",
        "## Research significance\n\n",
        "Most published models report single-dataset accuracy. "
        "Cross-dataset generalization directly measures how well the model "
        "transfers to unseen network environments — a critical practical metric "
        "ignored by virtually all existing open-source NIDS projects.\n",
    ]
    with open(out_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Markdown report saved → {out_path}")


def save_heatmap(df_results: pd.DataFrame, out_path: str):
    metrics_plot = df_results[["accuracy", "precision", "recall", "f1"]].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        metrics_plot.astype(float),
        annot=True, fmt=".1f", cmap="YlGn",
        vmin=50, vmax=100,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Score (%)"},
    )
    ax.set_title(
        "Cross-Dataset Generalization Benchmark\n"
        "(Train on 3 sources → Test on unseen 4th)\n"
        "Higher = better generalization",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Held-out test source", fontsize=11)
    ax.set_xlabel("Metric", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined",
                        default=os.path.join(BASE_DIR, "dataset", "combined.csv"))
    args = parser.parse_args()

    results = run_benchmark(args.combined)
    print("\n" + results.to_string())

    save_markdown_report(results,
        os.path.join(RESULTS_DIR, "cross_dataset_benchmark.md"))
    save_heatmap(results,
        os.path.join(RESULTS_DIR, "cross_dataset_heatmap.png"))
    print("\n✅  Benchmark complete — results in results/")
