from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from src.preprocessing import FEATURE_COLUMNS, _CICIDS_COL_MAP
from training.canonical_pipeline import _frame_to_array, _to_canonical

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
MODEL_PREFIX = "transfer_unsw_nsl_to_cicids"
OUTPUT_PREFIX = "transfer_unsw_nsl_to_cicids_failure_case_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-attack failure case analysis on CICIDS2017.")
    parser.add_argument("--cicids-path", default=str(BASE_DIR / "dataset" / "cicids2017.csv"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hybrid-batch-size", type=int, default=4096)
    return parser.parse_args()


def _load_models():
    signature = SignatureIDSBaseline.load(str(ARTIFACTS_DIR / f"{MODEL_PREFIX}_signature_ids.json"))
    random_forest = RandomForestThreatDetector.load(str(ARTIFACTS_DIR / f"{MODEL_PREFIX}_random_forest.joblib"))
    lstm_model = LSTMThreatDetector.load(str(ARTIFACTS_DIR / f"{MODEL_PREFIX}_lstm.pt"))
    transformer_model = TransformerThreatDetector.load(str(ARTIFACTS_DIR / f"{MODEL_PREFIX}_transformer.pt"))
    preprocessor = joblib.load(ARTIFACTS_DIR / f"{MODEL_PREFIX}_preprocessor.joblib")

    lstm_model.batch_size = max(lstm_model.batch_size, 4096)
    transformer_model.batch_size = max(transformer_model.batch_size, 4096)

    hybrid_model = DriftAwareHybridDetector(signature, random_forest, lstm_model, transformer_model)
    hybrid_model.load_state(str(ARTIFACTS_DIR / f"{MODEL_PREFIX}_hybrid.joblib"))
    return lstm_model, hybrid_model, preprocessor


def _iter_cicids_raw_chunks(path: str | Path, chunksize: int):
    usecols = list(_CICIDS_COL_MAP.keys())
    reader = pd.read_csv(
        path,
        usecols=lambda c: c.strip() in usecols,
        low_memory=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        yield chunk


def _top_attack_families(path: str | Path, chunksize: int, top_k: int) -> list[str]:
    counts: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    for chunk in _iter_cicids_raw_chunks(path, chunksize):
        labels = chunk["Label"].astype(str).str.strip()
        for label, count in labels.value_counts().items():
            normalized = label.upper()
            if normalized == "BENIGN":
                continue
            counts[normalized] += int(count)
            display_names.setdefault(normalized, label)
    top_labels = [label for label, _ in counts.most_common(top_k)]
    return [display_names[label] for label in top_labels]


def _metric_dict(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _plot_results(results_df: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(results_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, results_df["LSTM F1"], width=width, label="LSTM", color="#1d4ed8")
    ax.bar(x + width / 2, results_df["Drift-Adaptive Hybrid F1"], width=width, label="Drift-Adaptive Hybrid", color="#b45309")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Attack Type"], rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("Failure Case Analysis by Attack Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    lstm_model, hybrid_model, preprocessor = _load_models()
    attack_families = _top_attack_families(args.cicids_path, args.chunksize, args.top_k)
    attack_lookup = {family.upper(): family for family in attack_families}

    counts = {
        "LSTM": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}),
        "Drift-Adaptive Hybrid": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}),
    }
    adaptive_state = None

    for chunk in _iter_cicids_raw_chunks(args.cicids_path, args.chunksize):
        original_labels = chunk["Label"].astype(str).str.strip()
        normalized_labels = original_labels.str.upper()
        label = (normalized_labels != "BENIGN").astype(int)
        canonical = _to_canonical(chunk.rename(columns={key: value for key, value in _CICIDS_COL_MAP.items() if key in chunk.columns}), label, "cicids2017")
        X_chunk = preprocessor.transform(_frame_to_array(canonical[FEATURE_COLUMNS])).astype(np.float32)

        lstm_prob = lstm_model.predict_proba(X_chunk)[:, 1].astype(np.float32)
        hybrid_prob, adaptive_state = hybrid_model.predict_proba(
            canonical,
            X_chunk,
            batch_size=args.hybrid_batch_size,
            state=adaptive_state,
            return_state=True,
        )
        hybrid_prob = hybrid_prob[:, 1].astype(np.float32)

        lstm_pred = (lstm_prob >= 0.5).astype(np.int32)
        hybrid_pred = (hybrid_prob >= 0.5).astype(np.int32)

        for attack_key, attack_name in attack_lookup.items():
            mask = normalized_labels.isin(["BENIGN", attack_key]).to_numpy()
            if not np.any(mask):
                continue
            y_true = (normalized_labels.to_numpy()[mask] == attack_key).astype(np.int32)
            support = int(np.sum(y_true))
            if support == 0:
                continue

            for model_name, predictions in [("LSTM", lstm_pred), ("Drift-Adaptive Hybrid", hybrid_pred)]:
                y_pred = predictions[mask]
                stats = counts[model_name][attack_name]
                stats["tp"] += int(np.sum((y_true == 1) & (y_pred == 1)))
                stats["fp"] += int(np.sum((y_true == 0) & (y_pred == 1)))
                stats["fn"] += int(np.sum((y_true == 1) & (y_pred == 0)))
                stats["support"] += support

    rows = []
    for attack_name in attack_families:
        lstm_stats = counts["LSTM"][attack_name]
        hybrid_stats = counts["Drift-Adaptive Hybrid"][attack_name]
        lstm_metrics = _metric_dict(lstm_stats["tp"], lstm_stats["fp"], lstm_stats["fn"])
        hybrid_metrics = _metric_dict(hybrid_stats["tp"], hybrid_stats["fp"], hybrid_stats["fn"])
        rows.append(
            {
                "Attack Type": attack_name,
                "Support": int(hybrid_stats["support"]),
                "LSTM Precision": round(lstm_metrics["precision"], 4),
                "LSTM Recall": round(lstm_metrics["recall"], 4),
                "LSTM F1": round(lstm_metrics["f1"], 4),
                "Drift-Adaptive Hybrid Precision": round(hybrid_metrics["precision"], 4),
                "Drift-Adaptive Hybrid Recall": round(hybrid_metrics["recall"], 4),
                "Drift-Adaptive Hybrid F1": round(hybrid_metrics["f1"], 4),
                "Hybrid Gain": round(hybrid_metrics["f1"] - lstm_metrics["f1"], 4),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["Hybrid Gain", "Support"], ascending=[False, False]).reset_index(drop=True)
    csv_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.csv"
    md_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.md"
    png_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.png"
    json_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.json"

    results_df.to_csv(csv_path, index=False)
    _plot_results(results_df, png_path)

    lines = [
        "# CICIDS2017 Failure Case Analysis\n\n",
        "Each attack family is evaluated as a binary task against benign traffic only. Other attack families are excluded from the family-specific score so the table reflects detection quality for that family rather than multi-class attribution.\n\n",
        "| Attack Type | Support | LSTM Precision | LSTM Recall | LSTM F1 | Drift-Adaptive Hybrid Precision | Drift-Adaptive Hybrid Recall | Drift-Adaptive Hybrid F1 | Hybrid Gain |\n",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['Attack Type']} | {int(row['Support'])} | {row['LSTM Precision']:.4f} | {row['LSTM Recall']:.4f} | "
            f"{row['LSTM F1']:.4f} | {row['Drift-Adaptive Hybrid Precision']:.4f} | {row['Drift-Adaptive Hybrid Recall']:.4f} | "
            f"{row['Drift-Adaptive Hybrid F1']:.4f} | {row['Hybrid Gain']:+.4f} |\n"
        )
    md_path.write_text("".join(lines))

    summary = {
        "attack_families": attack_families,
        "results": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
