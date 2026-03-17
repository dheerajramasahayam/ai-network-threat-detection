from __future__ import annotations

import argparse
import json
import sys
import time
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

from evaluation.reporting import (
    classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    write_results_table,
)
from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from src.preprocessing import FEATURE_COLUMNS
from training.canonical_pipeline import _frame_to_array, iter_cicids_canonical_chunks

CLASS_NAMES = ["Benign", "Attack"]
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
PREFIX = "transfer_unsw_nsl_to_cicids"


def _latency_wrapper(callable_fn, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = callable_fn()
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings))


def _build_probability_frame(prob_attack: np.ndarray) -> np.ndarray:
    prob_attack = np.asarray(prob_attack, dtype=np.float32)
    return np.column_stack([1.0 - prob_attack, prob_attack])


def _load_models():
    signature = SignatureIDSBaseline.load(str(ARTIFACTS_DIR / f"{PREFIX}_signature_ids.json"))
    random_forest = RandomForestThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_random_forest.joblib"))
    lstm_model = LSTMThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_lstm.pt"))
    transformer_model = TransformerThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_transformer.pt"))
    preprocessor = joblib.load(ARTIFACTS_DIR / f"{PREFIX}_preprocessor.joblib")

    # Increase inference batch sizes for the full external evaluation.
    lstm_model.batch_size = max(lstm_model.batch_size, 4096)
    transformer_model.batch_size = max(transformer_model.batch_size, 4096)

    hybrid_model = DriftAwareHybridDetector(signature, random_forest, lstm_model, transformer_model)
    hybrid_model.load_state(str(ARTIFACTS_DIR / f"{PREFIX}_hybrid.joblib"))
    return signature, random_forest, lstm_model, transformer_model, hybrid_model, preprocessor


def _measure_latencies(
    signature,
    random_forest,
    lstm_model,
    transformer_model,
    hybrid_model,
    raw_sample: pd.DataFrame,
    scaled_sample: np.ndarray,
    hybrid_batch_size: int,
) -> dict[str, float]:
    sample_size = len(raw_sample)
    return {
        "Signature IDS": (_latency_wrapper(lambda: signature.predict_proba(raw_sample)) / sample_size) * 1000.0,
        "Random Forest": (_latency_wrapper(lambda: random_forest.predict_proba(scaled_sample)) / sample_size) * 1000.0,
        "LSTM": (_latency_wrapper(lambda: lstm_model.predict_proba(scaled_sample)) / sample_size) * 1000.0,
        "Transformer": (_latency_wrapper(lambda: transformer_model.predict_proba(scaled_sample)) / sample_size) * 1000.0,
        "Drift-Adaptive Hybrid": (
            _latency_wrapper(lambda: hybrid_model.predict_proba(raw_sample, scaled_sample, batch_size=hybrid_batch_size)) / sample_size
        ) * 1000.0,
        "Static Hybrid": (
            _latency_wrapper(lambda: hybrid_model.predict_proba_static(raw_sample, scaled_sample)) / sample_size
        ) * 1000.0,
    }


def _write_online_adaptation_report(
    static_metrics: dict[str, float],
    adaptive_metrics: dict[str, float],
    static_latency: float,
    adaptive_latency: float,
    trace_df: pd.DataFrame,
) -> None:
    rows = pd.DataFrame(
        [
            {
                "Variant": "Static Hybrid",
                "Accuracy": round(static_metrics["accuracy"] * 100, 2),
                "Precision": round(static_metrics["precision"] * 100, 2),
                "Recall": round(static_metrics["recall"] * 100, 2),
                "F1 Score": round(static_metrics["f1_score"] * 100, 2),
                "ROC AUC": round(static_metrics["roc_auc"], 4),
                "Latency (ms/flow)": round(static_latency, 4),
            },
            {
                "Variant": "Online Drift-Adaptive Hybrid",
                "Accuracy": round(adaptive_metrics["accuracy"] * 100, 2),
                "Precision": round(adaptive_metrics["precision"] * 100, 2),
                "Recall": round(adaptive_metrics["recall"] * 100, 2),
                "F1 Score": round(adaptive_metrics["f1_score"] * 100, 2),
                "ROC AUC": round(adaptive_metrics["roc_auc"], 4),
                "Latency (ms/flow)": round(adaptive_latency, 4),
            },
        ]
    )
    rows.to_csv(RESULTS_DIR / f"{PREFIX}_online_drift_adaptation.csv", index=False)

    md_lines = [
        "# UNSW+NSL -> CICIDS2017 Online Drift Adaptation\n\n",
        "Evaluation on the full CICIDS2017 external corpus.\n\n",
        "| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |\n",
        "| --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for _, row in rows.iterrows():
        md_lines.append(
            f"| {row['Variant']} | {row['Accuracy']:.2f}% | {row['Precision']:.2f}% | "
            f"{row['Recall']:.2f}% | {row['F1 Score']:.2f}% | {row['ROC AUC']:.4f} | "
            f"{row['Latency (ms/flow)']:.4f} |\n"
        )
    md_lines.extend(
        [
            "\nAdaptive batches summarize mean drift score, adaptation alpha, and the online ensemble weights.\n",
            f"\nAverage adaptation alpha: `{trace_df['adaptation_alpha'].mean():.4f}`\n",
            f"\nPeak adaptation alpha: `{trace_df['adaptation_alpha'].max():.4f}`\n",
        ]
    )
    (RESULTS_DIR / f"{PREFIX}_online_drift_adaptation.md").write_text("".join(md_lines))

    trace_df.to_csv(RESULTS_DIR / f"{PREFIX}_online_drift_trace.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(trace_df.index, trace_df["ema_drift_score"], label="EMA drift", color="#b45309", linewidth=2)
    axes[0].plot(trace_df.index, trace_df["adaptation_alpha"], label="Adaptation alpha", color="#1d4ed8", linewidth=2)
    axes[0].set_title("Online Drift State")
    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(trace_df.index, trace_df["random_forest_weight"], label="RF", color="#0f766e", linewidth=2)
    axes[1].plot(trace_df.index, trace_df["lstm_weight"], label="LSTM", color="#1d4ed8", linewidth=2)
    axes[1].plot(trace_df.index, trace_df["transformer_weight"], label="Transformer", color="#b45309", linewidth=2)
    axes[1].plot(trace_df.index, trace_df["stacked_meta_weight"], label="Meta", color="#7c3aed", linewidth=2)
    axes[1].set_title("Adaptive Ensemble Weights")
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Weight")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{PREFIX}_online_drift_adaptation.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-scale external transfer evaluation on CICIDS2017.")
    parser.add_argument("--cicids-path", default=str(BASE_DIR / "dataset" / "cicids2017.csv"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--hybrid-batch-size", type=int, default=4096)
    parser.add_argument("--max-chunks", type=int, default=0, help="Use 0 to process the full dataset.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    (
        signature,
        random_forest,
        lstm_model,
        transformer_model,
        hybrid_model,
        preprocessor,
    ) = _load_models()

    probability_store: dict[str, list[np.ndarray]] = {
        "Signature IDS": [],
        "Random Forest": [],
        "LSTM": [],
        "Transformer": [],
        "Drift-Adaptive Hybrid": [],
        "Static Hybrid": [],
    }
    labels = []
    trace_frames = []
    adaptive_state = None
    latency_sample_raw = None
    latency_sample_scaled = None
    class_counts = {0: 0, 1: 0}
    total_rows = 0

    for chunk_index, chunk_df in enumerate(
        iter_cicids_canonical_chunks(args.cicids_path, chunksize=args.chunksize),
        start=1,
    ):
        raw_chunk = chunk_df.reset_index(drop=True)
        y_chunk = raw_chunk["label"].to_numpy(dtype=np.int32)
        X_chunk = preprocessor.transform(_frame_to_array(raw_chunk[FEATURE_COLUMNS])).astype(np.float32)

        if latency_sample_raw is None:
            latency_sample_raw = raw_chunk.iloc[: min(2048, len(raw_chunk))].reset_index(drop=True)
            latency_sample_scaled = X_chunk[: len(latency_sample_raw)]

        signature_prob = signature.predict_proba(raw_chunk)[:, 1].astype(np.float32)
        rf_prob = random_forest.predict_proba(X_chunk)[:, 1].astype(np.float32)
        lstm_prob = lstm_model.predict_proba(X_chunk)[:, 1].astype(np.float32)
        transformer_prob = transformer_model.predict_proba(X_chunk)[:, 1].astype(np.float32)
        static_hybrid_prob = hybrid_model.predict_proba_static(raw_chunk, X_chunk)[:, 1].astype(np.float32)
        adaptive_prob, adaptive_state = hybrid_model.predict_proba(
            raw_chunk,
            X_chunk,
            batch_size=args.hybrid_batch_size,
            state=adaptive_state,
            return_state=True,
        )
        adaptive_prob = adaptive_prob[:, 1].astype(np.float32)

        probability_store["Signature IDS"].append(signature_prob)
        probability_store["Random Forest"].append(rf_prob)
        probability_store["LSTM"].append(lstm_prob)
        probability_store["Transformer"].append(transformer_prob)
        probability_store["Drift-Adaptive Hybrid"].append(adaptive_prob)
        probability_store["Static Hybrid"].append(static_hybrid_prob)
        labels.append(y_chunk)
        trace_frames.append(hybrid_model.last_adaptation_trace.copy())

        class_counts[0] += int(np.sum(y_chunk == 0))
        class_counts[1] += int(np.sum(y_chunk == 1))
        total_rows += len(raw_chunk)
        print(f"processed_rows={total_rows} chunk={chunk_index}")
        if args.max_chunks and chunk_index >= args.max_chunks:
            break

    y_true = np.concatenate(labels)
    latencies = _measure_latencies(
        signature,
        random_forest,
        lstm_model,
        transformer_model,
        hybrid_model,
        latency_sample_raw,
        latency_sample_scaled,
        args.hybrid_batch_size,
    )

    metrics_by_model = {}
    probability_map = {}
    predictions_by_model = {}
    for model_name in ["Signature IDS", "Random Forest", "LSTM", "Transformer", "Drift-Adaptive Hybrid"]:
        prob_attack = np.concatenate(probability_store[model_name]).astype(np.float32)
        y_prob = _build_probability_frame(prob_attack)
        y_pred = (prob_attack >= 0.5).astype(np.int32)
        metrics = classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
        metrics["latency_ms_per_flow"] = float(latencies[model_name])
        metrics_by_model[model_name] = metrics
        probability_map[model_name] = (y_true, y_prob)
        predictions_by_model[model_name] = y_pred

    static_prob_attack = np.concatenate(probability_store["Static Hybrid"]).astype(np.float32)
    static_prob = _build_probability_frame(static_prob_attack)
    static_pred = (static_prob_attack >= 0.5).astype(np.int32)
    static_metrics = classification_metrics(y_true, static_pred, static_prob, CLASS_NAMES)
    static_metrics["latency_ms_per_flow"] = float(latencies["Static Hybrid"])

    results_df = write_results_table(
        metrics_by_model,
        RESULTS_DIR / f"{PREFIX}_model_comparison.csv",
        RESULTS_DIR / f"{PREFIX}_model_comparison.md",
    )
    best_model_name = str(results_df.iloc[0]["Model"])
    plot_confusion_matrix(
        y_true,
        predictions_by_model[best_model_name],
        CLASS_NAMES,
        RESULTS_DIR / f"{PREFIX}_confusion_matrix.png",
        f"UNSW+NSL -> CICIDS2017 Full: {best_model_name}",
    )
    plot_roc_curves(probability_map, CLASS_NAMES, RESULTS_DIR / f"{PREFIX}_roc_curve.png")

    trace_df = pd.concat(trace_frames, ignore_index=True)
    _write_online_adaptation_report(
        static_metrics=static_metrics,
        adaptive_metrics=metrics_by_model["Drift-Adaptive Hybrid"],
        static_latency=latencies["Static Hybrid"],
        adaptive_latency=latencies["Drift-Adaptive Hybrid"],
        trace_df=trace_df,
    )

    summary = {
        "name": "UNSW+NSL -> CICIDS2017",
        "test_rows": int(total_rows),
        "test_distribution": class_counts,
        "results": metrics_by_model,
        "static_hybrid": static_metrics,
    }
    (RESULTS_DIR / f"{PREFIX}_full_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
