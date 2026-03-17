from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.reporting import classification_metrics
from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from src.preprocessing import FEATURE_COLUMNS, _CICIDS_COL_MAP
from training.canonical_pipeline import _frame_to_array, _to_canonical, iter_cicids_canonical_chunks

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
PREFIX = "transfer_unsw_nsl_to_cicids"
CLASS_NAMES = ["Benign", "Attack"]


def _load_models():
    signature = SignatureIDSBaseline.load(str(ARTIFACTS_DIR / f"{PREFIX}_signature_ids.json"))
    random_forest = RandomForestThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_random_forest.joblib"))
    lstm_model = LSTMThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_lstm.pt"))
    transformer_model = TransformerThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_transformer.pt"))
    preprocessor = joblib.load(ARTIFACTS_DIR / f"{PREFIX}_preprocessor.joblib")

    lstm_model.batch_size = max(lstm_model.batch_size, 4096)
    transformer_model.batch_size = max(transformer_model.batch_size, 4096)

    hybrid_model = DriftAwareHybridDetector(signature, random_forest, lstm_model, transformer_model)
    hybrid_model.load_state(str(ARTIFACTS_DIR / f"{PREFIX}_hybrid.joblib"))
    return signature, random_forest, lstm_model, transformer_model, hybrid_model, preprocessor


def _kafka_stream(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    max_messages: int,
) -> Iterator[pd.DataFrame]:
    try:
        from kafka import KafkaConsumer
    except Exception as exc:  # pragma: no cover - import depends on env
        raise RuntimeError(
            "Kafka source requested but kafka-python is not installed. "
            "Install kafka-python or use --source file."
        ) from exc

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    buffered_rows = []
    try:
        for index, message in enumerate(consumer, start=1):
            payload = message.value
            buffered_rows.append(payload)
            if len(buffered_rows) >= 2000:
                yield pd.DataFrame(buffered_rows)
                buffered_rows = []
            if max_messages and index >= max_messages:
                break
    finally:
        consumer.close()

    if buffered_rows:
        yield pd.DataFrame(buffered_rows)


def _stream_source(args: argparse.Namespace) -> Iterator[pd.DataFrame]:
    if args.source == "file":
        for chunk in iter_cicids_canonical_chunks(args.cicids_path, chunksize=args.chunksize):
            yield chunk
    else:
        yield from _kafka_stream(
            bootstrap_servers=args.kafka_bootstrap_servers,
            topic=args.kafka_topic,
            group_id=args.kafka_group_id,
            max_messages=args.kafka_max_messages,
        )


def _normalize_stream_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = normalized.columns.astype(str).str.strip()
    rename_map = {key: value for key, value in _CICIDS_COL_MAP.items() if key in normalized.columns}
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "label" in normalized.columns:
        label = pd.to_numeric(normalized["label"], errors="coerce").fillna(0).astype(int)
        return _to_canonical(normalized, label, "stream")
    if "Label" in normalized.columns:
        label = (normalized["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
        return _to_canonical(normalized, label, "stream")
    raise ValueError("Streaming input must include either 'label' or 'Label'.")


def _plot_timeline(timeline_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(timeline_df["batch"], timeline_df["f1_score"], label="Window F1", linewidth=2, color="#1d4ed8")
    axes[0].plot(timeline_df["batch"], timeline_df["accuracy"], label="Window Accuracy", linewidth=2, color="#0f766e")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Real-Time Streaming Evaluation Timeline")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(timeline_df["batch"], timeline_df["ema_drift_score"], label="EMA drift", linewidth=2, color="#b45309")
    axes[1].plot(timeline_df["batch"], timeline_df["adaptation_alpha"], label="Adaptation alpha", linewidth=2, color="#7c3aed")
    axes[1].set_xlabel("Streaming Batch")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time streaming evaluation with online drift adaptation.")
    parser.add_argument("--source", choices=["file", "kafka"], default="file")
    parser.add_argument("--cicids-path", default=str(BASE_DIR / "dataset" / "cicids2017.csv"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-chunks", type=int, default=0, help="Use 0 for all chunks.")
    parser.add_argument("--hybrid-batch-size", type=int, default=4096)

    parser.add_argument("--kafka-bootstrap-servers", default="localhost:9092")
    parser.add_argument("--kafka-topic", default="network_flows")
    parser.add_argument("--kafka-group-id", default="drift-adaptive-eval")
    parser.add_argument("--kafka-max-messages", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    (
        _signature,
        _random_forest,
        _lstm_model,
        _transformer_model,
        hybrid_model,
        preprocessor,
    ) = _load_models()

    stream_state = None
    timeline_rows = []
    y_all = []
    pred_all = []
    prob_all = []
    total_rows = 0

    for batch_index, chunk_df in enumerate(_stream_source(args), start=1):
        raw_chunk = _normalize_stream_frame(chunk_df).reset_index(drop=True)
        y_chunk = raw_chunk["label"].astype(np.int32).to_numpy()
        X_chunk = preprocessor.transform(_frame_to_array(raw_chunk[FEATURE_COLUMNS])).astype(np.float32)

        chunk_prob, stream_state = hybrid_model.predict_proba(
            raw_chunk,
            X_chunk,
            batch_size=args.hybrid_batch_size,
            state=stream_state,
            return_state=True,
        )
        chunk_prob_attack = chunk_prob[:, 1].astype(np.float32)
        chunk_pred = (chunk_prob_attack >= 0.5).astype(np.int32)
        chunk_prob_2d = np.column_stack([1.0 - chunk_prob_attack, chunk_prob_attack])
        chunk_metrics = classification_metrics(y_chunk, chunk_pred, chunk_prob_2d, CLASS_NAMES)

        trace_df = hybrid_model.last_adaptation_trace
        timeline_rows.append(
            {
                "batch": batch_index,
                "rows": int(len(raw_chunk)),
                "accuracy": float(chunk_metrics["accuracy"]),
                "f1_score": float(chunk_metrics["f1_score"]),
                "precision": float(chunk_metrics["precision"]),
                "recall": float(chunk_metrics["recall"]),
                "ema_drift_score": float(trace_df["ema_drift_score"].mean()),
                "adaptation_alpha": float(trace_df["adaptation_alpha"].mean()),
            }
        )

        y_all.append(y_chunk)
        pred_all.append(chunk_pred)
        prob_all.append(chunk_prob_2d)
        total_rows += len(raw_chunk)
        print(f"stream_batch={batch_index} rows={len(raw_chunk)} total_rows={total_rows}")

        if args.max_chunks and batch_index >= args.max_chunks:
            break

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(pred_all)
    y_prob = np.vstack(prob_all)
    overall = classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    timeline_df = pd.DataFrame(timeline_rows)
    timeline_csv = RESULTS_DIR / f"{PREFIX}_realtime_stream_timeline.csv"
    timeline_md = RESULTS_DIR / f"{PREFIX}_realtime_streaming_evaluation.md"
    timeline_png = RESULTS_DIR / f"{PREFIX}_drift_timeline.png"
    summary_json = RESULTS_DIR / f"{PREFIX}_realtime_stream_summary.json"

    timeline_df.to_csv(timeline_csv, index=False)
    _plot_timeline(timeline_df, timeline_png)

    lines = [
        "# Real-Time Streaming Evaluation\n\n",
        f"Processed rows: `{total_rows}`\n\n",
        "## Overall Online Drift-Adaptive Hybrid Metrics\n\n",
        f"- Accuracy: `{overall['accuracy'] * 100:.2f}%`\n",
        f"- Precision: `{overall['precision'] * 100:.2f}%`\n",
        f"- Recall: `{overall['recall'] * 100:.2f}%`\n",
        f"- F1 Score: `{overall['f1_score'] * 100:.2f}%`\n",
        f"- ROC AUC: `{overall['roc_auc']:.4f}`\n\n",
        "## Live Adaptation Timeline\n\n",
        "| Batch | Rows | Accuracy | F1 Score | EMA Drift | Adaptation Alpha |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
    ]
    for row in timeline_rows:
        lines.append(
            f"| {row['batch']} | {row['rows']} | {row['accuracy'] * 100:.2f}% | {row['f1_score'] * 100:.2f}% | "
            f"{row['ema_drift_score']:.4f} | {row['adaptation_alpha']:.4f} |\n"
        )
    timeline_md.write_text("".join(lines))

    summary = {
        "source": args.source,
        "processed_rows": int(total_rows),
        "metrics": overall,
        "timeline_rows": int(len(timeline_rows)),
        "artifacts": {
            "timeline_csv": str(timeline_csv.relative_to(BASE_DIR)),
            "timeline_markdown": str(timeline_md.relative_to(BASE_DIR)),
            "timeline_plot": str(timeline_png.relative_to(BASE_DIR)),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
