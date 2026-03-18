from __future__ import annotations

import argparse
import json
import sys
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

from evaluation.reporting import classification_metrics
from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from src.preprocessing import FEATURE_COLUMNS
from training.canonical_pipeline import _frame_to_array

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
MODEL_PREFIX = "transfer_unsw_nsl_to_cicids"
OUTPUT_PREFIX = "realtime_service_case_study"
CLASS_NAMES = ["Benign", "Attack"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a packet-capture replay case study over the local real-time dataset.")
    parser.add_argument(
        "--packet-path",
        default=str(BASE_DIR / "dataset" / "raw" / "new_2026" / "realtime_ids" / "RTN_traffic_dataset.csv"),
    )
    parser.add_argument("--bucket-seconds", type=int, default=1)
    parser.add_argument("--hybrid-batch-size", type=int, default=512)
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
    return signature, hybrid_model, preprocessor


def _read_packet_frame(path: str | Path) -> pd.DataFrame:
    usecols = [
        "frame.number",
        "frame.time_epoch",
        "frame.len",
        "ip.src",
        "ip.dst",
        "ip.proto",
        "ip.ttl",
        "tcp.srcport",
        "tcp.dstport",
        "tcp.len",
        "tcp.flags",
        "tcp.window_size",
        "udp.srcport",
        "udp.dstport",
        "udp.length",
        "label",
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["timestamp"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce").fillna(0.0)
    df["frame_len"] = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0.0)
    df["proto"] = pd.to_numeric(df["ip.proto"], errors="coerce").fillna(0).astype(int)
    df["src_port"] = (
        pd.to_numeric(df["tcp.srcport"], errors="coerce")
        .fillna(pd.to_numeric(df["udp.srcport"], errors="coerce"))
        .fillna(-1)
        .astype(int)
    )
    df["dst_port"] = (
        pd.to_numeric(df["tcp.dstport"], errors="coerce")
        .fillna(pd.to_numeric(df["udp.dstport"], errors="coerce"))
        .fillna(-1)
        .astype(int)
    )
    return df.sort_values(["timestamp", "frame.number"]).reset_index(drop=True)


def _flag_count(series: pd.Series, code: str) -> int:
    cleaned = series.fillna("").astype(str).str.upper()
    return int(cleaned.str.contains(code, regex=False).sum())


def _directional_payload(series: pd.Series, udp_series: pd.Series) -> np.ndarray:
    tcp_payload = pd.to_numeric(series, errors="coerce").fillna(0.0)
    udp_payload = pd.to_numeric(udp_series, errors="coerce").fillna(0.0) - 8.0
    udp_payload = udp_payload.clip(lower=0.0)
    payload = tcp_payload.where(tcp_payload > 0.0, udp_payload)
    return payload.to_numpy(dtype=np.float64)


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    return float(values.max()), float(values.min()), float(values.mean())


def _iat_stats(times: np.ndarray) -> tuple[float, float, float]:
    if len(times) <= 1:
        return 0.0, 0.0, 0.0
    diffs = np.diff(np.sort(times)) * 1_000_000.0
    return float(diffs.mean()), float(diffs.std(ddof=0)), float(diffs.sum())


def _header_length(proto: int) -> int:
    if proto == 6:
        return 20
    if proto == 17:
        return 8
    return 0


def _aggregate_packet_capture(df: pd.DataFrame, bucket_seconds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_ts = int(np.floor(df["timestamp"].min()))
    df = df.copy()
    df["bucket"] = ((df["timestamp"].astype(int) - min_ts) // bucket_seconds).astype(int)
    df["src_endpoint"] = df["ip.src"].astype(str) + ":" + df["src_port"].astype(str)
    df["dst_endpoint"] = df["ip.dst"].astype(str) + ":" + df["dst_port"].astype(str)
    df["session_key"] = df.apply(
        lambda row: f"{row['proto']}|{'|'.join(sorted([row['src_endpoint'], row['dst_endpoint']]))}",
        axis=1,
    )

    rows: list[dict[str, float | int | str]] = []
    second_rows: list[dict[str, float | int]] = []

    per_second_packet_counts = (
        df.assign(second=df["timestamp"].astype(int))
        .groupby(["second", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for _, second_row in per_second_packet_counts.iterrows():
        second_rows.append(
            {
                "second": int(second_row["second"]),
                "relative_second": int(second_row["second"] - min_ts),
                "benign_packets": int(second_row.get(0, 0)),
                "attack_packets": int(second_row.get(1, 0)),
            }
        )

    grouped = df.groupby(["bucket", "session_key"], sort=True)
    for (bucket, session_key), group in grouped:
        group = group.sort_values(["timestamp", "frame.number"]).reset_index(drop=True)
        first = group.iloc[0]
        forward_mask = (
            (group["ip.src"] == first["ip.src"])
            & (group["ip.dst"] == first["ip.dst"])
            & (group["src_port"] == first["src_port"])
            & (group["dst_port"] == first["dst_port"])
        )
        fwd = group[forward_mask]
        bwd = group[~forward_mask]

        times = group["timestamp"].to_numpy(dtype=np.float64)
        duration_s = float(times.max() - times.min()) if len(times) else 0.0
        safe_duration_s = max(duration_s, 1e-6)
        flow_duration = duration_s * 1_000_000.0

        all_lengths = group["frame_len"].to_numpy(dtype=np.float64)
        fwd_lengths = fwd["frame_len"].to_numpy(dtype=np.float64)
        bwd_lengths = bwd["frame_len"].to_numpy(dtype=np.float64)
        max_all = float(all_lengths.max()) if len(all_lengths) else 0.0
        min_all = float(all_lengths.min()) if len(all_lengths) else 0.0
        packet_mean = float(all_lengths.mean()) if len(all_lengths) else 0.0
        packet_std = float(all_lengths.std(ddof=0)) if len(all_lengths) else 0.0
        packet_var = float(all_lengths.var(ddof=0)) if len(all_lengths) else 0.0

        fwd_max, fwd_min, fwd_mean = _stats(fwd_lengths)
        bwd_max, bwd_min, bwd_mean = _stats(bwd_lengths)

        flow_iat_mean, flow_iat_std, _ = _iat_stats(times)
        _, _, fwd_iat_total = _iat_stats(fwd["timestamp"].to_numpy(dtype=np.float64))
        _, _, bwd_iat_total = _iat_stats(bwd["timestamp"].to_numpy(dtype=np.float64))

        total_bytes = float(all_lengths.sum())
        fwd_bytes = float(fwd_lengths.sum())
        bwd_bytes = float(bwd_lengths.sum())
        total_packets = len(group)
        fwd_packets = len(fwd)
        bwd_packets = len(bwd)

        fwd_payload = _directional_payload(fwd["tcp.len"], fwd["udp.length"])
        row = {column: 0.0 for column in FEATURE_COLUMNS}
        row.update(
            {
                "flow_duration": flow_duration,
                "total_fwd_packets": float(fwd_packets),
                "total_bwd_packets": float(bwd_packets),
                "total_len_fwd_packets": fwd_bytes,
                "total_len_bwd_packets": bwd_bytes,
                "fwd_packet_len_max": fwd_max,
                "fwd_packet_len_min": fwd_min,
                "fwd_packet_len_mean": fwd_mean,
                "bwd_packet_len_max": bwd_max,
                "bwd_packet_len_min": bwd_min,
                "bwd_packet_len_mean": bwd_mean,
                "flow_bytes_s": total_bytes / safe_duration_s,
                "flow_packets_s": total_packets / safe_duration_s,
                "flow_iat_mean": flow_iat_mean,
                "flow_iat_std": flow_iat_std,
                "fwd_iat_total": fwd_iat_total,
                "bwd_iat_total": bwd_iat_total,
                "fwd_psh_flags": float(_flag_count(fwd["tcp.flags"], "P")),
                "bwd_psh_flags": float(_flag_count(bwd["tcp.flags"], "P")),
                "fwd_header_len": float(fwd_packets * _header_length(int(first["proto"]))),
                "bwd_header_len": float(bwd_packets * _header_length(int(first["proto"]))),
                "fwd_packets_s": fwd_packets / safe_duration_s,
                "bwd_packets_s": bwd_packets / safe_duration_s,
                "min_packet_len": min_all,
                "max_packet_len": max_all,
                "packet_len_mean": packet_mean,
                "packet_len_std": packet_std,
                "packet_len_variance": packet_var,
                "fin_flag_cnt": float(_flag_count(group["tcp.flags"], "F")),
                "syn_flag_cnt": float(_flag_count(group["tcp.flags"], "S")),
                "rst_flag_cnt": float(_flag_count(group["tcp.flags"], "R")),
                "psh_flag_cnt": float(_flag_count(group["tcp.flags"], "P")),
                "ack_flag_cnt": float(_flag_count(group["tcp.flags"], "A")),
                "urg_flag_cnt": float(_flag_count(group["tcp.flags"], "U")),
                "avg_packet_size": packet_mean,
                "avg_fwd_seg_size": fwd_mean,
                "avg_bwd_seg_size": bwd_mean,
                "init_win_bytes_fwd": float(pd.to_numeric(fwd["tcp.window_size"], errors="coerce").fillna(0.0).iloc[0]) if fwd_packets else 0.0,
                "init_win_bytes_bwd": float(pd.to_numeric(bwd["tcp.window_size"], errors="coerce").fillna(0.0).iloc[0]) if bwd_packets else 0.0,
                "act_data_pkt_fwd": float((fwd_payload > 0.0).sum()),
                "min_seg_size_fwd": float(fwd_payload.min()) if len(fwd_payload) else 0.0,
            }
        )

        row.update(
            {
                "label": int(group["label"].max()),
                "_bucket": int(bucket),
                "_bucket_start": int(min_ts + bucket * bucket_seconds),
                "_session_key": session_key,
                "_src_ip": str(first["ip.src"]),
                "_dst_ip": str(first["ip.dst"]),
                "_src_port": int(first["src_port"]),
                "_dst_port": int(first["dst_port"]),
                "_proto": int(first["proto"]),
                "_packet_count": int(total_packets),
                "_attack_packet_count": int((group["label"] == 1).sum()),
                "_benign_packet_count": int((group["label"] == 0).sum()),
            }
        )
        rows.append(row)

    flow_df = pd.DataFrame(rows).sort_values(["_bucket", "_attack_packet_count", "_packet_count"], ascending=[True, False, False]).reset_index(drop=True)
    second_df = pd.DataFrame(second_rows).sort_values("relative_second").reset_index(drop=True)
    return flow_df, second_df


def _plot_case_study(second_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(second_df["relative_second"], second_df["attack_packets"], label="Attack packets", color="#b91c1c", alpha=0.75)
    axes[0].plot(second_df["relative_second"], second_df["benign_packets"], label="Benign packets", color="#2563eb", linewidth=2)
    axes[0].set_ylabel("Packet Count")
    axes[0].set_title("Packet-Capture Replay Case Study")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(second_df["relative_second"], second_df["max_hybrid_prob"], label="Max hybrid prob", color="#7c3aed", linewidth=2)
    axes[1].plot(second_df["relative_second"], second_df["max_signature_prob"], label="Max signature prob", color="#b45309", linewidth=2)
    axes[1].axhline(0.5, linestyle="--", color="#111827", linewidth=1)
    axes[1].set_xlabel("Seconds Since Replay Start")
    axes[1].set_ylabel("Attack Probability")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    signature_model, hybrid_model, preprocessor = _load_models()
    packet_df = _read_packet_frame(args.packet_path)
    flow_df, second_df = _aggregate_packet_capture(packet_df, bucket_seconds=args.bucket_seconds)

    X = preprocessor.transform(_frame_to_array(flow_df[FEATURE_COLUMNS])).astype(np.float32)
    signature_prob = signature_model.predict_proba(flow_df[FEATURE_COLUMNS])[:, 1].astype(np.float32)

    hybrid_state = None
    hybrid_prob_chunks: list[np.ndarray] = []
    trace_frames: list[pd.DataFrame] = []
    for bucket in sorted(flow_df["_bucket"].unique()):
        batch = flow_df[flow_df["_bucket"] == bucket].reset_index(drop=True)
        X_batch = X[flow_df["_bucket"].to_numpy() == bucket]
        batch_prob, hybrid_state = hybrid_model.predict_proba(
            batch[FEATURE_COLUMNS],
            X_batch,
            batch_size=args.hybrid_batch_size,
            state=hybrid_state,
            return_state=True,
        )
        hybrid_prob_chunks.append(batch_prob[:, 1].astype(np.float32))
        trace = hybrid_model.last_adaptation_trace.copy()
        trace["bucket"] = int(bucket)
        trace_frames.append(trace)

    hybrid_prob = np.concatenate(hybrid_prob_chunks)
    flow_df["signature_prob_attack"] = signature_prob
    flow_df["hybrid_prob_attack"] = hybrid_prob
    flow_df["signature_pred"] = (signature_prob >= 0.5).astype(int)
    flow_df["hybrid_pred"] = (hybrid_prob >= 0.5).astype(int)

    y_true = flow_df["label"].to_numpy(dtype=np.int32)
    hybrid_metrics = classification_metrics(
        y_true,
        flow_df["hybrid_pred"].to_numpy(dtype=np.int32),
        np.column_stack([1.0 - hybrid_prob, hybrid_prob]),
        CLASS_NAMES,
    )
    signature_metrics = classification_metrics(
        y_true,
        flow_df["signature_pred"].to_numpy(dtype=np.int32),
        np.column_stack([1.0 - signature_prob, signature_prob]),
        CLASS_NAMES,
    )

    attack_flow = flow_df[(flow_df["label"] == 1) & (flow_df["_dst_port"] == 3000)].copy()
    top_attack_rules = []
    if not attack_flow.empty:
        top_flow = attack_flow.sort_values("hybrid_prob_attack", ascending=False).iloc[0]
        top_attack_rules = signature_model.explain(top_flow[FEATURE_COLUMNS])

    bucket_summary = (
        flow_df.groupby("_bucket")
        .agg(
            bucket_start=("_bucket_start", "first"),
            second_attack_flows=("label", "sum"),
            max_hybrid_prob=("hybrid_prob_attack", "max"),
            max_signature_prob=("signature_prob_attack", "max"),
            target_hybrid_prob=("hybrid_prob_attack", lambda s: float(s.loc[attack_flow.index.intersection(s.index)].max()) if not attack_flow.empty and len(attack_flow.index.intersection(s.index)) else 0.0),
            target_signature_prob=("signature_prob_attack", lambda s: float(s.loc[attack_flow.index.intersection(s.index)].max()) if not attack_flow.empty and len(attack_flow.index.intersection(s.index)) else 0.0),
            hybrid_detected=("hybrid_pred", "max"),
            signature_detected=("signature_pred", "max"),
        )
        .reset_index()
    )
    second_df = second_df.merge(bucket_summary, left_on="relative_second", right_on="_bucket", how="left").fillna(
        {
            "second_attack_flows": 0,
            "max_hybrid_prob": 0.0,
            "max_signature_prob": 0.0,
            "target_hybrid_prob": 0.0,
            "target_signature_prob": 0.0,
            "hybrid_detected": 0,
            "signature_detected": 0,
        }
    )
    second_df["attack_active"] = (second_df["attack_packets"] > 0).astype(int)

    attack_onset = int(second_df.loc[second_df["attack_active"] == 1, "relative_second"].min())
    attack_end = int(second_df.loc[second_df["attack_active"] == 1, "relative_second"].max())
    hybrid_detection = second_df[(second_df["relative_second"] >= attack_onset) & (second_df["hybrid_detected"] == 1)]
    signature_detection = second_df[(second_df["relative_second"] >= attack_onset) & (second_df["signature_detected"] == 1)]
    hybrid_delay = int(hybrid_detection["relative_second"].iloc[0] - attack_onset) if not hybrid_detection.empty else None
    signature_delay = int(signature_detection["relative_second"].iloc[0] - attack_onset) if not signature_detection.empty else None

    attack_period = second_df[(second_df["relative_second"] >= attack_onset) & (second_df["relative_second"] <= attack_end)]
    benign_warmup = second_df[second_df["relative_second"] < attack_onset]
    benign_cooldown = second_df[second_df["relative_second"] > attack_end]
    target_attack_mean = float(attack_period["target_hybrid_prob"].mean()) if len(attack_period) else 0.0
    target_attack_min = float(attack_period["target_hybrid_prob"].min()) if len(attack_period) else 0.0
    target_attack_max = float(attack_period["target_hybrid_prob"].max()) if len(attack_period) else 0.0
    target_attack_onset = float(attack_period["target_hybrid_prob"].iloc[0]) if len(attack_period) else 0.0

    trace_df = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    csv_path = RESULTS_DIR / f"{OUTPUT_PREFIX}_timeline.csv"
    md_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.md"
    json_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.json"
    png_path = RESULTS_DIR / f"{OUTPUT_PREFIX}.png"

    second_df.to_csv(csv_path, index=False)
    _plot_case_study(second_df, png_path)

    lines = [
        "# Real-World Packet Replay Case Study\n\n",
        "This case study replays the local `RTN_traffic_dataset.csv` export from the paired packet capture `RealTimeNetworkTrafficCapture.pcapng`.\n\n",
        f"- Packet rows: `{len(packet_df)}`\n",
        f"- Aggregated one-second bidirectional flow windows: `{len(flow_df)}`\n",
        f"- Benign warm-up seconds: `{len(benign_warmup)}`\n",
        f"- Attack seconds: `{len(attack_period)}`\n",
        f"- Benign cool-down seconds: `{len(benign_cooldown)}`\n",
        f"- Attack onset: `t={attack_onset}s`\n",
        f"- Attack end: `t={attack_end}s`\n",
        f"- Dominant attack path: `192.168.76.9 -> 192.168.12.56:3000/UDP`\n",
        f"- Hybrid detection delay: `{hybrid_delay if hybrid_delay is not None else 'not detected'}` seconds\n",
        f"- Signature detection delay: `{signature_delay if signature_delay is not None else 'not detected'}` seconds\n",
        f"- Dominant attack-flow hybrid probability at onset: `{target_attack_onset:.4f}`\n",
        f"- Dominant attack-flow hybrid probability during attack: mean `{target_attack_mean:.4f}`, range `[{target_attack_min:.4f}, {target_attack_max:.4f}]`\n",
        f"- Mean max hybrid probability during benign warm-up: `{benign_warmup['max_hybrid_prob'].mean():.4f}`\n",
        f"- Strongest signature explanation on the dominant attack flow: `{', '.join(top_attack_rules) if top_attack_rules else 'n/a'}`\n\n",
        "## Flow-Window Metrics\n\n",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
        f"| Signature IDS | {signature_metrics['accuracy'] * 100:.2f}% | {signature_metrics['precision'] * 100:.2f}% | {signature_metrics['recall'] * 100:.2f}% | {signature_metrics['f1_score'] * 100:.2f}% | {signature_metrics['roc_auc']:.4f} |\n",
        f"| Drift-Adaptive Hybrid | {hybrid_metrics['accuracy'] * 100:.2f}% | {hybrid_metrics['precision'] * 100:.2f}% | {hybrid_metrics['recall'] * 100:.2f}% | {hybrid_metrics['f1_score'] * 100:.2f}% | {hybrid_metrics['roc_auc']:.4f} |\n",
    ]
    md_path.write_text("".join(lines))

    summary = {
        "packet_rows": int(len(packet_df)),
        "flow_rows": int(len(flow_df)),
        "attack_onset_second": attack_onset,
        "attack_end_second": attack_end,
        "attack_seconds": int(len(attack_period)),
        "hybrid_detection_delay_seconds": hybrid_delay,
        "signature_detection_delay_seconds": signature_delay,
        "dominant_attack_flow_probability_at_onset": target_attack_onset,
        "dominant_attack_flow_probability_mean": target_attack_mean,
        "dominant_attack_flow_probability_min": target_attack_min,
        "dominant_attack_flow_probability_max": target_attack_max,
        "mean_hybrid_attack_probability_during_attack": float(attack_period["max_hybrid_prob"].mean()),
        "mean_hybrid_attack_probability_during_benign_warmup": float(benign_warmup["max_hybrid_prob"].mean()) if len(benign_warmup) else 0.0,
        "hybrid_metrics": hybrid_metrics,
        "signature_metrics": signature_metrics,
        "dominant_attack": {
            "src_ip": "192.168.76.9",
            "dst_ip": "192.168.12.56",
            "dst_port": 3000,
            "protocol": "UDP",
            "signature_explanation": top_attack_rules,
        },
        "artifacts": {
            "timeline_csv": str(csv_path.relative_to(BASE_DIR)),
            "timeline_plot": str(png_path.relative_to(BASE_DIR)),
            "case_study_markdown": str(md_path.relative_to(BASE_DIR)),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
