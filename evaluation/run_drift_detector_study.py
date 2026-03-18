from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from models.drift_aware_hybrid import DriftAwareHybridDetector, _normalize_weights
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from src.preprocessing import FEATURE_COLUMNS
from training.canonical_pipeline import (
    _frame_to_array,
    _load_nsl_kdd_canonical,
    _load_unsw_canonical,
    iter_cicids_canonical_chunks,
)

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
PREFIX = "transfer_unsw_nsl_to_cicids"
CLASS_NAMES = ["Benign", "Attack"]


@dataclass
class WindowPacket:
    phase: str
    window_index: int
    rows: int
    y_true: np.ndarray
    components: np.ndarray
    mean_drift: float
    static_error_rate: float


class ADWINWindowDetector:
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: list[float] = []

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        if len(self.window) < 4:
            return False
        values = np.asarray(self.window, dtype=np.float64)
        midpoint = len(values) // 2
        left = values[:midpoint]
        right = values[midpoint:]
        epsilon = np.sqrt(2.0 * np.log(2.0 / self.delta) * (1.0 / len(left) + 1.0 / len(right)))
        if abs(left.mean() - right.mean()) > epsilon:
            self.window = right.tolist()
            return True
        return False


class DDMWindowDetector:
    def __init__(self):
        self.count = 0
        self.mean_error = 0.0
        self.best = float("inf")

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean_error += (float(value) - self.mean_error) / self.count
        std = np.sqrt(max(self.mean_error * (1.0 - self.mean_error), 1e-6) / self.count)
        score = self.mean_error + std
        if score < self.best:
            self.best = score
        if self.count < 3:
            return False
        return score > self.best + 0.08


class PageHinkleyWindowDetector:
    def __init__(self, delta: float = 0.005, threshold: float = 0.08):
        self.delta = delta
        self.threshold = threshold
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean += (float(value) - self.mean) / self.count
        self.cumulative += float(value) - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        return (self.cumulative - self.minimum) > self.threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a drift detector comparison study for the adaptive hybrid.")
    parser.add_argument("--window-size", type=int, default=50_000)
    parser.add_argument("--hybrid-batch-size", type=int, default=4096)
    parser.add_argument("--unsw-dir", default=str(BASE_DIR / "dataset" / "raw" / "unsw_nb15"))
    parser.add_argument("--nsl-dir", default=str(BASE_DIR / "dataset" / "raw" / "nsl_kdd"))
    parser.add_argument("--cicids-path", default=str(BASE_DIR / "dataset" / "cicids2017.csv"))
    return parser.parse_args()


def _load_models():
    signature = SignatureIDSBaseline.load(str(ARTIFACTS_DIR / f"{PREFIX}_signature_ids.json"))
    random_forest = RandomForestThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_random_forest.joblib"))
    lstm_model = LSTMThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_lstm.pt"))
    transformer_model = TransformerThreatDetector.load(str(ARTIFACTS_DIR / f"{PREFIX}_transformer.pt"))
    preprocessor = joblib.load(ARTIFACTS_DIR / f"{PREFIX}_preprocessor.joblib")
    hybrid_model = DriftAwareHybridDetector(signature, random_forest, lstm_model, transformer_model)
    hybrid_model.load_state(str(ARTIFACTS_DIR / f"{PREFIX}_hybrid.joblib"))
    return preprocessor, hybrid_model


def _yield_source_windows(unsw_dir: str, nsl_dir: str, window_size: int):
    source_df = pd.concat(
        [
            _load_unsw_canonical(Path(unsw_dir) / "UNSW_NB15_training-set.csv"),
            _load_nsl_kdd_canonical(Path(nsl_dir) / "KDDTrain+.txt"),
        ],
        ignore_index=True,
    ).reset_index(drop=True)
    for start in range(0, len(source_df), window_size):
        stop = min(start + window_size, len(source_df))
        yield source_df.iloc[start:stop].reset_index(drop=True)


def _apply_components(
    hybrid_model: DriftAwareHybridDetector,
    components: np.ndarray,
    alpha: float,
    rolling_attack_rate: float,
) -> tuple[np.ndarray, float]:
    weights = _normalize_weights((1.0 - alpha) * hybrid_model.stable_weights + alpha * hybrid_model.stressed_weights)
    attack_prob = components @ weights
    disagreement = np.std(components[:, :4], axis=1)
    temporal_support = np.maximum(components[:, 2], components[:, 3])
    boost = np.clip(alpha * disagreement, 0.0, 0.20)
    attack_prob = (1.0 - boost) * attack_prob + boost * temporal_support

    batch_attack_rate = float(np.mean(attack_prob))
    rolling_attack_rate = 0.85 * rolling_attack_rate + 0.15 * batch_attack_rate
    bias = float(np.clip((rolling_attack_rate - hybrid_model.source_attack_rate) * alpha, -0.08, 0.08))
    attack_prob = np.clip(attack_prob + bias, 1e-3, 1.0 - 1e-3).astype(np.float32)
    return np.column_stack([1.0 - attack_prob, attack_prob]), rolling_attack_rate


def _collect_windows(args: argparse.Namespace, preprocessor, hybrid_model: DriftAwareHybridDetector) -> list[WindowPacket]:
    packets: list[WindowPacket] = []
    window_index = 0

    def register_window(phase: str, frame: pd.DataFrame) -> None:
        nonlocal window_index
        window_index += 1
        y_true = frame["label"].to_numpy(dtype=np.int32)
        X_scaled = preprocessor.transform(_frame_to_array(frame[FEATURE_COLUMNS])).astype(np.float32)
        rule_prob, rf_prob, lstm_prob, transformer_prob, drift_prob = hybrid_model._base_probabilities(frame, X_scaled)
        meta_features = np.column_stack(
            [
                rule_prob,
                rf_prob,
                lstm_prob,
                transformer_prob,
                drift_prob,
                rule_prob * drift_prob,
                transformer_prob * drift_prob,
                rf_prob * (1.0 - drift_prob),
            ]
        )
        meta_prob = hybrid_model.meta_model.predict_proba(meta_features)[:, 1]
        components = np.column_stack([rule_prob, rf_prob, lstm_prob, transformer_prob, meta_prob]).astype(np.float32)
        static_prob = np.column_stack(
            [
                1.0 - np.clip(meta_prob, 1e-3, 1.0 - 1e-3),
                np.clip(meta_prob, 1e-3, 1.0 - 1e-3),
            ]
        )
        static_pred = (static_prob[:, 1] >= 0.5).astype(np.int32)
        packets.append(
            WindowPacket(
                phase=phase,
                window_index=window_index,
                rows=len(frame),
                y_true=y_true,
                components=components.astype(np.float32),
                mean_drift=float(np.mean(drift_prob)),
                static_error_rate=float(np.mean(static_pred != y_true)),
            )
        )

    for frame in _yield_source_windows(args.unsw_dir, args.nsl_dir, args.window_size):
        register_window("source", frame)
    for frame in iter_cicids_canonical_chunks(args.cicids_path, chunksize=args.window_size):
        register_window("external", frame.reset_index(drop=True))
    return packets


def _evaluate_detector(name: str, packets: list[WindowPacket], hybrid_model: DriftAwareHybridDetector) -> dict[str, float | int | str]:
    if name == "ADWIN":
        detector = ADWINWindowDetector()
    elif name == "DDM":
        detector = DDMWindowDetector()
    elif name == "Page-Hinkley":
        detector = PageHinkleyWindowDetector()
    else:
        detector = None

    onset_window = next(packet.window_index for packet in packets if packet.phase == "external")
    detection_window = None
    false_positives = 0
    fired = False
    rolling_attack_rate = hybrid_model.source_attack_rate
    ema_drift = hybrid_model.drift_stats["normalized_median"]

    y_external = []
    pred_external = []
    prob_external = []

    for packet in packets:
        if name == "Isolation Forest":
            ema_drift = hybrid_model.ema_decay * ema_drift + (1.0 - hybrid_model.ema_decay) * packet.mean_drift
            alpha = float(
                np.clip(
                    (ema_drift - hybrid_model.adaptation_start) /
                    (hybrid_model.adaptation_end - hybrid_model.adaptation_start + 1e-8),
                    0.0,
                    1.0,
                )
            )
            signal = alpha >= 0.5
        else:
            signal = bool(detector.update(packet.static_error_rate))
            if signal:
                fired = True
            alpha = 1.0 if fired else 0.0

        if signal and detection_window is None:
            detection_window = packet.window_index
        if packet.phase == "source" and signal:
            false_positives += 1

        if packet.phase == "external":
            y_prob, rolling_attack_rate = _apply_components(
                hybrid_model,
                packet.components,
                alpha,
                rolling_attack_rate,
            )
            y_pred = (y_prob[:, 1] >= 0.5).astype(np.int32)
            y_external.append(packet.y_true)
            pred_external.append(y_pred)
            prob_external.append(y_prob)

    y_true = np.concatenate(y_external)
    y_pred = np.concatenate(pred_external)
    y_prob = np.vstack(prob_external)
    metrics = classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    delay = "not_detected" if detection_window is None else int(max(0, detection_window - onset_window))

    return {
        "Drift Detector": name,
        "Detection Delay (windows)": delay,
        "False Positives": int(false_positives),
        "F1 After Adaptation": round(metrics["f1_score"] * 100, 2),
        "Accuracy After Adaptation": round(metrics["accuracy"] * 100, 2),
        "ROC AUC": round(metrics["roc_auc"], 4),
    }


def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    preprocessor, hybrid_model = _load_models()
    packets = _collect_windows(args, preprocessor, hybrid_model)

    rows = [
        _evaluate_detector(detector_name, packets, hybrid_model)
        for detector_name in ["Isolation Forest", "ADWIN", "DDM", "Page-Hinkley"]
    ]
    results_df = pd.DataFrame(rows).sort_values("F1 After Adaptation", ascending=False)
    csv_path = RESULTS_DIR / "drift_detector_study.csv"
    md_path = RESULTS_DIR / "drift_detector_study.md"
    png_path = RESULTS_DIR / "drift_detector_study.png"
    json_path = RESULTS_DIR / "drift_detector_study.json"

    results_df.to_csv(csv_path, index=False)

    lines = [
        "# Drift Detector Comparison Study\n\n",
        "Source windows come from the combined UNSW-NB15 + NSL-KDD source-domain corpus. External windows come from the full CICIDS2017 corpus.\n\n",
        f"Window size: `{args.window_size}` rows\n\n",
        "| Drift Detector | Detection Delay (windows) | False Positives | F1 After Adaptation | Accuracy After Adaptation | ROC AUC |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['Drift Detector']} | {row['Detection Delay (windows)']} | {int(row['False Positives'])} | "
            f"{row['F1 After Adaptation']:.2f} | {row['Accuracy After Adaptation']:.2f} | {row['ROC AUC']:.4f} |\n"
        )
    md_path.write_text("".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(results_df["Drift Detector"], results_df["F1 After Adaptation"], color=["#7c3aed", "#1d4ed8", "#0f766e", "#b45309"])
    axes[0].set_title("F1 After Adaptation")
    axes[0].set_ylabel("Weighted F1 (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.2)

    delay_values = [0 if value == "not_detected" else value for value in results_df["Detection Delay (windows)"]]
    axes[1].bar(results_df["Drift Detector"], delay_values, color=["#7c3aed", "#1d4ed8", "#0f766e", "#b45309"])
    axes[1].set_title("Detection Delay")
    axes[1].set_ylabel("Windows after drift onset")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    summary = {
        "window_size": args.window_size,
        "source_windows": int(sum(packet.phase == "source" for packet in packets)),
        "external_windows": int(sum(packet.phase == "external" for packet in packets)),
        "results": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
