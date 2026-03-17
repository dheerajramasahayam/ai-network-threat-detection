from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SignatureIDSBaseline:
    """
    Traditional IDS-style deterministic baseline over flow features.

    The model does not learn decision boundaries in feature space. Instead, it
    calibrates a fixed set of expert-inspired signatures from benign traffic
    percentiles and applies those signatures as explicit rules.
    """

    model_name = "Signature IDS"

    def __init__(self):
        self.thresholds: dict[str, float] = {}
        self.rule_weights = {
            "volumetric_flood": 1.6,
            "syn_scan": 1.4,
            "asymmetric_scan": 1.3,
            "flag_anomaly": 0.9,
            "microburst": 1.0,
            "handshake_failure": 1.1,
        }

    def fit(self, train_df: pd.DataFrame, label_col: str = "label") -> None:
        benign = train_df[train_df[label_col] == 0]
        if benign.empty:
            raise ValueError("SignatureIDSBaseline requires benign traffic in the training data.")

        def q(column: str, quantile: float, fallback: float = 0.0) -> float:
            if column not in benign.columns:
                return fallback
            value = benign[column].quantile(quantile)
            if pd.isna(value):
                return fallback
            return float(value)

        self.thresholds = {
            "packets_s_high": q("flow_packets_s", 0.995, 1.0),
            "bytes_s_high": q("flow_bytes_s", 0.995, 1.0),
            "syn_high": q("syn_flag_cnt", 0.99, 1.0),
            "rst_high": q("rst_flag_cnt", 0.99, 1.0),
            "urg_high": q("urg_flag_cnt", 0.99, 1.0),
            "ack_low": q("ack_flag_cnt", 0.50, 0.0),
            "short_duration": max(q("flow_duration", 0.10, 1.0), 1e-6),
            "fwd_packets_high": q("total_fwd_packets", 0.995, 2.0),
            "bwd_packets_low": q("total_bwd_packets", 0.30, 1.0),
            "min_packet_small": q("min_packet_len", 0.20, 0.0),
            "packet_mean_high": q("packet_len_mean", 0.995, 1.0),
        }

    def _rules(self, df: pd.DataFrame) -> pd.DataFrame:
        t = self.thresholds
        rules = pd.DataFrame(index=df.index)
        rules["volumetric_flood"] = (
            (df["flow_packets_s"] >= t["packets_s_high"]) |
            (df["flow_bytes_s"] >= t["bytes_s_high"])
        )
        rules["syn_scan"] = (
            (df["syn_flag_cnt"] >= max(1.0, t["syn_high"])) &
            (df["ack_flag_cnt"] <= t["ack_low"] + 1.0)
        )
        rules["asymmetric_scan"] = (
            (df["total_fwd_packets"] >= max(2.0, t["fwd_packets_high"])) &
            (df["total_bwd_packets"] <= t["bwd_packets_low"] + 1.0) &
            (df["flow_duration"] <= t["short_duration"] * 5.0)
        )
        rules["flag_anomaly"] = (
            (df["rst_flag_cnt"] >= max(1.0, t["rst_high"])) |
            (df["urg_flag_cnt"] >= max(1.0, t["urg_high"]))
        )
        rules["microburst"] = (
            (df["flow_duration"] <= t["short_duration"]) &
            (df["flow_packets_s"] >= t["packets_s_high"] * 0.50)
        )
        rules["handshake_failure"] = (
            (df["ack_flag_cnt"] <= t["ack_low"]) &
            (df["total_bwd_packets"] <= t["bwd_packets_low"]) &
            (df["packet_len_mean"] >= t["packet_mean_high"] * 0.50)
        )
        return rules.astype(int)

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        rules = self._rules(df)
        weighted_sum = np.zeros(len(df), dtype=np.float32)
        for rule_name, weight in self.rule_weights.items():
            weighted_sum += rules[rule_name].to_numpy(dtype=np.float32) * weight
        return weighted_sum

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        score = self.decision_function(df)
        attack_prob = 0.02 + 0.96 * _sigmoid(score - 1.25)
        attack_prob = np.clip(attack_prob, 0.001, 0.999)
        benign_prob = 1.0 - attack_prob
        return np.column_stack([benign_prob, attack_prob])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)

    def explain(self, row: pd.Series) -> list[str]:
        rules = self._rules(pd.DataFrame([row]))
        fired = [name for name, value in rules.iloc[0].items() if int(value) == 1]
        return fired or ["no_signature_triggered"]

    def save(self, path: str) -> None:
        payload = {
            "thresholds": self.thresholds,
            "rule_weights": self.rule_weights,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str) -> "SignatureIDSBaseline":
        payload = json.loads(Path(path).read_text())
        instance = cls()
        instance.thresholds = payload["thresholds"]
        instance.rule_weights = payload["rule_weights"]
        return instance
