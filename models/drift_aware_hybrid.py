from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

COMPONENT_NAMES = [
    "signature_ids",
    "random_forest",
    "lstm",
    "transformer",
    "stacked_meta",
]


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-6, None)
    return clipped / clipped.sum()


class DriftAwareHybridDetector:
    """
    Proposed method: a drift-adaptive hybrid that fuses a traditional signature
    IDS score with RF/LSTM/Transformer probabilities and an unsupervised drift
    detector, then reweights those signals online when the traffic stream moves
    away from the training distribution.
    """

    model_name = "Drift-Adaptive Hybrid"

    def __init__(self, signature_model, rf_model, lstm_model, transformer_model):
        self.signature_model = signature_model
        self.rf_model = rf_model
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.meta_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.drift_detector = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.fitted = False
        self.stable_weights = _normalize_weights(np.array([0.15, 0.30, 0.20, 0.15, 0.20]))
        self.stressed_weights = _normalize_weights(np.array([0.12, 0.12, 0.32, 0.24, 0.20]))
        self.drift_stats = {
            "q10": 0.0,
            "q50": 0.5,
            "q90": 1.0,
            "normalized_median": 0.5,
        }
        self.source_attack_rate = 0.5
        self.adaptation_start = 0.55
        self.adaptation_end = 0.80
        self.ema_decay = 0.75
        self.last_adaptation_trace = pd.DataFrame()

    def _raw_drift_score(self, X_scaled: np.ndarray) -> np.ndarray:
        return -self.drift_detector.score_samples(X_scaled)

    def _drift_score(self, X_scaled: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._raw_drift_score(X_scaled), dtype=np.float32)
        if not self.fitted:
            raw = raw - raw.min()
            return raw / (raw.max() + 1e-8)
        lo = self.drift_stats["q10"]
        hi = self.drift_stats["q90"]
        return np.clip((raw - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    def _base_probabilities(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rule_prob = self.signature_model.predict_proba(raw_df)[:, 1]
        rf_prob = self.rf_model.predict_proba(X_scaled)[:, 1]
        lstm_prob = self.lstm_model.predict_proba(X_scaled)[:, 1]
        transformer_prob = self.transformer_model.predict_proba(X_scaled)[:, 1]
        drift_prob = self._drift_score(X_scaled)
        return rule_prob, rf_prob, lstm_prob, transformer_prob, drift_prob

    def _meta_features(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        rule_prob, rf_prob, lstm_prob, transformer_prob, drift_prob = self._base_probabilities(raw_df, X_scaled)
        return np.column_stack(
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

    def _component_probabilities(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rule_prob, rf_prob, lstm_prob, transformer_prob, _ = self._base_probabilities(raw_df, X_scaled)
        meta_prob = self.meta_model.predict_proba(self._meta_features(raw_df, X_scaled))[:, 1]
        components = np.column_stack(
            [rule_prob, rf_prob, lstm_prob, transformer_prob, meta_prob]
        )
        return components.astype(np.float32), self._drift_score(X_scaled)

    def _derive_regime_weights(
        self,
        y_true: np.ndarray,
        components: np.ndarray,
        temporal_bias: float = 1.0,
    ) -> np.ndarray:
        scores = []
        for column in range(components.shape[1]):
            preds = (components[:, column] >= 0.5).astype(int)
            score = f1_score(y_true, preds, average="weighted", zero_division=0)
            scores.append(max(score, 0.01))

        weights = np.array(scores, dtype=np.float64)
        if temporal_bias > 1.0:
            weights[2] *= temporal_bias
            weights[3] *= 1.0 + (temporal_bias - 1.0) * 0.75
            weights[1] *= max(0.70, 1.0 - (temporal_bias - 1.0) * 0.60)
        return _normalize_weights(weights)

    def _simulate_drift(
        self,
        raw_df: pd.DataFrame,
        X_scaled: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(42)
        drift_raw = raw_df.copy()
        stress_columns = [
            "flow_duration",
            "total_len_fwd_packets",
            "total_len_bwd_packets",
            "flow_bytes_s",
            "flow_packets_s",
            "flow_iat_mean",
            "flow_iat_std",
            "packet_len_mean",
            "packet_len_std",
            "avg_packet_size",
            "init_win_bytes_fwd",
            "init_win_bytes_bwd",
        ]
        for column in stress_columns:
            if column not in drift_raw.columns:
                continue
            values = drift_raw[column].to_numpy(dtype=np.float64, copy=True)
            spread = float(np.nanstd(values))
            multiplicative = rng.uniform(0.70, 1.45, size=len(values))
            additive = rng.normal(0.0, max(spread, 1.0) * 0.08, size=len(values))
            stressed = np.clip(values * multiplicative + additive, a_min=0.0, a_max=None)
            drift_raw[column] = stressed

        drift_scaled = X_scaled.astype(np.float32, copy=True)
        drift_scaled += rng.normal(0.0, 0.35, size=drift_scaled.shape).astype(np.float32)
        mask = rng.random(size=drift_scaled.shape) < 0.08
        drift_scaled[mask] *= 0.70
        return drift_raw, drift_scaled

    def fit(
        self,
        train_scaled: np.ndarray,
        val_raw_df: pd.DataFrame,
        val_scaled: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.drift_detector.fit(train_scaled)

        train_raw_drift = self._raw_drift_score(train_scaled)
        self.drift_stats = {
            "q10": float(np.percentile(train_raw_drift, 10)),
            "q50": float(np.percentile(train_raw_drift, 50)),
            "q90": float(np.percentile(train_raw_drift, 90)),
            "normalized_median": 0.5,
        }
        self.source_attack_rate = float(np.mean(y_val))

        meta_X = self._meta_features(val_raw_df, val_scaled)
        self.meta_model.fit(meta_X, y_val)
        self.fitted = True

        stable_components, _ = self._component_probabilities(val_raw_df, val_scaled)
        self.stable_weights = self._derive_regime_weights(y_val, stable_components)

        stressed_raw_df, stressed_scaled = self._simulate_drift(val_raw_df, val_scaled)
        stressed_components, _ = self._component_probabilities(stressed_raw_df, stressed_scaled)
        self.stressed_weights = self._derive_regime_weights(
            y_val,
            stressed_components,
            temporal_bias=1.35,
        )

    def predict_proba_static(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_proba_static().")
        attack_prob = self.meta_model.predict_proba(self._meta_features(raw_df, X_scaled))[:, 1]
        attack_prob = np.clip(attack_prob, 1e-3, 1.0 - 1e-3)
        return np.column_stack([1.0 - attack_prob, attack_prob])

    def predict_proba(
        self,
        raw_df: pd.DataFrame,
        X_scaled: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        probabilities = []
        trace_rows = []
        ema_drift = self.drift_stats["normalized_median"]
        rolling_attack_rate = self.source_attack_rate

        for start in range(0, len(raw_df), batch_size):
            stop = min(start + batch_size, len(raw_df))
            raw_batch = raw_df.iloc[start:stop].reset_index(drop=True)
            scaled_batch = X_scaled[start:stop]

            components, drift_prob = self._component_probabilities(raw_batch, scaled_batch)
            batch_drift = float(np.mean(drift_prob))
            ema_drift = self.ema_decay * ema_drift + (1.0 - self.ema_decay) * batch_drift
            alpha = float(
                np.clip(
                    (ema_drift - self.adaptation_start) /
                    (self.adaptation_end - self.adaptation_start + 1e-8),
                    0.0,
                    1.0,
                )
            )
            weights = _normalize_weights(
                (1.0 - alpha) * self.stable_weights + alpha * self.stressed_weights
            )

            attack_prob = components @ weights
            disagreement = np.std(components[:, :4], axis=1)
            temporal_support = np.maximum(components[:, 2], components[:, 3])
            boost = np.clip(alpha * disagreement, 0.0, 0.20)
            attack_prob = (1.0 - boost) * attack_prob + boost * temporal_support

            batch_attack_rate = float(np.mean(attack_prob))
            rolling_attack_rate = 0.85 * rolling_attack_rate + 0.15 * batch_attack_rate
            bias = float(np.clip((rolling_attack_rate - self.source_attack_rate) * alpha, -0.08, 0.08))
            attack_prob = np.clip(attack_prob + bias, 1e-3, 1.0 - 1e-3)
            probabilities.append(np.column_stack([1.0 - attack_prob, attack_prob]))

            trace_rows.append(
                {
                    "batch_start": start,
                    "batch_end": stop,
                    "mean_drift_score": round(batch_drift, 4),
                    "ema_drift_score": round(ema_drift, 4),
                    "adaptation_alpha": round(alpha, 4),
                    "signature_weight": round(float(weights[0]), 4),
                    "random_forest_weight": round(float(weights[1]), 4),
                    "lstm_weight": round(float(weights[2]), 4),
                    "transformer_weight": round(float(weights[3]), 4),
                    "stacked_meta_weight": round(float(weights[4]), 4),
                    "rolling_attack_rate": round(rolling_attack_rate, 4),
                    "bias": round(bias, 4),
                }
            )

        self.last_adaptation_trace = pd.DataFrame(trace_rows)
        return np.vstack(probabilities)

    def predict(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        return (self.predict_proba(raw_df, X_scaled)[:, 1] >= 0.5).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "meta_model": self.meta_model,
                "drift_detector": self.drift_detector,
                "fitted": self.fitted,
                "stable_weights": self.stable_weights,
                "stressed_weights": self.stressed_weights,
                "drift_stats": self.drift_stats,
                "source_attack_rate": self.source_attack_rate,
                "adaptation_start": self.adaptation_start,
                "adaptation_end": self.adaptation_end,
                "ema_decay": self.ema_decay,
            },
            path,
        )

    def load_state(self, path: str) -> None:
        payload = joblib.load(path)
        self.meta_model = payload["meta_model"]
        self.drift_detector = payload["drift_detector"]
        self.fitted = payload["fitted"]
        self.stable_weights = payload.get("stable_weights", self.stable_weights)
        self.stressed_weights = payload.get("stressed_weights", self.stressed_weights)
        self.drift_stats = payload.get("drift_stats", self.drift_stats)
        self.source_attack_rate = payload.get("source_attack_rate", self.source_attack_rate)
        self.adaptation_start = payload.get("adaptation_start", self.adaptation_start)
        self.adaptation_end = payload.get("adaptation_end", self.adaptation_end)
        self.ema_decay = payload.get("ema_decay", self.ema_decay)
        self.last_adaptation_trace = pd.DataFrame()
