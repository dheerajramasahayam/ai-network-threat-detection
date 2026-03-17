from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression


class DriftAwareHybridDetector:
    """
    Proposed method: a drift-aware hybrid that fuses a traditional signature IDS
    score with RF/LSTM/Transformer probabilities and a drift detector signal.
    """

    model_name = "Drift-Aware Hybrid"

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

    def _drift_score(self, X_scaled: np.ndarray) -> np.ndarray:
        raw = -self.drift_detector.score_samples(X_scaled)
        raw = np.asarray(raw, dtype=np.float32)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return raw

    def _meta_features(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        rule_prob = self.signature_model.predict_proba(raw_df)[:, 1]
        rf_prob = self.rf_model.predict_proba(X_scaled)[:, 1]
        lstm_prob = self.lstm_model.predict_proba(X_scaled)[:, 1]
        transformer_prob = self.transformer_model.predict_proba(X_scaled)[:, 1]
        drift_prob = self._drift_score(X_scaled)
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

    def fit(
        self,
        train_scaled: np.ndarray,
        val_raw_df: pd.DataFrame,
        val_scaled: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.drift_detector.fit(train_scaled)
        meta_X = self._meta_features(val_raw_df, val_scaled)
        self.meta_model.fit(meta_X, y_val)
        self.fitted = True

    def predict_proba(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.meta_model.predict_proba(self._meta_features(raw_df, X_scaled))

    def predict(self, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        return (self.predict_proba(raw_df, X_scaled)[:, 1] >= 0.5).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "meta_model": self.meta_model,
                "drift_detector": self.drift_detector,
                "fitted": self.fitted,
            },
            path,
        )

    def load_state(self, path: str) -> None:
        payload = joblib.load(path)
        self.meta_model = payload["meta_model"]
        self.drift_detector = payload["drift_detector"]
        self.fitted = payload["fitted"]
