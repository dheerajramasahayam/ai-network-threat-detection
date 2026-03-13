"""
detection.py
------------
Real-time network traffic inspection module.

Provides a ThreatDetector that loads trained model artifacts,
preprocesses incoming feature vectors, and classifies each
flow as BENIGN or ATTACK with a confidence score.
"""

from __future__ import annotations

import os
import sys
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.model import IntrusionDetectionModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    label: str                   # 'BENIGN' or 'ATTACK'
    confidence: float            # probability of the predicted class [0, 1]
    attack_probability: float    # raw probability of being an attack
    model_name: str
    latency_ms: float            # inference latency in milliseconds
    features_used: int           # number of feature dimensions processed
    metadata: dict = field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        return self.label == 'ATTACK'

    def __str__(self) -> str:
        return (
            f"[{self.label}] confidence={self.confidence:.1%}  "
            f"attack_prob={self.attack_probability:.4f}  "
            f"model={self.model_name}  latency={self.latency_ms:.2f}ms"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────────────

class ThreatDetector:
    """
    Loads a trained IntrusionDetectionModel artifact and exposes an
    `inspect` / `inspect_batch` interface for live traffic analysis.

    Parameters
    ----------
    model_path : str
        Path to the .joblib file produced by training.py.
    scaler_path : str | None
        Path to a serialised StandardScaler. If None, raw features are used.
    threshold : float
        Decision threshold for classifying a flow as ATTACK (default 0.5).
    """

    LABEL_MAP = {0: 'BENIGN', 1: 'ATTACK'}

    def __init__(
        self,
        model_path: str,
        scaler_path: str | None = None,
        threshold: float = 0.5,
    ):
        self.model: IntrusionDetectionModel = IntrusionDetectionModel.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.threshold = threshold
        self._alert_callbacks: list = []
        logger.info(
            f"ThreatDetector initialised — model={self.model.model_type}  "
            f"threshold={threshold}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self, features: dict[str, float] | list[float] | np.ndarray,
                metadata: dict | None = None) -> DetectionResult:
        """
        Classify a single network flow.

        Parameters
        ----------
        features : dict, list, or ndarray
            Feature values. If dict, keys must match the model's feature_names.
        metadata : dict (optional)
            Extra context (e.g. src_ip, dst_port) attached to the result.
        """
        X = self._to_array(features)
        t0 = time.perf_counter()
        attack_prob = float(self.model.predict_proba(X)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        predicted = 1 if attack_prob >= self.threshold else 0
        label = self.LABEL_MAP[predicted]
        confidence = attack_prob if predicted == 1 else 1.0 - attack_prob

        result = DetectionResult(
            label=label,
            confidence=confidence,
            attack_probability=attack_prob,
            model_name=self.model.model_type,
            latency_ms=latency_ms,
            features_used=X.shape[1],
            metadata=metadata or {},
        )

        if result.is_attack:
            self._fire_alert(result)

        return result

    def inspect_batch(
        self,
        feature_rows: list[dict | list | np.ndarray],
        metadata_list: list[dict] | None = None,
    ) -> list[DetectionResult]:
        """Classify multiple flows at once."""
        metadata_list = metadata_list or [{}] * len(feature_rows)
        return [
            self.inspect(row, meta)
            for row, meta in zip(feature_rows, metadata_list)
        ]

    def register_alert_callback(self, fn) -> None:
        """Register a callable that is invoked on every ATTACK detection."""
        self._alert_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_array(self, features: Any) -> np.ndarray:
        if isinstance(features, dict):
            if self.model.feature_names:
                arr = np.array([[features.get(f, 0.0) for f in self.model.feature_names]])
            else:
                arr = np.array([list(features.values())])
        elif isinstance(features, np.ndarray):
            arr = features.reshape(1, -1) if features.ndim == 1 else features
        else:
            arr = np.array([features])

        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return arr

    def _fire_alert(self, result: DetectionResult) -> None:
        for cb in self._alert_callbacks:
            try:
                cb(result)
            except Exception as exc:
                logger.warning(f"Alert callback raised an exception: {exc}")

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'f1',
        thresholds: list[float] | None = None,
    ) -> float:
        """
        Grid-search over thresholds and return the one maximising `metric`.
        Supported metrics: 'f1', 'precision', 'recall'.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        fn_map = {'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
        if metric not in fn_map:
            raise ValueError(f"metric must be one of {list(fn_map)}")
        score_fn = fn_map[metric]
        thresholds = thresholds or [t / 100 for t in range(10, 91, 5)]
        probs = self.model.predict_proba(X_val)
        best_thresh, best_score = 0.5, 0.0
        for t in thresholds:
            preds = (probs >= t).astype(int)
            s = score_fn(y_val, preds, zero_division=0)
            if s > best_score:
                best_score, best_thresh = s, t
        logger.info(f"Best threshold={best_thresh:.2f} → {metric}={best_score:.4f}")
        self.threshold = best_thresh
        return best_thresh


# ──────────────────────────────────────────────────────────────────────────────
# Demo / CLI
# ──────────────────────────────────────────────────────────────────────────────

def _demo_alert(result: DetectionResult):
    print(f"🚨 ALERT  {result}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AI Threat Detector — demo')
    parser.add_argument('--model', default=os.path.join(BASE_DIR, 'models', 'random_forest.joblib'))
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run src/training.py first.")
        sys.exit(1)

    detector = ThreatDetector(args.model, threshold=args.threshold)
    detector.register_alert_callback(_demo_alert)

    # Synthetic samples
    rng = np.random.default_rng(42)
    n_features = len(detector.model.feature_names) or 41

    print("\n──── Classifying 5 synthetic flows ────")
    for i in range(5):
        flow = rng.normal(size=n_features)
        result = detector.inspect(flow)
        print(f"  Flow {i+1}: {result}")
