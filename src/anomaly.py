"""
anomaly.py
----------
Zero-day / unsupervised anomaly detection layer for the AI-NIDS.

WHY this matters
----------------
Supervised models (RF, XGBoost, CNN) can only detect attacks they were
trained on. Zero-day exploits — attacks with no known signature — are
completely invisible to them.

This module adds a complementary unsupervised ensemble that learns what
NORMAL traffic looks like from unlabelled data, then flags *anything
statistically unusual* as a potential zero-day threat.

Architecture: 3-detector ensemble vote
  1. Isolation Forest  — efficient outlier detection via random partitioning
  2. One-Class SVM     — learns a tight decision boundary around normal traffic
  3. Local Outlier Factor (LOF) — density-based anomaly scoring

A flow is flagged as ZERO-DAY if ≥ 2 of 3 detectors report it as anomalous
(majority vote). This minimises false positives vs any single detector.

Usage
-----
    from src.anomaly import ZeroDayDetector
    detector = ZeroDayDetector()
    detector.fit(X_benign_train)            # train on BENIGN traffic only

    result = detector.inspect(X_flow)
    print(result)                            # ZeroDayResult
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import joblib

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZeroDayResult:
    """Anomaly detection result for one flow."""
    is_anomaly: bool
    anomaly_score: float              # higher = more anomalous [0, 1]
    votes: dict[str, bool]           # per-detector vote
    vote_count: int                  # how many detectors flagged it
    detector_scores: dict[str, float]
    latency_ms: float

    @property
    def label(self) -> str:
        return "ZERO-DAY" if self.is_anomaly else "NORMAL"

    @property
    def risk_level(self) -> str:
        if self.anomaly_score >= 0.75:  return "CRITICAL"
        if self.anomaly_score >= 0.50:  return "HIGH"
        if self.anomaly_score >= 0.25:  return "MEDIUM"
        return "LOW"

    def __str__(self) -> str:
        vote_str = " | ".join(f"{k}={'🔴' if v else '🟢'}" for k, v in self.votes.items())
        return (
            f"[{self.label}] risk={self.risk_level}  "
            f"score={self.anomaly_score:.4f}  "
            f"votes={self.vote_count}/3  "
            f"latency={self.latency_ms:.2f}ms\n"
            f"   {vote_str}"
        )

    def to_dict(self) -> dict:
        return {
            "label":            self.label,
            "is_anomaly":       self.is_anomaly,
            "anomaly_score":    round(self.anomaly_score, 6),
            "risk_level":       self.risk_level,
            "votes":            self.votes,
            "vote_count":       self.vote_count,
            "detector_scores":  {k: round(v, 6) for k, v in self.detector_scores.items()},
            "latency_ms":       round(self.latency_ms, 3),
        }


# ──────────────────────────────────────────────────────────────────────────────
# ZeroDayDetector
# ──────────────────────────────────────────────────────────────────────────────

class ZeroDayDetector:
    """
    Ensemble of 3 unsupervised anomaly detectors.

    Parameters
    ----------
    contamination : expected fraction of anomalies in training data (default 0.01)
    vote_threshold : how many detectors must agree to flag ZERO-DAY (default 2)
    n_jobs        : parallel jobs for Isolation Forest
    """

    DETECTOR_NAMES = ("isolation_forest", "one_class_svm", "lof")

    def __init__(
        self,
        contamination: float = 0.01,
        vote_threshold: int = 2,
        n_jobs: int = -1,
    ):
        self.contamination = contamination
        self.vote_threshold = vote_threshold
        self.n_jobs = n_jobs
        self._detectors: dict = {}
        self.is_fitted = False
        self._score_mins: dict[str, float] = {}
        self._score_maxs: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_benign: np.ndarray) -> None:
        """
        Train all three detectors on BENIGN-only traffic.
        The models learn 'normal' and will flag deviations.

        Parameters
        ----------
        X_benign : ndarray of shape (n_samples, n_features)
                   should contain ONLY benign flow features (not attacks)
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor

        logger.info(
            f"Training ZeroDayDetector on {len(X_benign):,} benign samples "
            f"(contamination={self.contamination}) …"
        )

        # 1. Isolation Forest
        logger.info("  Fitting IsolationForest …")
        iso = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
            n_jobs=self.n_jobs,
        )
        iso.fit(X_benign)

        # 2. One-Class SVM  (RBF kernel, trained on subset for speed)
        logger.info("  Fitting One-Class SVM …")
        n_svm = min(10_000, len(X_benign))
        idx = np.random.default_rng(42).choice(len(X_benign), n_svm, replace=False)
        ocsvm = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
        ocsvm.fit(X_benign[idx])

        # 3. Local Outlier Factor (novelty=True for predict on unseen data)
        logger.info("  Fitting LOF …")
        n_lof = min(20_000, len(X_benign))
        idx2 = np.random.default_rng(0).choice(len(X_benign), n_lof, replace=False)
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,
            n_jobs=self.n_jobs,
        )
        lof.fit(X_benign[idx2])

        self._detectors = {
            "isolation_forest": iso,
            "one_class_svm":   ocsvm,
            "lof":             lof,
        }

        # Calibrate score ranges on held-out benign data for normalisation
        self._calibrate_scores(X_benign[:5_000])
        self.is_fitted = True
        logger.info("ZeroDayDetector ready.")

    def _calibrate_scores(self, X: np.ndarray) -> None:
        """Compute min/max raw scores on benign data for [0,1] normalisation."""
        for name, det in self._detectors.items():
            raw = -det.score_samples(X)   # negate so higher = more anomalous
            self._score_mins[name] = float(raw.min())
            self._score_maxs[name] = float(raw.max())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inspect(self, X: np.ndarray) -> ZeroDayResult:
        """
        Classify a single flow as NORMAL or ZERO-DAY.

        Parameters
        ----------
        X : ndarray of shape (1, n_features) or (n_features,)
        """
        self._check_fitted()
        X = np.atleast_2d(X)
        t0 = time.perf_counter()

        votes = {}
        raw_scores = {}

        for name, det in self._detectors.items():
            pred = det.predict(X)[0]          # +1 = normal, -1 = anomaly
            raw = float(-det.score_samples(X)[0])  # higher = more anomalous
            votes[name] = (pred == -1)
            raw_scores[name] = raw

        # Normalise scores to [0,1] per detector then average
        norm_scores = {}
        for name, raw in raw_scores.items():
            lo = self._score_mins.get(name, raw)
            hi = self._score_maxs.get(name, raw)
            span = hi - lo if hi > lo else 1.0
            norm_scores[name] = float(np.clip((raw - lo) / span, 0.0, 1.0))

        anomaly_score = float(np.mean(list(norm_scores.values())))
        vote_count = sum(votes.values())
        is_anomaly = vote_count >= self.vote_threshold
        latency_ms = (time.perf_counter() - t0) * 1000

        return ZeroDayResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            votes=votes,
            vote_count=vote_count,
            detector_scores=norm_scores,
            latency_ms=latency_ms,
        )

    def inspect_batch(self, X: np.ndarray) -> list[ZeroDayResult]:
        """Classify multiple flows."""
        return [self.inspect(X[i:i+1]) for i in range(len(X))]

    def anomaly_score_batch(self, X: np.ndarray) -> np.ndarray:
        """Return just the composite anomaly score for each row (fast path)."""
        self._check_fitted()
        scores = []
        for name, det in self._detectors.items():
            raw = -det.score_samples(X)
            lo = self._score_mins.get(name, raw.min())
            hi = self._score_maxs.get(name, raw.max())
            span = hi - lo if hi > lo else 1.0
            scores.append(np.clip((raw - lo) / span, 0.0, 1.0))
        return np.mean(scores, axis=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"ZeroDayDetector saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ZeroDayDetector":
        logger.info(f"Loading ZeroDayDetector ← {path}")
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("ZeroDayDetector not fitted. Call .fit(X_benign) first.")

    def plot_score_distribution(
        self, X_benign: np.ndarray, X_attack: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot composite anomaly score distributions for benign vs attack.
        Useful for visualising the detector's discrimination ability.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        scores_b = self.anomaly_score_batch(X_benign)
        scores_a = self.anomaly_score_batch(X_attack)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(scores_b, bins=60, alpha=0.6, color="#16a34a", label="Benign", density=True)
        ax.hist(scores_a, bins=60, alpha=0.6, color="#dc2626", label="Attack", density=True)
        ax.axvline(0.5, color="#f59e0b", linestyle="--", label="Default threshold (0.5)")
        ax.set_xlabel("Composite Anomaly Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            "Zero-Day Detector — Anomaly Score Distribution\n"
            "Benign vs Known Attacks (should separate well → catches zero-days too)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Score distribution plot saved → {save_path}")
        else:
            plt.show()
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Zero-day training helper (standalone script)
# ──────────────────────────────────────────────────────────────────────────────

def train_zero_day_detector(dataset_path: str, out_path: str) -> ZeroDayDetector:
    """
    Load CICIDS2017, extract BENIGN-only samples, train the ensemble,
    save the detector artifact.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE)
    from src.preprocessing import FEATURE_COLUMNS, LABEL_COLUMN

    logger.info("Loading dataset for zero-day detector training …")
    df = pd.read_csv(dataset_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    benign_mask = df[LABEL_COLUMN].str.strip().str.upper() == "BENIGN"
    X_benign = df.loc[benign_mask, available].values

    # Scale
    scaler = StandardScaler()
    X_benign_s = scaler.fit_transform(X_benign)

    # Sample up to 100K benign rows for training (plenty for unsupervised)
    n = min(100_000, len(X_benign_s))
    idx = np.random.default_rng(42).choice(len(X_benign_s), n, replace=False)
    X_train = X_benign_s[idx]

    detector = ZeroDayDetector(contamination=0.01)
    detector.fit(X_train)
    detector.save(out_path)
    logger.info(f"Zero-day detector saved → {out_path}")
    return detector


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=os.path.join(BASE, "dataset", "cicids2017.csv"))
    parser.add_argument("--out",
                        default=os.path.join(BASE, "models", "zero_day_detector.joblib"))
    args = parser.parse_args()

    detector = train_zero_day_detector(args.dataset, args.out)

    # Quick sanity test with random data
    rng = np.random.default_rng(42)
    n_features = 41
    print("\n── Sanity check (synthetic flows) ──")
    for label, X in [
        ("Benign-like", rng.normal(loc=0.0, scale=1.0, size=(1, n_features))),
        ("Attack-like", rng.normal(loc=6.0, scale=1.5, size=(1, n_features))),
    ]:
        r = detector.inspect(X)
        print(f"  {label:14s} → {r}")
