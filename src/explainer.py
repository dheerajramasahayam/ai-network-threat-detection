"""
explainer.py
------------
SHAP-based Explainability for the AI-NIDS.

Provides per-prediction explanations: WHY was this flow classified as an
ATTACK or BENIGN? Which features pushed the model toward that decision?

This is the #1 differentiator in 2025-2026 research — security analysts
need actionable reasons, not just labels.

Usage
-----
    from src.explainer import ThreatExplainer
    from src.model import IntrusionDetectionModel

    model = IntrusionDetectionModel.load('models/random_forest.joblib')
    explainer = ThreatExplainer(model)
    explainer.fit_background(X_train_sample)   # one-time setup

    report = explainer.explain(X_single_flow)
    print(report.summary())
    report.plot()

Supported model types: random_forest, xgboost (TreeExplainer — exact, fast)
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.model import IntrusionDetectionModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Explanation result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    """Holds SHAP values and human-readable explanation for one flow."""
    label: str
    attack_probability: float
    shap_values: np.ndarray          # shape (n_features,)
    feature_names: list[str]
    base_value: float                # expected model output over background
    top_k: int = 10

    # ── Derived ──────────────────────────────────────────────────────

    @property
    def top_positive(self) -> list[tuple[str, float]]:
        """Features that most pushed toward ATTACK (positive SHAP)."""
        pairs = list(zip(self.feature_names, self.shap_values))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:self.top_k]

    @property
    def top_negative(self) -> list[tuple[str, float]]:
        """Features that most pushed toward BENIGN (negative SHAP)."""
        pairs = list(zip(self.feature_names, self.shap_values))
        return sorted(pairs, key=lambda x: x[1])[:self.top_k]

    @property
    def top_absolute(self) -> list[tuple[str, float]]:
        """Top-k features by absolute SHAP magnitude."""
        pairs = list(zip(self.feature_names, self.shap_values))
        return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:self.top_k]

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  SHAP Explanation — {self.label}",
            f"  Attack probability: {self.attack_probability:.4f}",
            f"  Model base value:   {self.base_value:.4f}",
            f"{'='*55}",
            f"  Top features pushing → ATTACK:",
        ]
        for feat, val in self.top_positive:
            bar = "▓" * min(20, int(abs(val) * 100))
            lines.append(f"    {feat:<38s}  +{val:+.4f}  {bar}")
        lines.append(f"  Top features pushing → BENIGN:")
        for feat, val in self.top_negative:
            bar = "░" * min(20, int(abs(val) * 100))
            lines.append(f"    {feat:<38s}  {val:+.4f}  {bar}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label":             self.label,
            "attack_probability": round(self.attack_probability, 6),
            "base_value":        round(self.base_value, 6),
            "feature_shap":      {n: round(float(v), 6)
                                  for n, v in zip(self.feature_names, self.shap_values)},
            "top_attack_drivers": [{"feature": n, "shap": round(float(v), 6)}
                                   for n, v in self.top_positive],
            "top_benign_drivers": [{"feature": n, "shap": round(float(v), 6)}
                                   for n, v in self.top_negative],
        }

    def plot(self, save_path: Optional[str] = None, title: str = "SHAP Waterfall"):
        """Horizontal bar chart of top SHAP values."""
        feats, vals = zip(*self.top_absolute)
        colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(feats))
        ax.barh(y, vals, color=colors, edgecolor="none", height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=10)
        ax.axvline(0, color="#64748b", linewidth=1)
        ax.set_xlabel("SHAP value (impact on attack probability)", fontsize=11)
        ax.set_title(
            f"{title}\n{self.label}  (attack_prob={self.attack_probability:.3f})",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#dc2626", label="Pushes → ATTACK"),
            Patch(facecolor="#16a34a", label="Pushes → BENIGN"),
        ], loc="lower right", fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP plot saved → {save_path}")
        else:
            plt.show()
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# ThreatExplainer
# ──────────────────────────────────────────────────────────────────────────────

class ThreatExplainer:
    """
    Wraps a trained IntrusionDetectionModel with SHAP TreeExplainer
    to generate per-prediction explanations.

    Parameters
    ----------
    model        : trained IntrusionDetectionModel
    background_n : number of background samples for KernelExplainer fallback
    """

    def __init__(self, model: IntrusionDetectionModel, background_n: int = 100):
        self._check_shap()
        self.model = model
        self.background_n = background_n
        self._explainer = None
        self._base_value = 0.0

    @staticmethod
    def _check_shap():
        try:
            import shap  # noqa: F401
        except ImportError:
            raise ImportError(
                "shap is required. Install with: pip install shap"
            )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit_background(self, X_background: np.ndarray) -> None:
        """
        Build the SHAP explainer from a background sample.
        Must be called once before explain().

        For tree models (RF/XGBoost) this uses the fast TreeExplainer
        (exact Shapley values, no background needed).
        For other models it falls back to KernelExplainer.
        """
        import shap

        clf = self.model.model

        if self.model.model_type in ("random_forest", "xgboost"):
            logger.info(f"Building TreeExplainer for {self.model.model_type} …")
            self._explainer = shap.TreeExplainer(clf)
            # Compute base value from background
            sv = self._explainer.shap_values(X_background[:50])
            if isinstance(sv, list):
                sv = sv[1]  # class-1 SHAP for RF multiclass output
            self._base_value = float(self._explainer.expected_value
                                     if not hasattr(self._explainer.expected_value, "__len__")
                                     else self._explainer.expected_value[1])
        else:
            logger.info("Falling back to KernelExplainer …")
            n = min(self.background_n, len(X_background))
            bg = shap.sample(X_background, n, random_state=42)
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba, bg
            )
            self._base_value = float(self._explainer.expected_value)

        logger.info(f"ThreatExplainer ready. Base value: {self._base_value:.4f}")

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        top_k: int = 10,
        attack_prob: Optional[float] = None,
        threshold: float = 0.5,
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a single flow (first row of X).

        Parameters
        ----------
        X           : ndarray of shape (1, n_features) or (n_features,)
        top_k       : number of top features to highlight
        attack_prob : pre-computed probability (reuses if already computed)
        threshold   : decision threshold for labeling
        """
        if self._explainer is None:
            raise RuntimeError("Call fit_background() before explain().")

        import shap
        X = np.atleast_2d(X)

        # SHAP values
        sv = self._explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]                     # RF: class-1 SHAP
        shap_row = sv[0]                   # shape (n_features,)

        # Probability
        if attack_prob is None:
            attack_prob = float(self.model.predict_proba(X)[0])

        label = "ATTACK" if attack_prob >= threshold else "BENIGN"

        feature_names = (
            self.model.feature_names
            or [f"feature_{i}" for i in range(X.shape[1])]
        )

        return ExplanationResult(
            label=label,
            attack_probability=attack_prob,
            shap_values=shap_row,
            feature_names=feature_names,
            base_value=self._base_value,
            top_k=top_k,
        )

    def explain_batch(
        self, X: np.ndarray, top_k: int = 10, threshold: float = 0.5
    ) -> list[ExplanationResult]:
        """Explain every row in X."""
        probs = self.model.predict_proba(X)
        return [
            self.explain(X[i:i+1], top_k=top_k,
                         attack_prob=float(probs[i]), threshold=threshold)
            for i in range(len(X))
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"ThreatExplainer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ThreatExplainer":
        import joblib
        explainer = joblib.load(path)
        logger.info(f"ThreatExplainer loaded ← {path}")
        return explainer

    def global_importance_plot(
        self, X_sample: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        """
        Plot mean absolute SHAP values for a sample — shows overall
        which features matter most across many flows.
        """
        import shap
        sv = self._explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        feature_names = (
            self.model.feature_names
            or [f"feature_{i}" for i in range(X_sample.shape[1])]
        )
        order = np.argsort(mean_abs)[-20:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            [feature_names[i] for i in order],
            mean_abs[order],
            color="#38bdf8", edgecolor="none",
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Global Feature Importance (SHAP)\nMean absolute impact on attack probability",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Global SHAP importance plot saved → {save_path}")
        else:
            plt.show()
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, joblib
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "random_forest.joblib"))
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Background samples for explainer calibration")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "results", "shap_explanation.png"))
    args = parser.parse_args()

    model = IntrusionDetectionModel.load(args.model)
    rng = np.random.default_rng(42)
    n_features = len(model.feature_names) or 41

    X_background = rng.normal(size=(args.n_samples, n_features))
    X_flow = rng.normal(loc=4.0, size=(1, n_features))  # attack-like

    exp = ThreatExplainer(model)
    exp.fit_background(X_background)
    result = exp.explain(X_flow)
    print(result.summary())
    result.plot(save_path=args.out)
    exp.global_importance_plot(X_background,
                               save_path=args.out.replace(".png", "_global.png"))
    print(f"\nPlots saved to {args.out}")
