"""
adversarial.py
--------------
Adversarial Robustness Test for AI-NIDS.

Research gap addressed
----------------------
Papers report accuracy on clean data. Zero papers in the open-source NIDS
space test how well models resist *adversarially crafted* traffic — flows
specifically modified to evade detection.

Methods implemented
-------------------
1. FGSM-style feature perturbation (sign-gradient attack adapted for tabular data)
2. Random noise perturbation (baseline)
3. Feature masking attack (zeroing top SHAP features)
4. Boundary walk (iterative gradient-free attack until misclassified)

Produces
--------
  results/adversarial_report.md        — robustness metrics per attack method
  results/adversarial_robustness.png   — evasion rate chart

Usage
-----
    python3 src/adversarial.py --model models/random_forest_cicids2017.joblib
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.preprocessing import FEATURE_COLUMNS, _CICIDS_COL_MAP
from src.model import IntrusionDetectionModel


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AdversarialResult:
    method: str
    epsilon: float
    evasion_rate: float      # % of attacks successfully misclassified as BENIGN
    avg_perturbation: float  # L2 norm of perturbation
    n_tested: int
    time_ms: float

    def __str__(self):
        return (
            f"{self.method:<30s}  ε={self.epsilon:.3f}  "
            f"evasion={self.evasion_rate:.2f}%  "
            f"|δ|={self.avg_perturbation:.4f}  "
            f"n={self.n_tested}  {self.time_ms:.1f}ms"
        )


# ── Attack methods ────────────────────────────────────────────────────────────

def _random_noise(X: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Baseline: add uniform random noise scaled by ε × feature std."""
    noise = rng.uniform(-1, 1, X.shape) * epsilon
    return X + noise


def _sign_gradient_attack(
    X: np.ndarray,
    model: IntrusionDetectionModel,
    epsilon: float,
    n_steps: int = 10,
) -> np.ndarray:
    """
    FGSM-inspired sign-gradient attack for black-box tree models.
    Uses finite differences to approximate gradient direction,
    then perturbs features in the direction that decreases attack probability.
    """
    X_adv = X.copy()
    step_size = epsilon / n_steps

    for _ in range(n_steps):
        prob = model.predict_proba(X_adv)
        # Finite-difference gradient approximation
        grad = np.zeros_like(X_adv)
        for j in range(X_adv.shape[1]):
            X_plus = X_adv.copy(); X_plus[:, j] += 1e-3
            X_minus = X_adv.copy(); X_minus[:, j] -= 1e-3
            prob_plus  = model.predict_proba(X_plus)
            prob_minus = model.predict_proba(X_minus)
            grad[:, j] = (prob_plus - prob_minus) / (2e-3)

        # Step in the sign direction that reduces attack probability
        X_adv -= step_size * np.sign(grad)

    return X_adv


def _feature_masking(
    X: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> np.ndarray:
    """
    Zero out the top-k most important features.
    Simulates an attacker who knows SHAP importances and sets those
    discriminative features to 0 (e.g. sending minimum-size packets).
    """
    # Use simple variance-based importance as proxy (avoids needing SHAP at runtime)
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:top_k]
    X_adv = X.copy()
    X_adv[:, top_idx] = 0.0
    return X_adv


def _boundary_walk(
    X: np.ndarray,
    model: IntrusionDetectionModel,
    epsilon: float,
    max_steps: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Gradient-free boundary walk: randomly perturb until misclassified,
    bounded by ε. Simulates an attacker with only black-box API access.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_adv = X.copy()
    prob = model.predict_proba(X_adv)
    still_attack = prob >= 0.5

    step = epsilon / max_steps
    for _ in range(max_steps):
        if not still_attack.any():
            break
        idx = np.where(still_attack)[0]
        delta = rng.normal(0, step, (len(idx), X.shape[1]))
        X_adv[idx] += delta
        # Clip total perturbation
        total_delta = X_adv[idx] - X[idx]
        norms = np.linalg.norm(total_delta, axis=1, keepdims=True)
        over = norms > epsilon
        X_adv[idx] = np.where(over, X[idx] + total_delta / norms * epsilon, X_adv[idx])
        prob = model.predict_proba(X_adv)
        still_attack = prob >= 0.5

    return X_adv


# ── Main robustness test ──────────────────────────────────────────────────────

def run_adversarial_test(
    model: IntrusionDetectionModel,
    X_attacks: np.ndarray,
    feature_names: list[str],
    epsilons: list[float] | None = None,
    n_samples: int = 500,
) -> list[AdversarialResult]:
    """
    Run all 4 adversarial methods across multiple ε values.

    Parameters
    ----------
    model         : trained IntrusionDetectionModel
    X_attacks     : scaled feature array of true ATTACK flows only
    feature_names : feature column names
    epsilons      : perturbation magnitudes to sweep
    n_samples     : number of attack flows to test per epsilon
    """
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2, 0.5, 1.0]

    rng = np.random.default_rng(42)
    n = min(n_samples, len(X_attacks))
    idx = rng.choice(len(X_attacks), n, replace=False)
    X_test = X_attacks[idx]

    # Confirm all are initially classified as ATTACK
    probs = model.predict_proba(X_test)
    initially_attack = X_test[probs >= 0.5]
    if len(initially_attack) < 10:
        logger.warning("Very few samples initially classified as ATTACK — results may be unreliable.")
        initially_attack = X_test  # use all

    logger.info(f"Testing {len(initially_attack)} true-attack flows …")
    results: list[AdversarialResult] = []

    for eps in epsilons:
        logger.info(f"\n── ε = {eps} ──")

        for method_name, attack_fn in [
            ("Random Noise",        lambda X: _random_noise(X, eps, rng)),
            ("Sign-Gradient (FGSM)", lambda X: _sign_gradient_attack(X, model, eps, n_steps=5)),
            ("Feature Masking",     lambda X: _feature_masking(X, feature_names, top_k=5)),
            ("Boundary Walk",       lambda X: _boundary_walk(X, model, eps, max_steps=30, rng=rng)),
        ]:
            t0 = time.perf_counter()
            X_adv = attack_fn(initially_attack)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            prob_adv = model.predict_proba(X_adv)
            evaded = (prob_adv < 0.5).sum()
            evasion_rate = 100.0 * evaded / len(initially_attack)
            avg_pert = float(np.mean(np.linalg.norm(X_adv - initially_attack, axis=1)))

            r = AdversarialResult(
                method=method_name,
                epsilon=eps,
                evasion_rate=round(evasion_rate, 2),
                avg_perturbation=round(avg_pert, 4),
                n_tested=len(initially_attack),
                time_ms=round(elapsed_ms, 1),
            )
            results.append(r)
            logger.info(f"  {r}")

    return results


def save_report(results: list[AdversarialResult], out_md: str, out_png: str):
    # Markdown
    lines = [
        "# Adversarial Robustness Report\n\n",
        "Tests how well the model withstands adversarially crafted flows.\n",
        "**Evasion rate** = % of true attack flows re-classified as BENIGN after perturbation.\n",
        "Lower evasion rate = more robust model.\n\n",
        "| Method | ε | Evasion Rate | Avg Perturbation |δ| | n Tested |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in results:
        flag = "🔴" if r.evasion_rate > 30 else "🟡" if r.evasion_rate > 10 else "🟢"
        lines.append(
            f"| {flag} {r.method} | {r.epsilon} "
            f"| **{r.evasion_rate:.2f}%** | {r.avg_perturbation:.4f} "
            f"| {r.n_tested} |\n"
        )
    lines += [
        "\n🟢 Evasion ≤ 10% (robust)  🟡 10–30%  🔴 > 30% (vulnerable)\n\n",
        "## Methods\n\n",
        "- **Random Noise**: uniform noise ±ε (baseline)\n",
        "- **Sign-Gradient**: FGSM-inspired finite-difference gradient attack\n",
        "- **Feature Masking**: zero out top-k most discriminative features\n",
        "- **Boundary Walk**: gradient-free random walk to decision boundary\n",
    ]
    with open(out_md, "w") as f:
        f.writelines(lines)
    logger.info(f"Adversarial report saved → {out_md}")

    # Chart
    df = pd.DataFrame([vars(r) for r in results])
    methods = df["method"].unique()
    epsilons = sorted(df["epsilon"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]
    for method, color in zip(methods, colors):
        sub = df[df["method"] == method].sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["evasion_rate"], "o-",
                label=method, color=color, linewidth=2, markersize=7)
    ax.axhline(10, color="#64748b", linestyle="--", alpha=0.5, label="10% threshold (robust)")
    ax.axhline(30, color="#dc2626", linestyle="--", alpha=0.5, label="30% threshold (vulnerable)")
    ax.set_xlabel("Perturbation magnitude (ε)", fontsize=12)
    ax.set_ylabel("Evasion Rate (%)", fontsize=12)
    ax.set_title("Adversarial Robustness — Evasion Rate vs Perturbation\n"
                 "Lower = more robust to adversarial traffic",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Adversarial robustness chart saved → {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=os.path.join(BASE_DIR, "models", "random_forest_cicids2017.joblib"))
    parser.add_argument("--dataset",
                        default=os.path.join(BASE_DIR, "dataset", "cicids2017.csv"))
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    from sklearn.preprocessing import StandardScaler as _SS

    logger.info("Loading model and dataset …")
    model = IntrusionDetectionModel.load(args.model)

    df = pd.read_csv(args.dataset, low_memory=False)
    df.columns = df.columns.str.strip()
    rename = {k: v for k, v in _CICIDS_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

    label_col = "Label" if "Label" in df.columns else "label"
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]

    y_bin = (df[label_col].astype(str).str.upper() != "BENIGN").astype(int)
    X_all = df[feats].values
    scaler = _SS()
    X_s = scaler.fit_transform(X_all)
    X_attacks = X_s[y_bin == 1]
    logger.info(f"Attack samples for adversarial test: {len(X_attacks):,}")

    results = run_adversarial_test(
        model=model,
        X_attacks=X_attacks,
        feature_names=feats,
        epsilons=[0.05, 0.1, 0.2, 0.5, 1.0],
        n_samples=args.n_samples,
    )

    save_report(
        results,
        out_md=os.path.join(RESULTS_DIR, "adversarial_report.md"),
        out_png=os.path.join(RESULTS_DIR, "adversarial_robustness.png"),
    )
    print("\n✅  Adversarial test complete — see results/adversarial_report.md")
