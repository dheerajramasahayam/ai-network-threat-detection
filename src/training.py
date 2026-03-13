"""
training.py
-----------
Main training script for the AI Network Intrusion Detection System.
Trains RandomForest and XGBoost models, saves artifacts, and generates
performance reports (metrics + plots).

Usage
-----
    # Default — CICIDS2017 only
    python3 src/training.py

    # Multi-dataset (cross-dataset generalization)
    python3 src/training.py --dataset dataset/combined.csv --label-col label
"""

import os
import sys
import json
import time
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preprocessing import preprocess
from src.model import ModelFactory, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, 'results', 'logs.txt'), mode='w'),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['BENIGN', 'ATTACK'],
        yticklabels=['BENIGN', 'ATTACK'],
        ax=ax, linewidths=0.5, linecolor='gray',
    )
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_roc_curves(results: dict, save_path: str):
    """Overlay ROC curves for all trained models."""
    colors = ['#2563EB', '#DC2626', '#16A34A', '#9333EA']
    fig, ax = plt.subplots(figsize=(8, 7))
    for (name, data), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — Intrusion Detection Models', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


def write_accuracy_report(all_metrics: dict, save_path: str):
    lines = [
        "# Accuracy Report — AI Network Intrusion Detection System\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Summary\n\n",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for model_name, metrics in all_metrics.items():
        lines.append(
            f"| {model_name} "
            f"| {metrics['accuracy']*100:.2f}% "
            f"| {metrics['precision']*100:.2f}% "
            f"| {metrics['recall']*100:.2f}% "
            f"| {metrics['f1_score']*100:.2f}% "
            f"| {metrics['roc_auc']:.4f} |\n"
        )
    lines += [
        "\n## Notes\n\n",
        f"- Dataset: {os.path.basename(save_path).replace('accuracy_report_','').replace('.md','')}\n",
        "- Train/Test split: 80/20 stratified\n",
        "- Class balancing: Random undersampling of majority class\n",
        "- Feature scaling: StandardScaler (zero-mean, unit-variance)\n",
    ]
    with open(save_path, 'w') as f:
        f.writelines(lines)
    logger.info(f"Accuracy report written → {save_path}")


def train_and_evaluate(dataset_path: str, label_col: str = "Label", tag: str = ""):
    """
    Full train-evaluate pipeline.

    Parameters
    ----------
    dataset_path : path to CSV (cicids2017.csv or combined.csv)
    label_col    : name of the target column in the CSV
    tag          : short string appended to artifact filenames
    """
    tag = tag or os.path.splitext(os.path.basename(dataset_path))[0]
    results_dir = RESULTS_DIR
    models_dir  = MODELS_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Preprocessing ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STEP 1: Preprocessing  [{tag}]")
    logger.info("=" * 60)

    # Handle combined.csv (canonical lowercase column names)
    import pandas as pd
    sample = pd.read_csv(dataset_path, nrows=2)
    actual_label = next(
        (c for c in sample.columns if c.lower() == label_col.lower()),
        label_col
    )

    data = preprocess(
        dataset_path,
        balance_strategy='undersample',
        label_col_override=actual_label if actual_label != 'Label' else None,
    )

    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_test  = data['y_test']
    feature_names = data['feature_names']
    logger.info(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ── Train models ─────────────────────────────────────────────────
    config = ModelConfig()
    models = ModelFactory.create_all(config)
    all_metrics = {}
    roc_data = {}

    for model_name, model in models.items():
        logger.info("=" * 60)
        logger.info(f"STEP 2: Training {model_name.upper()}")
        logger.info("=" * 60)
        t0 = time.time()
        model.fit(X_train, y_train, feature_names=feature_names)
        elapsed = time.time() - t0
        logger.info(f"Training time: {elapsed:.1f}s")

        logger.info(f"Evaluating {model_name.upper()} …")
        metrics = model.evaluate(X_test, y_test)
        all_metrics[model_name] = metrics

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        roc_data[model_name] = {'y_test': y_test, 'y_prob': y_prob}

        cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name}_{tag}.png")
        plot_confusion_matrix(y_test, y_pred, f"{model_name.replace('_',' ').title()} [{tag}]", cm_path)

        model_path = os.path.join(models_dir, f"{model_name}_{tag}.joblib")
        model.save(model_path)
        logger.info(f"Model saved → {model_path}")

        fi = model.feature_importance()
        if fi:
            top10 = list(fi.items())[:10]
            logger.info("Top-10 feature importances:")
            for feat, importance in top10:
                logger.info(f"  {feat:<40s} {importance:.4f}")

    # Canonical confusion matrix (spec requirement)
    rf_cm = os.path.join(results_dir, f"confusion_matrix_random_forest_{tag}.png")
    canonical = os.path.join(results_dir, "confusion_matrix.png")
    if os.path.exists(rf_cm):
        import shutil
        shutil.copy(rf_cm, canonical)

    # ── Plots & reports ──────────────────────────────────────────────
    plot_roc_curves(roc_data, os.path.join(results_dir, f"roc_curve_{tag}.png"))
    write_accuracy_report(all_metrics, os.path.join(results_dir, f"accuracy_report_{tag}.md"))

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE [{tag}]")
    logger.info("=" * 60)
    for name, m in all_metrics.items():
        logger.info(
            f"{name:<18s}  acc={m['accuracy']*100:.2f}%  "
            f"prec={m['precision']*100:.2f}%  "
            f"rec={m['recall']*100:.2f}%  "
            f"f1={m['f1_score']*100:.2f}%  "
            f"auc={m['roc_auc']:.4f}"
        )
    return all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default=os.path.join(BASE_DIR, 'dataset', 'cicids2017.csv'),
        help='Path to training CSV (default: dataset/cicids2017.csv)',
    )
    parser.add_argument(
        '--label-col', default='Label',
        help='Name of the label column in the CSV (default: Label)',
    )
    parser.add_argument(
        '--tag', default='',
        help='Short tag appended to artifact filenames (auto-detected from filename)',
    )
    args = parser.parse_args()
    train_and_evaluate(args.dataset, label_col=args.label_col, tag=args.tag)
