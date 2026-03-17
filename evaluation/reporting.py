from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_prob.shape[1] > 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        )
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.25,
        linecolor="#d1d5db",
    )
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_roc_curves(
    probability_map: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#0f766e", "#b45309", "#1d4ed8", "#7c2d12"]

    for index, (model_name, (y_true, y_prob)) in enumerate(probability_map.items()):
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            auc_score = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        ax.plot(fpr, tpr, linewidth=2.0, color=palette[index % len(palette)], label=f"{model_name} (AUC={auc_score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#6b7280", label="Chance")
    ax.set_title("Macro ROC Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_feature_importance(importances: list[tuple[str, float]], save_path: str | Path, top_n: int = 15) -> None:
    top_features = importances[:top_n]
    labels = [name for name, _ in reversed(top_features)]
    scores = [score for _, score in reversed(top_features)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, scores, color="#0f766e")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_training_loss_curves(history_map: dict[str, dict[str, list[float]]], save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"LSTM": "#1d4ed8", "Transformer": "#b45309"}

    for model_name, history in history_map.items():
        if not history:
            continue
        ax.plot(history.get("loss", []), label=f"{model_name} train", linewidth=2, color=palette.get(model_name, "#111827"))
        if history.get("val_loss"):
            ax.plot(history["val_loss"], label=f"{model_name} val", linewidth=2, linestyle="--", color=palette.get(model_name, "#374151"))

    ax.set_title("Deep Learning Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def write_results_table(
    metrics_by_model: dict[str, dict[str, float]],
    csv_path: str | Path,
    markdown_path: str | Path,
) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"] * 100, 2),
                "Precision": round(metrics["precision"] * 100, 2),
                "Recall": round(metrics["recall"] * 100, 2),
                "F1 Score": round(metrics["f1_score"] * 100, 2),
                "ROC AUC": round(metrics["roc_auc"], 4),
                "Latency (ms/flow)": round(metrics["latency_ms_per_flow"], 4),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["F1 Score", "Accuracy"], ascending=False)
    results_df.to_csv(csv_path, index=False)

    markdown_lines = [
        "# Experimental Results\n\n",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |\n",
        "| --- | --- | --- | --- | --- | --- | --- |\n",
    ]

    for _, row in results_df.iterrows():
        markdown_lines.append(
            f"| {row['Model']} | {row['Accuracy']:.2f}% | {row['Precision']:.2f}% | "
            f"{row['Recall']:.2f}% | {row['F1 Score']:.2f}% | {row['ROC AUC']:.4f} | "
            f"{row['Latency (ms/flow)']:.4f} |\n"
        )

    Path(markdown_path).write_text("".join(markdown_lines))
    return results_df


def write_summary_json(summary: dict, save_path: str | Path) -> None:
    Path(save_path).write_text(json.dumps(summary, indent=2))
