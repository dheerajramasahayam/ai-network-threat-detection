from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from evaluation.reporting import (
    classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    plot_training_loss_curves,
    write_results_table,
    write_summary_json,
)
from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from src.preprocessing import FEATURE_COLUMNS

CLASS_NAMES = ["Benign", "Attack"]
FLOAT32_CLIP = np.finfo(np.float32).max / 1024.0


@dataclass
class ExperimentArtifacts:
    split_name: str
    results_df: pd.DataFrame
    metrics_by_model: dict[str, dict[str, float]]
    best_model_name: str
    hybrid_variant_name: str
    hybrid_variant_mode: str
    signature_model: SignatureIDSBaseline
    random_forest: RandomForestThreatDetector
    lstm_model: LSTMThreatDetector
    transformer_model: TransformerThreatDetector
    hybrid_model: DriftAwareHybridDetector
    split: object


def _latency_wrapper(callable_fn, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = callable_fn()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return float(np.mean(timings))


def _predict_hybrid_proba(
    experiment: ExperimentArtifacts,
    raw_df: pd.DataFrame,
    X_scaled: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    if experiment.hybrid_variant_mode == "static":
        return experiment.hybrid_model.predict_proba_static(raw_df, X_scaled)
    return experiment.hybrid_model.predict_proba(raw_df, X_scaled, batch_size=batch_size)


def _predict_proba(model_name: str, experiment: ExperimentArtifacts, raw_df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
    if model_name == "Signature IDS":
        return experiment.signature_model.predict_proba(raw_df)
    if model_name == "Random Forest":
        return experiment.random_forest.predict_proba(X_scaled)
    if model_name == "LSTM":
        return experiment.lstm_model.predict_proba(X_scaled)
    if model_name == "Transformer":
        return experiment.transformer_model.predict_proba(X_scaled)
    if model_name in {"Drift-Aware Hybrid", "Drift-Aware Hybrid (Static)", "Drift-Adaptive Hybrid", experiment.hybrid_variant_name}:
        return _predict_hybrid_proba(experiment, raw_df, X_scaled)
    raise KeyError(model_name)


def _transform_frame(experiment: ExperimentArtifacts, frame: pd.DataFrame) -> np.ndarray:
    values = (
        frame[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=-FLOAT32_CLIP, upper=FLOAT32_CLIP)
        .to_numpy(dtype=np.float64, copy=True)
    )
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=FLOAT32_CLIP,
        neginf=-FLOAT32_CLIP,
        copy=False,
    )
    return experiment.split.preprocessor.transform(values).astype(np.float32)


def _save_metadata(experiment: ExperimentArtifacts, artifacts_dir: Path, prefix: str) -> None:
    write_summary_json(
        {
            "split_name": experiment.split_name,
            "feature_names": FEATURE_COLUMNS,
            "class_names": CLASS_NAMES,
            "best_model": experiment.best_model_name,
            "artifacts": {
                "signature_ids": str((artifacts_dir / f"{prefix}_signature_ids.json")),
                "random_forest": str((artifacts_dir / f"{prefix}_random_forest.joblib")),
                "lstm": str((artifacts_dir / f"{prefix}_lstm.pt")),
                "transformer": str((artifacts_dir / f"{prefix}_transformer.pt")),
                "hybrid": str((artifacts_dir / f"{prefix}_hybrid.joblib")),
                "preprocessor": str((artifacts_dir / f"{prefix}_preprocessor.joblib")),
            },
            "results": experiment.metrics_by_model,
            "hybrid_variant_name": experiment.hybrid_variant_name,
            "hybrid_variant_mode": experiment.hybrid_variant_mode,
        },
        artifacts_dir / f"{prefix}_metadata.json",
    )


def run_model_family_experiment(
    split,
    results_dir: str | Path,
    artifacts_dir: str | Path,
    prefix: str,
    epochs: int = 3,
    batch_size: int = 256,
    rf_trees: int = 200,
    hybrid_variant_name: str = "Drift-Aware Hybrid (Static)",
    hybrid_variant_mode: str = "static",
) -> ExperimentArtifacts:
    results_dir = Path(results_dir)
    artifacts_dir = Path(artifacts_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if hybrid_variant_mode not in {"static", "adaptive"}:
        raise ValueError(f"Unsupported hybrid_variant_mode: {hybrid_variant_mode}")

    signature_model = SignatureIDSBaseline()
    signature_model.fit(split.train_df)

    random_forest = RandomForestThreatDetector(n_estimators=rf_trees)
    random_forest.fit(split.X_train, split.y_train)

    lstm_model = LSTMThreatDetector(
        input_dim=split.X_train.shape[1],
        num_classes=2,
        epochs=epochs,
        batch_size=batch_size,
    )
    lstm_history = lstm_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

    transformer_model = TransformerThreatDetector(
        input_dim=split.X_train.shape[1],
        num_classes=2,
        epochs=epochs,
        batch_size=batch_size,
    )
    transformer_history = transformer_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

    hybrid_model = DriftAwareHybridDetector(
        signature_model=signature_model,
        rf_model=random_forest,
        lstm_model=lstm_model,
        transformer_model=transformer_model,
    )
    hybrid_model.fit(split.X_train, split.val_df, split.X_val, split.y_val)

    models = {
        "Signature IDS": ("raw", signature_model),
        "Random Forest": ("scaled", random_forest),
        "LSTM": ("scaled", lstm_model),
        "Transformer": ("scaled", transformer_model),
        hybrid_variant_name: ("hybrid", hybrid_model),
    }

    metrics_by_model: dict[str, dict[str, float]] = {}
    probability_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    history_map = {
        "LSTM": lstm_history,
        "Transformer": transformer_history,
    }

    latency_sample_raw = split.test_df.iloc[: min(2048, len(split.test_df))].reset_index(drop=True)
    latency_sample_scaled = split.X_test[: len(latency_sample_raw)]

    for model_name, (mode, model) in models.items():
        if mode == "raw":
            y_prob = model.predict_proba(split.test_df)
            latency_ms = (_latency_wrapper(lambda: model.predict_proba(latency_sample_raw)) / len(latency_sample_raw)) * 1000.0
        elif mode == "hybrid":
            if hybrid_variant_mode == "static":
                y_prob = model.predict_proba_static(split.test_df, split.X_test)
                latency_ms = (
                    _latency_wrapper(lambda: model.predict_proba_static(latency_sample_raw, latency_sample_scaled)) /
                    len(latency_sample_raw)
                ) * 1000.0
            else:
                y_prob = model.predict_proba(split.test_df, split.X_test)
                latency_ms = (
                    _latency_wrapper(lambda: model.predict_proba(latency_sample_raw, latency_sample_scaled)) /
                    len(latency_sample_raw)
                ) * 1000.0
        else:
            y_prob = model.predict_proba(split.X_test)
            latency_ms = (_latency_wrapper(lambda: model.predict_proba(latency_sample_scaled)) / len(latency_sample_raw)) * 1000.0

        y_pred = (y_prob[:, 1] >= 0.5).astype(int)
        metrics = classification_metrics(split.y_test, y_pred, y_prob, CLASS_NAMES)
        metrics["latency_ms_per_flow"] = float(latency_ms)
        metrics_by_model[model_name] = metrics
        probability_map[model_name] = (split.y_test, y_prob)
        predictions_by_model[model_name] = y_pred

    results_df = write_results_table(
        metrics_by_model,
        results_dir / f"{prefix}_model_comparison.csv",
        results_dir / f"{prefix}_model_comparison.md",
    )
    best_model_name = str(results_df.iloc[0]["Model"])

    plot_confusion_matrix(
        split.y_test,
        predictions_by_model[best_model_name],
        CLASS_NAMES,
        results_dir / f"{prefix}_confusion_matrix.png",
        f"{split.name}: {best_model_name}",
    )
    plot_roc_curves(probability_map, CLASS_NAMES, results_dir / f"{prefix}_roc_curve.png")
    plot_feature_importance(
        random_forest.feature_importances(FEATURE_COLUMNS),
        results_dir / f"{prefix}_feature_importance.png",
    )
    plot_training_loss_curves(history_map, results_dir / f"{prefix}_training_loss_curves.png")

    signature_model.save(str(artifacts_dir / f"{prefix}_signature_ids.json"))
    random_forest.save(str(artifacts_dir / f"{prefix}_random_forest.joblib"))
    lstm_model.save(str(artifacts_dir / f"{prefix}_lstm.pt"))
    transformer_model.save(str(artifacts_dir / f"{prefix}_transformer.pt"))
    hybrid_model.save(str(artifacts_dir / f"{prefix}_hybrid.joblib"))
    joblib.dump(split.preprocessor, artifacts_dir / f"{prefix}_preprocessor.joblib")

    experiment = ExperimentArtifacts(
        split_name=split.name,
        results_df=results_df,
        metrics_by_model=metrics_by_model,
        best_model_name=best_model_name,
        hybrid_variant_name=hybrid_variant_name,
        hybrid_variant_mode=hybrid_variant_mode,
        signature_model=signature_model,
        random_forest=random_forest,
        lstm_model=lstm_model,
        transformer_model=transformer_model,
        hybrid_model=hybrid_model,
        split=split,
    )
    _save_metadata(experiment, artifacts_dir, prefix)
    return experiment


def run_latency_under_load(
    experiment: ExperimentArtifacts,
    results_dir: str | Path,
    prefix: str,
    batch_sizes: list[int] | None = None,
) -> None:
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 256, 1024]

    results_dir = Path(results_dir)
    rows = []
    model_names = ["Signature IDS", "Random Forest", "LSTM", "Transformer", experiment.hybrid_variant_name]

    for model_name in model_names:
        for batch_size in batch_sizes:
            raw_batch = experiment.split.test_df.iloc[:batch_size].reset_index(drop=True)
            scaled_batch = experiment.split.X_test[:batch_size]
            if model_name == "Signature IDS":
                elapsed = _latency_wrapper(lambda: experiment.signature_model.predict_proba(raw_batch), repeats=5)
            elif model_name == experiment.hybrid_variant_name:
                if experiment.hybrid_variant_mode == "static":
                    elapsed = _latency_wrapper(
                        lambda: experiment.hybrid_model.predict_proba_static(raw_batch, scaled_batch),
                        repeats=5,
                    )
                else:
                    elapsed = _latency_wrapper(
                        lambda: experiment.hybrid_model.predict_proba(raw_batch, scaled_batch, batch_size=batch_size),
                        repeats=5,
                    )
            elif model_name == "Random Forest":
                elapsed = _latency_wrapper(lambda: experiment.random_forest.predict_proba(scaled_batch), repeats=5)
            elif model_name == "LSTM":
                elapsed = _latency_wrapper(lambda: experiment.lstm_model.predict_proba(scaled_batch), repeats=5)
            else:
                elapsed = _latency_wrapper(lambda: experiment.transformer_model.predict_proba(scaled_batch), repeats=5)

            ms_per_flow = (elapsed / batch_size) * 1000.0
            throughput = batch_size / max(elapsed, 1e-8)
            rows.append(
                {
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Latency (ms/flow)": round(ms_per_flow, 4),
                    "Throughput (flows/s)": round(throughput, 2),
                }
            )

    latency_df = pd.DataFrame(rows)
    latency_df.to_csv(results_dir / f"{prefix}_latency_under_load.csv", index=False)

    markdown_lines = [
        f"# {experiment.split_name} Latency Under Load\n\n",
        "| Model | Batch Size | Latency (ms/flow) | Throughput (flows/s) |\n",
        "| --- | --- | --- | --- |\n",
    ]
    for _, row in latency_df.iterrows():
        markdown_lines.append(
            f"| {row['Model']} | {row['Batch Size']} | {row['Latency (ms/flow)']:.4f} | {row['Throughput (flows/s)']:.2f} |\n"
        )
    (results_dir / f"{prefix}_latency_under_load.md").write_text("".join(markdown_lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, group in latency_df.groupby("Model"):
        axes[0].plot(group["Batch Size"], group["Latency (ms/flow)"], marker="o", label=model_name)
        axes[1].plot(group["Batch Size"], group["Throughput (flows/s)"], marker="o", label=model_name)
    axes[0].set_title("Latency Under Load")
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("ms per flow")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Throughput Under Load")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("flows per second")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_latency_under_load.png", dpi=180)
    plt.close()


def run_explainability_ablation(
    experiment: ExperimentArtifacts,
    results_dir: str | Path,
    prefix: str,
    top_k: int = 5,
    sample_size: int = 2000,
) -> None:
    results_dir = Path(results_dir)
    test_df = experiment.split.test_df.reset_index(drop=True)
    X_test = experiment.split.X_test
    y_test = experiment.split.y_test
    benign_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    if len(benign_idx) == 0 or len(attack_idx) == 0:
        return

    rf_importance = experiment.random_forest.feature_importances(FEATURE_COLUMNS)
    top_features = [name for name, _ in rf_importance[:top_k]]
    rng = np.random.default_rng(42)
    remaining_features = [name for name in FEATURE_COLUMNS if name not in top_features]
    random_features = rng.choice(remaining_features, size=min(top_k, len(remaining_features)), replace=False).tolist()
    benign_reference = (
        experiment.split.train_df.loc[experiment.split.train_df["label"] == 0, FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .median()
    )

    benign_take = min(len(benign_idx), max(1, sample_size // 2))
    attack_take = min(len(attack_idx), max(1, sample_size - benign_take))
    sample_idx = np.concatenate(
        [
            rng.choice(benign_idx, size=benign_take, replace=False),
            rng.choice(attack_idx, size=attack_take, replace=False),
        ]
    )
    sample_idx.sort()

    baseline_pred = experiment.random_forest.predict(X_test[sample_idx])
    baseline_f1 = float(
        f1_score(y_test[sample_idx], baseline_pred, average="weighted", zero_division=0)
    )
    sample_raw = test_df.iloc[sample_idx][FEATURE_COLUMNS].copy()
    ablated_top_raw = sample_raw.copy()
    ablated_random_raw = sample_raw.copy()
    for feature_name in top_features:
        ablated_top_raw[feature_name] = benign_reference[feature_name]
    for feature_name in random_features:
        ablated_random_raw[feature_name] = benign_reference[feature_name]

    top_pred = experiment.random_forest.predict(_transform_frame(experiment, ablated_top_raw))
    random_pred = experiment.random_forest.predict(_transform_frame(experiment, ablated_random_raw))
    top_drop = baseline_f1 - float(
        f1_score(y_test[sample_idx], top_pred, average="weighted", zero_division=0)
    )
    random_drop = baseline_f1 - float(
        f1_score(y_test[sample_idx], random_pred, average="weighted", zero_division=0)
    )

    rows = pd.DataFrame(
        [
            {"Ablation": "Top feature importance", "Weighted F1 drop": round(top_drop, 4)},
            {"Ablation": "Random features", "Weighted F1 drop": round(random_drop, 4)},
        ]
    )
    rows.to_csv(results_dir / f"{prefix}_explainability_ablation.csv", index=False)

    lines = [
        f"# {experiment.split_name} Explainability Ablation\n\n",
        "Top RF importance features were replaced with benign-reference values and compared against random feature ablation.\n\n",
        f"Top features: {', '.join(top_features)}\n\n",
        f"Random features: {', '.join(random_features)}\n\n",
        f"Baseline weighted F1 on the sampled evaluation set: {baseline_f1:.4f}\n\n",
        "| Ablation | Weighted F1 drop |\n",
        "| --- | --- |\n",
    ]
    for _, row in rows.iterrows():
        lines.append(f"| {row['Ablation']} | {row['Weighted F1 drop']:.4f} |\n")
    (results_dir / f"{prefix}_explainability_ablation.md").write_text("".join(lines))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rows["Ablation"], rows["Weighted F1 drop"], color=["#0f766e", "#94a3b8"])
    ax.set_ylabel("Weighted F1 drop")
    ax.set_title("Explainability Validated by Ablation")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_explainability_ablation.png", dpi=180)
    plt.close()


def run_online_drift_adaptation(
    experiment: ExperimentArtifacts,
    results_dir: str | Path,
    prefix: str,
    batch_size: int = 512,
) -> dict[str, dict[str, float]]:
    results_dir = Path(results_dir)
    static_prob = experiment.hybrid_model.predict_proba_static(
        experiment.split.test_df,
        experiment.split.X_test,
    )
    adaptive_prob = experiment.hybrid_model.predict_proba(
        experiment.split.test_df,
        experiment.split.X_test,
        batch_size=batch_size,
    )
    trace = experiment.hybrid_model.last_adaptation_trace.copy()

    static_pred = (static_prob[:, 1] >= 0.5).astype(int)
    adaptive_pred = (adaptive_prob[:, 1] >= 0.5).astype(int)

    metrics = {
        "Static Hybrid": classification_metrics(
            experiment.split.y_test,
            static_pred,
            static_prob,
            CLASS_NAMES,
        ),
        "Online Drift-Adaptive Hybrid": classification_metrics(
            experiment.split.y_test,
            adaptive_pred,
            adaptive_prob,
            CLASS_NAMES,
        ),
    }
    metrics["Static Hybrid"]["latency_ms_per_flow"] = float("nan")
    metrics["Online Drift-Adaptive Hybrid"]["latency_ms_per_flow"] = float("nan")

    rows = pd.DataFrame(
        [
            {
                "Variant": name,
                "Accuracy": round(values["accuracy"] * 100, 2),
                "Precision": round(values["precision"] * 100, 2),
                "Recall": round(values["recall"] * 100, 2),
                "F1 Score": round(values["f1_score"] * 100, 2),
                "ROC AUC": round(values["roc_auc"], 4),
            }
            for name, values in metrics.items()
        ]
    )
    rows.to_csv(results_dir / f"{prefix}_online_drift_adaptation.csv", index=False)

    md_lines = [
        f"# {experiment.split_name} Online Drift Adaptation\n\n",
        f"Batch size for online adaptation: `{batch_size}`\n\n",
        "| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
    ]
    for _, row in rows.iterrows():
        md_lines.append(
            f"| {row['Variant']} | {row['Accuracy']:.2f}% | {row['Precision']:.2f}% | "
            f"{row['Recall']:.2f}% | {row['F1 Score']:.2f}% | {row['ROC AUC']:.4f} |\n"
        )
    if not trace.empty:
        md_lines.extend(
            [
                "\nAdaptive batches summarize mean drift score, adaptation alpha, and the online ensemble weights.\n",
                f"\nAverage adaptation alpha: `{trace['adaptation_alpha'].mean():.4f}`\n",
                f"\nPeak adaptation alpha: `{trace['adaptation_alpha'].max():.4f}`\n",
            ]
        )
    (results_dir / f"{prefix}_online_drift_adaptation.md").write_text("".join(md_lines))

    if not trace.empty:
        trace.to_csv(results_dir / f"{prefix}_online_drift_trace.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(trace.index, trace["ema_drift_score"], label="EMA drift", color="#b45309", linewidth=2)
        axes[0].plot(trace.index, trace["adaptation_alpha"], label="Adaptation alpha", color="#1d4ed8", linewidth=2)
        axes[0].set_title("Online Drift State")
        axes[0].set_xlabel("Batch")
        axes[0].set_ylabel("Score")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(trace.index, trace["random_forest_weight"], label="RF", color="#0f766e", linewidth=2)
        axes[1].plot(trace.index, trace["lstm_weight"], label="LSTM", color="#1d4ed8", linewidth=2)
        axes[1].plot(trace.index, trace["transformer_weight"], label="Transformer", color="#b45309", linewidth=2)
        axes[1].plot(trace.index, trace["stacked_meta_weight"], label="Meta", color="#7c3aed", linewidth=2)
        axes[1].set_title("Adaptive Ensemble Weights")
        axes[1].set_xlabel("Batch")
        axes[1].set_ylabel("Weight")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")
        plt.tight_layout()
        plt.savefig(results_dir / f"{prefix}_online_drift_adaptation.png", dpi=180)
        plt.close()

    return metrics


def write_experiment_manifest(summary: dict, save_path: str | Path) -> None:
    Path(save_path).write_text(json.dumps(summary, indent=2))
