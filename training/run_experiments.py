from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.reporting import (
    classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    plot_training_loss_curves,
    write_results_table,
    write_summary_json,
)
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.transformer_model import TransformerThreatDetector
from training.data_pipeline import prepare_research_dataset, summarize_unsw_nb15

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"


def benchmark_latency(model, X: np.ndarray, repeats: int = 3) -> float:
    sample = X[: min(2048, len(X))]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.predict_proba(sample)
        elapsed = time.perf_counter() - start
        timings.append((elapsed / len(sample)) * 1000.0)
    return float(np.mean(timings))


def _save_metadata(dataset, dataset_summary: dict, results_df) -> None:
    joblib.dump(dataset.preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(dataset.label_encoder, ARTIFACTS_DIR / "label_encoder.joblib")

    metadata = {
        "dataset": dataset_summary,
        "feature_names": dataset.feature_names,
        "class_names": dataset.class_names,
        "artifacts": {
            "random_forest": str((ARTIFACTS_DIR / "random_forest.joblib").relative_to(BASE_DIR)),
            "lstm": str((ARTIFACTS_DIR / "lstm_model.pt").relative_to(BASE_DIR)),
            "transformer": str((ARTIFACTS_DIR / "transformer_model.pt").relative_to(BASE_DIR)),
        },
        "best_model": results_df.iloc[0]["Model"],
    }
    write_summary_json(metadata, ARTIFACTS_DIR / "metadata.json")


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = prepare_research_dataset(
        dataset_dir=args.dataset_dir,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        validation_size=args.validation_size,
        random_state=args.seed,
    )
    dataset_summary = summarize_unsw_nb15(args.dataset_dir)

    metrics_by_model: dict[str, dict[str, float]] = {}
    probability_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    history_map: dict[str, dict[str, list[float]]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}

    random_forest = RandomForestThreatDetector(n_estimators=args.rf_trees, random_state=args.seed)
    random_forest.fit(dataset.X_train, dataset.y_train)
    rf_prob = random_forest.predict_proba(dataset.X_test)
    rf_pred = rf_prob.argmax(axis=1)
    rf_metrics = classification_metrics(dataset.y_test, rf_pred, rf_prob, dataset.class_names)
    rf_metrics["latency_ms_per_flow"] = benchmark_latency(random_forest, dataset.X_test)
    metrics_by_model[random_forest.model_name] = rf_metrics
    probability_map[random_forest.model_name] = (dataset.y_test, rf_prob)
    predictions_by_model[random_forest.model_name] = rf_pred
    random_forest.save(str(ARTIFACTS_DIR / "random_forest.joblib"))
    plot_feature_importance(
        random_forest.feature_importances(dataset.feature_names),
        RESULTS_DIR / "feature_importance.png",
    )

    lstm_model = LSTMThreatDetector(
        input_dim=dataset.X_train.shape[1],
        num_classes=len(dataset.class_names),
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    history_map[lstm_model.model_name] = lstm_model.fit(
        dataset.X_train,
        dataset.y_train,
        dataset.X_val,
        dataset.y_val,
    )
    lstm_prob = lstm_model.predict_proba(dataset.X_test)
    lstm_pred = lstm_prob.argmax(axis=1)
    lstm_metrics = classification_metrics(dataset.y_test, lstm_pred, lstm_prob, dataset.class_names)
    lstm_metrics["latency_ms_per_flow"] = benchmark_latency(lstm_model, dataset.X_test)
    metrics_by_model[lstm_model.model_name] = lstm_metrics
    probability_map[lstm_model.model_name] = (dataset.y_test, lstm_prob)
    predictions_by_model[lstm_model.model_name] = lstm_pred
    lstm_model.save(str(ARTIFACTS_DIR / "lstm_model.pt"))

    transformer_model = TransformerThreatDetector(
        input_dim=dataset.X_train.shape[1],
        num_classes=len(dataset.class_names),
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    history_map[transformer_model.model_name] = transformer_model.fit(
        dataset.X_train,
        dataset.y_train,
        dataset.X_val,
        dataset.y_val,
    )
    transformer_prob = transformer_model.predict_proba(dataset.X_test)
    transformer_pred = transformer_prob.argmax(axis=1)
    transformer_metrics = classification_metrics(
        dataset.y_test,
        transformer_pred,
        transformer_prob,
        dataset.class_names,
    )
    transformer_metrics["latency_ms_per_flow"] = benchmark_latency(transformer_model, dataset.X_test)
    metrics_by_model[transformer_model.model_name] = transformer_metrics
    probability_map[transformer_model.model_name] = (dataset.y_test, transformer_prob)
    predictions_by_model[transformer_model.model_name] = transformer_pred
    transformer_model.save(str(ARTIFACTS_DIR / "transformer_model.pt"))

    results_df = write_results_table(
        metrics_by_model,
        RESULTS_DIR / "model_comparison.csv",
        RESULTS_DIR / "model_comparison.md",
    )

    best_model_name = str(results_df.iloc[0]["Model"])
    plot_confusion_matrix(
        dataset.y_test,
        predictions_by_model[best_model_name],
        dataset.class_names,
        RESULTS_DIR / "confusion_matrix.png",
        f"{best_model_name} Confusion Matrix",
    )
    plot_roc_curves(probability_map, dataset.class_names, RESULTS_DIR / "roc_curve.png")
    plot_training_loss_curves(history_map, RESULTS_DIR / "training_loss_curves.png")

    experiment_summary = {
        "dataset_summary": dataset_summary,
        "sampled_train_rows": int(dataset.X_train.shape[0] + dataset.X_val.shape[0]),
        "sampled_test_rows": int(dataset.X_test.shape[0]),
        "best_model": best_model_name,
        "results": metrics_by_model,
    }
    write_summary_json(experiment_summary, RESULTS_DIR / "experiment_summary.json")
    _save_metadata(dataset, dataset_summary, results_df)

    print(json.dumps(experiment_summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research-paper experiments on UNSW-NB15.")
    parser.add_argument(
        "--dataset-dir",
        default=str(BASE_DIR / "dataset" / "raw" / "unsw_nb15"),
        help="Directory containing UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv",
    )
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-test-rows", type=int, default=25000)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
