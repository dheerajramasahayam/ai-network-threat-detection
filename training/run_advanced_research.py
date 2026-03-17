from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.research_suite import (
    run_explainability_ablation,
    run_latency_under_load,
    run_model_family_experiment,
    write_experiment_manifest,
)
from training.canonical_pipeline import (
    dataset_summary,
    prepare_joint_unsw_nsl_to_cicids,
    prepare_official_nsl_split,
    prepare_official_unsw_split,
)

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"


def _advanced_metadata_path() -> Path:
    return ARTIFACTS_DIR / "advanced_metadata.json"


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    unsw_split = prepare_official_unsw_split(args.unsw_dir)
    unsw_experiment = run_model_family_experiment(
        split=unsw_split,
        results_dir=RESULTS_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        prefix="official_unsw",
        epochs=args.epochs,
        batch_size=args.batch_size,
        rf_trees=args.rf_trees,
    )
    run_latency_under_load(unsw_experiment, RESULTS_DIR, prefix="official_unsw")
    run_explainability_ablation(unsw_experiment, RESULTS_DIR, prefix="official_unsw")

    nsl_split = prepare_official_nsl_split(args.nsl_dir)
    nsl_experiment = run_model_family_experiment(
        split=nsl_split,
        results_dir=RESULTS_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        prefix="official_nsl_kdd",
        epochs=args.epochs,
        batch_size=args.batch_size,
        rf_trees=args.rf_trees,
    )
    run_latency_under_load(nsl_experiment, RESULTS_DIR, prefix="official_nsl_kdd")

    transfer_split = prepare_joint_unsw_nsl_to_cicids(
        unsw_dir=args.unsw_dir,
        nsl_dir=args.nsl_dir,
        cicids_path=args.cicids_path,
        cicids_sample_size=args.cicids_sample_size,
    )
    transfer_experiment = run_model_family_experiment(
        split=transfer_split,
        results_dir=RESULTS_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        prefix="transfer_unsw_nsl_to_cicids",
        epochs=args.epochs,
        batch_size=args.batch_size,
        rf_trees=args.rf_trees,
    )
    run_latency_under_load(transfer_experiment, RESULTS_DIR, prefix="transfer_unsw_nsl_to_cicids")

    advanced_metadata = {
        "best_deployment_split": "official_unsw",
        "best_model": unsw_experiment.best_model_name,
        "feature_names": unsw_split.feature_names,
        "class_names": ["Benign", "Attack"],
        "artifacts": {
            "official_unsw_signature_ids": "models/artifacts/official_unsw_signature_ids.json",
            "official_unsw_random_forest": "models/artifacts/official_unsw_random_forest.joblib",
            "official_unsw_lstm": "models/artifacts/official_unsw_lstm.pt",
            "official_unsw_transformer": "models/artifacts/official_unsw_transformer.pt",
            "official_unsw_hybrid": "models/artifacts/official_unsw_hybrid.joblib",
            "official_unsw_preprocessor": "models/artifacts/official_unsw_preprocessor.joblib",
        },
    }
    _advanced_metadata_path().write_text(json.dumps(advanced_metadata, indent=2))

    manifest = {
        "primary_official_split": dataset_summary(unsw_split),
        "secondary_official_split": dataset_summary(nsl_split),
        "external_transfer_split": dataset_summary(transfer_split),
        "external_cicids_rows": int(len(transfer_split.test_df)),
        "results": {
            "official_unsw": unsw_experiment.metrics_by_model,
            "official_nsl_kdd": nsl_experiment.metrics_by_model,
            "transfer_unsw_nsl_to_cicids": transfer_experiment.metrics_by_model,
        },
        "best_models": {
            "official_unsw": unsw_experiment.best_model_name,
            "official_nsl_kdd": nsl_experiment.best_model_name,
            "transfer_unsw_nsl_to_cicids": transfer_experiment.best_model_name,
        },
    }
    write_experiment_manifest(manifest, RESULTS_DIR / "advanced_experiment_summary.json")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the upgraded novelty-oriented NIDS research suite.")
    parser.add_argument(
        "--unsw-dir",
        default=str(BASE_DIR / "dataset" / "raw" / "unsw_nb15"),
        help="Directory containing official UNSW-NB15 train/test CSV files.",
    )
    parser.add_argument(
        "--nsl-dir",
        default=str(BASE_DIR / "dataset" / "raw" / "nsl_kdd"),
        help="Directory containing official NSL-KDD train/test text files.",
    )
    parser.add_argument(
        "--cicids-path",
        default=str(BASE_DIR / "dataset" / "cicids2017.csv"),
        help="Path to the merged CICIDS2017 CSV file.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument(
        "--cicids-sample-size",
        type=int,
        default=200000,
        help="Rows to use from CICIDS2017 for external transfer evaluation. Use 0 for full data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cicids_sample_size == 0:
        args.cicids_sample_size = None
    run(args)
