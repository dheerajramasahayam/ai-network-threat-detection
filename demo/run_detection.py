from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.signature_ids import SignatureIDSBaseline
from models.transformer_model import TransformerThreatDetector
from training.canonical_pipeline import FEATURE_COLUMNS, load_official_unsw_frames

ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"


def _metadata_path() -> Path:
    advanced = ARTIFACTS_DIR / "advanced_metadata.json"
    if advanced.exists():
        return advanced
    return ARTIFACTS_DIR / "metadata.json"


def _load_official_unsw_sample(dataset_dir: str, sample_index: int, attack_only: bool):
    _, test_df = load_official_unsw_frames(dataset_dir)
    if attack_only:
        test_df = test_df[test_df["label"] == 1].reset_index(drop=True)
    row = test_df.iloc[sample_index].copy()
    return row, test_df.iloc[[sample_index]].reset_index(drop=True)


def load_best_model(metadata: dict):
    best_model = metadata["best_model"]
    artifacts = metadata["artifacts"]

    if "official_unsw_random_forest" in artifacts:
        signature = SignatureIDSBaseline.load(str(BASE_DIR / artifacts["official_unsw_signature_ids"]))
        rf = RandomForestThreatDetector.load(str(BASE_DIR / artifacts["official_unsw_random_forest"]))
        lstm = LSTMThreatDetector.load(str(BASE_DIR / artifacts["official_unsw_lstm"]))
        transformer = TransformerThreatDetector.load(str(BASE_DIR / artifacts["official_unsw_transformer"]))
        preprocessor = joblib.load(BASE_DIR / artifacts["official_unsw_preprocessor"])
        if best_model == "Signature IDS":
            return best_model, signature, preprocessor
        if best_model == "Random Forest":
            return best_model, rf, preprocessor
        if best_model == "LSTM":
            return best_model, lstm, preprocessor
        if best_model == "Transformer":
            return best_model, transformer, preprocessor
        hybrid = DriftAwareHybridDetector(signature, rf, lstm, transformer)
        hybrid.load_state(str(BASE_DIR / artifacts["official_unsw_hybrid"]))
        return best_model, hybrid, preprocessor

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    if best_model == "Random Forest":
        return best_model, RandomForestThreatDetector.load(str(ARTIFACTS_DIR / "random_forest.joblib")), preprocessor
    if best_model == "LSTM":
        return best_model, LSTMThreatDetector.load(str(ARTIFACTS_DIR / "lstm_model.pt")), preprocessor
    return best_model, TransformerThreatDetector.load(str(ARTIFACTS_DIR / "transformer_model.pt")), preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo threat classification on an official UNSW-NB15 sample.")
    parser.add_argument(
        "--dataset-dir",
        default=str(BASE_DIR / "dataset" / "raw" / "unsw_nb15"),
        help="Directory containing the official UNSW-NB15 train/test CSV files.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--include-normal", action="store_true", help="Allow benign samples in the demo selection.")
    args = parser.parse_args()

    metadata = json.loads(_metadata_path().read_text())
    best_model_name, model, preprocessor = load_best_model(metadata)
    row, row_df = _load_official_unsw_sample(args.dataset_dir, args.sample_index, not args.include_normal)
    feature_frame = row_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = preprocessor.transform(feature_frame.to_numpy(dtype=np.float64)).astype(np.float32)

    if best_model_name == "Signature IDS":
        probabilities = model.predict_proba(row_df)[0]
    elif best_model_name == "Drift-Aware Hybrid":
        probabilities = model.predict_proba(row_df, X)[0]
    else:
        probabilities = model.predict_proba(X)[0]

    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction]) * 100.0
    predicted_label = "Attack" if prediction == 1 else "Benign"
    true_label = "Attack" if int(row["label"]) == 1 else "Benign"

    print(f"Threat detected: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Ground truth: {true_label}")
    print(f"Model: {best_model_name}")
    if best_model_name == "Signature IDS":
        print(f"Rules fired: {', '.join(model.explain(row))}")


if __name__ == "__main__":
    main()
