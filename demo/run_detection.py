from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.lstm_model import LSTMThreatDetector
from models.random_forest import RandomForestThreatDetector
from models.transformer_model import TransformerThreatDetector
from training.data_pipeline import build_feature_table, engineer_unsw_features, load_unsw_nb15

ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"


def load_best_model():
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    best_model = metadata["best_model"]

    if best_model == "Random Forest":
        model = RandomForestThreatDetector.load(str(ARTIFACTS_DIR / "random_forest.joblib"))
    elif best_model == "LSTM":
        model = LSTMThreatDetector.load(str(ARTIFACTS_DIR / "lstm_model.pt"))
    else:
        model = TransformerThreatDetector.load(str(ARTIFACTS_DIR / "transformer_model.pt"))
    return best_model, model


def choose_sample(dataset_dir: str, sample_index: int, attack_only: bool):
    _, test_df = load_unsw_nb15(dataset_dir)
    test_df = engineer_unsw_features(test_df)
    if attack_only:
        test_df = test_df[test_df["attack_cat"].ne("Normal")].reset_index(drop=True)
    row = test_df.iloc[sample_index].copy()
    features_df, _ = build_feature_table(pd.DataFrame([row]))
    return row, features_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo threat classification on a sample UNSW-NB15 flow.")
    parser.add_argument(
        "--dataset-dir",
        default=str(BASE_DIR / "dataset" / "raw" / "unsw_nb15"),
        help="Directory containing the UNSW-NB15 split files.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--include-normal", action="store_true", help="Allow benign samples in the demo selection.")
    args = parser.parse_args()

    best_model_name, model = load_best_model()
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")

    row, features_df = choose_sample(args.dataset_dir, args.sample_index, not args.include_normal)
    X = preprocessor.transform(features_df).astype(np.float32)
    probabilities = model.predict_proba(X)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction]) * 100.0
    predicted_label = str(label_encoder.inverse_transform([prediction])[0])
    true_label = str(row["attack_cat"])

    print(f"Threat detected: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Ground truth: {true_label}")
    print(f"Model: {best_model_name}")


if __name__ == "__main__":
    main()
