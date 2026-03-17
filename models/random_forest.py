from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestThreatDetector:
    model_name = "Random Forest"

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self, feature_names: list[str]) -> list[tuple[str, float]]:
        pairs = list(zip(feature_names, self.model.feature_importances_.tolist()))
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestThreatDetector":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
