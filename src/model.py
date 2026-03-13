"""
model.py
--------
ML model definitions for network intrusion detection.
Provides RandomForest and XGBoost classifiers with sensible defaults
and a unified ModelFactory for instantiation.
"""

from __future__ import annotations

import os
import joblib
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ModelType = Literal['random_forest', 'xgboost']


@dataclass
class ModelConfig:
    """Hyperparameter configuration for both estimators."""
    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    rf_min_samples_split: int = 5
    rf_class_weight: str = 'balanced'
    rf_n_jobs: int = -1

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    xgb_eval_metric: str = 'logloss'
    xgb_n_jobs: int = -1
    xgb_random_state: int = 42

    random_state: int = 42


class IntrusionDetectionModel:
    """
    Wrapper around a scikit-learn-compatible classifier that adds:
      - unified fit / predict / evaluate interface
      - artifact save / load helpers
    """

    SUPPORTED = ('random_forest', 'xgboost')

    def __init__(self, model_type: ModelType, config: ModelConfig | None = None):
        if model_type not in self.SUPPORTED:
            raise ValueError(f"model_type must be one of {self.SUPPORTED}, got '{model_type}'")
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.model = self._build_model()
        self.is_trained = False
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_model(self):
        c = self.config
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=c.rf_n_estimators,
                max_depth=c.rf_max_depth,
                min_samples_split=c.rf_min_samples_split,
                class_weight=c.rf_class_weight,
                n_jobs=c.rf_n_jobs,
                random_state=c.random_state,
            )
        return XGBClassifier(
            n_estimators=c.xgb_n_estimators,
            max_depth=c.xgb_max_depth,
            learning_rate=c.xgb_learning_rate,
            subsample=c.xgb_subsample,
            colsample_bytree=c.xgb_colsample_bytree,
            eval_metric=c.xgb_eval_metric,
            n_jobs=c.xgb_n_jobs,
            random_state=c.xgb_random_state,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str] | None = None) -> None:
        logger.info(f"Training {self.model_type} on {X_train.shape[0]:,} samples …")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        if feature_names:
            self.feature_names = feature_names
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Return a dict of standard classification metrics."""
        self._check_trained()
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        metrics = {
            'accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
        }
        logger.info(f"\n{'='*40}\nModel: {self.model_type.upper()}\n{classification_report(y_test, y_pred, target_names=['BENIGN','ATTACK'])}")
        return metrics

    def feature_importance(self) -> dict[str, float] | None:
        """Return feature importances if available."""
        self._check_trained()
        if not hasattr(self.model, 'feature_importances_'):
            return None
        importances = self.model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importances):
            return dict(sorted(zip(self.feature_names, importances),
                               key=lambda x: x[1], reverse=True))
        return {'feature_' + str(i): v for i, v in enumerate(importances)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'IntrusionDetectionModel':
        logger.info(f"Loading model from {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an IntrusionDetectionModel: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")

    def __repr__(self):
        return f"IntrusionDetectionModel(type={self.model_type}, trained={self.is_trained})"


class ModelFactory:
    """Convenience factory to create models by name."""

    @staticmethod
    def create(model_type: ModelType, config: ModelConfig | None = None) -> IntrusionDetectionModel:
        return IntrusionDetectionModel(model_type, config)

    @staticmethod
    def create_all(config: ModelConfig | None = None) -> dict[str, IntrusionDetectionModel]:
        return {mt: IntrusionDetectionModel(mt, config) for mt in IntrusionDetectionModel.SUPPORTED}
