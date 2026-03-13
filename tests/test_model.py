"""
test_model.py
-------------
Unit and integration tests for the AI Network Intrusion Detection System.
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.model import IntrusionDetectionModel, ModelConfig, ModelFactory


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

N_FEATURES = 20
N_TRAIN = 300
N_TEST = 80

RNG = np.random.default_rng(0)


def _make_benign_traffic() -> np.ndarray:
    """Return a feature vector that resembles benign (low-intensity) traffic."""
    return RNG.normal(loc=0.1, scale=0.05, size=(1, N_FEATURES))


def _make_attack_traffic(kind: str = 'port_scan') -> np.ndarray:
    """Return a feature vector that mimics attack traffic patterns."""
    if kind == 'port_scan':
        # High packet rate, many unique ports, short flows → distinctive signature
        vec = RNG.normal(loc=5.0, scale=1.0, size=(1, N_FEATURES))
    elif kind == 'ddos':
        vec = RNG.normal(loc=8.0, scale=2.0, size=(1, N_FEATURES))
    else:
        vec = RNG.normal(loc=3.0, scale=0.5, size=(1, N_FEATURES))
    return vec


def _synthetic_dataset(n_benign: int = 150, n_attack: int = 150):
    X_benign = RNG.normal(loc=0.0, scale=1.0, size=(n_benign, N_FEATURES))
    X_attack = RNG.normal(loc=4.0, scale=1.0, size=(n_attack, N_FEATURES))
    X = np.vstack([X_benign, X_attack])
    y = np.array([0] * n_benign + [1] * n_attack)
    idx = RNG.permutation(len(y))
    return X[idx], y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

class TestModelInstantiation(unittest.TestCase):
    """Verify that models can be created with default and custom configs."""

    def test_random_forest_creation(self):
        model = ModelFactory.create('random_forest')
        self.assertIsInstance(model, IntrusionDetectionModel)
        self.assertEqual(model.model_type, 'random_forest')
        self.assertFalse(model.is_trained)

    def test_xgboost_creation(self):
        model = ModelFactory.create('xgboost')
        self.assertIsInstance(model, IntrusionDetectionModel)
        self.assertEqual(model.model_type, 'xgboost')

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            IntrusionDetectionModel('neural_net')

    def test_create_all_returns_both_models(self):
        models = ModelFactory.create_all()
        self.assertIn('random_forest', models)
        self.assertIn('xgboost', models)
        self.assertEqual(len(models), 2)

    def test_custom_config(self):
        cfg = ModelConfig(rf_n_estimators=50, xgb_n_estimators=50)
        model = ModelFactory.create('random_forest', cfg)
        self.assertEqual(model.model.n_estimators, 50)


class TestModelTraining(unittest.TestCase):
    """Ensure training completes and updates internal state."""

    def setUp(self):
        self.X_train, self.y_train = _synthetic_dataset()
        self.X_test, self.y_test = _synthetic_dataset(50, 50)

    def test_random_forest_trains(self):
        model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=10))
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)

    def test_xgboost_trains(self):
        model = ModelFactory.create('xgboost', ModelConfig(xgb_n_estimators=10))
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)

    def test_predict_before_fit_raises(self):
        model = ModelFactory.create('random_forest')
        with self.assertRaises(RuntimeError):
            model.predict(self.X_test)

    def test_predict_returns_correct_shape(self):
        model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=10))
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_predict_proba_in_range(self):
        model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=10))
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))


class TestModelEvaluation(unittest.TestCase):
    """Verify that evaluation metrics are sane and within expected bounds."""

    @classmethod
    def setUpClass(cls):
        X_train, y_train = _synthetic_dataset(200, 200)
        cls.X_test, cls.y_test = _synthetic_dataset(80, 80)
        cls.model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=30))
        cls.model.fit(X_train, y_train)
        cls.metrics = cls.model.evaluate(cls.X_test, cls.y_test)

    def test_accuracy_key_present(self):
        self.assertIn('accuracy', self.metrics)

    def test_accuracy_above_threshold(self):
        # Synthetic data is well-separated; expect high accuracy
        self.assertGreater(self.metrics['accuracy'], 0.85)

    def test_precision_in_range(self):
        self.assertGreaterEqual(self.metrics['precision'], 0.0)
        self.assertLessEqual(self.metrics['precision'], 1.0)

    def test_recall_in_range(self):
        self.assertGreaterEqual(self.metrics['recall'], 0.0)
        self.assertLessEqual(self.metrics['recall'], 1.0)

    def test_roc_auc_in_range(self):
        self.assertGreater(self.metrics['roc_auc'], 0.5)
        self.assertLessEqual(self.metrics['roc_auc'], 1.0)


class TestTrafficClassification(unittest.TestCase):
    """
    Black-box tests: does the model correctly classify synthetic
    benign vs attack flows after training on distinguishable data?

    Test Case 1 — Benign Traffic
    Input : Feature vector drawn from low-mean Gaussian (benign pattern)
    Expected : BENIGN (label = 0)

    Test Case 2 — Port Scan Traffic
    Input : Feature vector drawn from high-mean Gaussian (attack pattern)
    Expected : ATTACK (label = 1)
    """

    @classmethod
    def setUpClass(cls):
        X_train, y_train = _synthetic_dataset(400, 400)
        cls.model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=50))
        cls.model.fit(X_train, y_train)

    def test_case_1_benign_traffic(self):
        """Test Case 1: benign traffic → expected label = BENIGN (0)"""
        X = _make_benign_traffic()
        pred = self.model.predict(X)[0]
        label = 'BENIGN' if pred == 0 else 'ATTACK'
        print(f"\n  TC-1  Input=benign_traffic  Predicted={label}  Expected=BENIGN")
        self.assertEqual(pred, 0, f"Expected BENIGN (0) but got {pred}")

    def test_case_2_port_scan_traffic(self):
        """Test Case 2: port scan traffic → expected label = ATTACK (1)"""
        X = _make_attack_traffic('port_scan')
        pred = self.model.predict(X)[0]
        label = 'BENIGN' if pred == 0 else 'ATTACK'
        print(f"\n  TC-2  Input=port_scan_traffic  Predicted={label}  Expected=ATTACK")
        self.assertEqual(pred, 1, f"Expected ATTACK (1) but got {pred}")

    def test_case_3_ddos_traffic(self):
        """Test Case 3: DDoS traffic → expected label = ATTACK (1)"""
        X = _make_attack_traffic('ddos')
        pred = self.model.predict(X)[0]
        label = 'BENIGN' if pred == 0 else 'ATTACK'
        print(f"\n  TC-3  Input=ddos_traffic  Predicted={label}  Expected=ATTACK")
        self.assertEqual(pred, 1, f"Expected ATTACK (1) but got {pred}")


class TestFeatureImportance(unittest.TestCase):
    """Verify feature importance is available and well-formed."""

    def test_feature_importance_returns_dict(self):
        X, y = _synthetic_dataset()
        model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=10))
        feature_names = [f'feat_{i}' for i in range(N_FEATURES)]
        model.fit(X, y, feature_names=feature_names)
        fi = model.feature_importance()
        self.assertIsNotNone(fi)
        self.assertEqual(len(fi), N_FEATURES)

    def test_importances_sum_to_one(self):
        X, y = _synthetic_dataset()
        model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=10))
        model.fit(X, y)
        fi = model.feature_importance()
        total = sum(fi.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestModelPersistence(unittest.TestCase):
    """Verify save/load round-trip preserves predictions."""

    def test_save_and_reload(self):
        import tempfile, shutil
        X_train, y_train = _synthetic_dataset()
        X_test, _ = _synthetic_dataset(50, 50)

        model = ModelFactory.create('random_forest', ModelConfig(rf_n_estimators=10))
        model.fit(X_train, y_train)
        preds_before = model.predict(X_test)

        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, 'rf.joblib')
            model.save(path)
            loaded = IntrusionDetectionModel.load(path)
            preds_after = loaded.predict(X_test)
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
