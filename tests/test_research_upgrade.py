import unittest

import numpy as np
import pandas as pd

from models.drift_aware_hybrid import DriftAwareHybridDetector
from models.signature_ids import SignatureIDSBaseline
from training.canonical_pipeline import _to_canonical
from src.preprocessing import FEATURE_COLUMNS


class _DummyProbModel:
    def predict_proba(self, X):
        score = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - score, score])


class _DummySignature:
    def predict_proba(self, raw_df):
        score = np.clip(raw_df["flow_packets_s"].to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


class TestSignatureIDSBaseline(unittest.TestCase):
    def test_signature_ids_detects_attack_like_flow(self):
        benign = pd.DataFrame(
            {
                "flow_packets_s": [1.0, 1.5, 0.8, 1.2],
                "flow_bytes_s": [5.0, 7.0, 4.0, 6.0],
                "syn_flag_cnt": [0.0, 0.0, 0.0, 0.0],
                "rst_flag_cnt": [0.0, 0.0, 0.0, 0.0],
                "urg_flag_cnt": [0.0, 0.0, 0.0, 0.0],
                "ack_flag_cnt": [3.0, 2.0, 3.0, 4.0],
                "flow_duration": [20.0, 25.0, 18.0, 22.0],
                "total_fwd_packets": [3.0, 4.0, 2.0, 3.0],
                "total_bwd_packets": [3.0, 4.0, 3.0, 2.0],
                "min_packet_len": [20.0, 18.0, 22.0, 19.0],
                "packet_len_mean": [70.0, 75.0, 68.0, 72.0],
                "label": [0, 0, 0, 0],
            }
        )
        attack = pd.DataFrame(
            {
                "flow_packets_s": [500.0],
                "flow_bytes_s": [3000.0],
                "syn_flag_cnt": [15.0],
                "rst_flag_cnt": [4.0],
                "urg_flag_cnt": [0.0],
                "ack_flag_cnt": [0.0],
                "flow_duration": [0.2],
                "total_fwd_packets": [120.0],
                "total_bwd_packets": [0.0],
                "min_packet_len": [10.0],
                "packet_len_mean": [600.0],
                "label": [1],
            }
        )

        model = SignatureIDSBaseline()
        model.fit(benign)

        prediction = model.predict(attack)[0]
        confidence = model.predict_proba(attack)[0, 1]
        explanation = model.explain(attack.iloc[0])

        self.assertEqual(prediction, 1)
        self.assertGreater(confidence, 0.5)
        self.assertNotEqual(explanation, ["no_signature_triggered"])


class TestCanonicalPipeline(unittest.TestCase):
    def test_to_canonical_replaces_inf_and_missing_values(self):
        frame = pd.DataFrame(
            {
                "flow_packets_s": [np.inf, 12.0],
                "flow_bytes_s": [np.nan, 40.0],
                "flow_duration": [1.0, -np.inf],
            }
        )
        labels = pd.Series([0, 1])
        canonical = _to_canonical(frame, labels, "synthetic")

        self.assertEqual(set(["label", "_source"]).issubset(canonical.columns), True)
        values = canonical[FEATURE_COLUMNS].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(canonical["label"].tolist(), [0, 1])


class TestDriftAwareHybrid(unittest.TestCase):
    def test_hybrid_detector_fits_and_predicts(self):
        rng = np.random.default_rng(42)
        train_scaled = rng.normal(size=(64, 4)).astype(np.float32)
        val_scaled = rng.normal(size=(24, 4)).astype(np.float32)
        y_val = (val_scaled[:, 0] + val_scaled[:, 1] > 0.0).astype(int)
        raw_df = pd.DataFrame(
            {
                "flow_packets_s": np.clip(val_scaled[:, 0] * 6.0 + 6.0, 0.0, None),
            }
        )

        model = DriftAwareHybridDetector(
            signature_model=_DummySignature(),
            rf_model=_DummyProbModel(),
            lstm_model=_DummyProbModel(),
            transformer_model=_DummyProbModel(),
        )
        model.fit(train_scaled, raw_df, val_scaled, y_val)
        probabilities = model.predict_proba(raw_df.iloc[:8], val_scaled[:8])

        self.assertEqual(probabilities.shape, (8, 2))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(8), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
