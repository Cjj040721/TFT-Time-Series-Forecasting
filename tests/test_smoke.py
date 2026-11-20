import unittest

import torch
from fastapi.testclient import TestClient

from app.api.main import app
from src.model.factory import build_model


class TFTForwardTests(unittest.TestCase):
    def test_forward_attention_and_variable_weights(self):
        model = build_model(
            model_name="tft",
            n_features=7,
            n_time_features=4,
            seq_len=96,
            label_len=48,
            pred_len=24,
            d_model=16,
            n_heads=4,
            lstm_layers=1,
            dropout=0.0,
        ).eval()

        with torch.no_grad():
            prediction = model(
                torch.randn(1, 96, 7),
                torch.randn(1, 96, 4),
                torch.randn(1, 72, 7),
                torch.randn(1, 72, 4),
            )

        self.assertEqual(tuple(prediction.shape), (1, 24))
        self.assertEqual(tuple(model.last_attention.shape), (1, 24, 96))
        self.assertEqual(tuple(model.last_var_weights.shape), (1, 96, 11))
        totals = model.last_var_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_and_models(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})

        models = self.client.get("/models")
        self.assertEqual(models.status_code, 200)
        self.assertGreater(len(models.json()["models"]), 0)

    def test_preserved_metrics_and_attention(self):
        run_id = "ETTh1/tft/seq96_pred24_run1"
        metrics = self.client.get(f"/results/{run_id}/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json()["run_id"], run_id)

        attention = self.client.get(f"/results/{run_id}/attention")
        self.assertEqual(attention.status_code, 200)
        self.assertEqual(attention.json()["run_id"], run_id)
        self.assertGreater(len(attention.json()["attention"]), 0)


if __name__ == "__main__":
    unittest.main()
