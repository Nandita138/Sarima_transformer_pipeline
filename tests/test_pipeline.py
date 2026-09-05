import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import SyntheticFeederGenerator, ColumnMapper, DataConfig, GridGuardDataLoader
from src.models.sarima_model import SARIMAModel
from src.models.transformer_model import TransformerEncoder, TransformerAnomalyDetector
from src.explainability.shap_explainer import GridGuardExplainer
from src.evaluation import ModelEvaluator


class TestSyntheticGenerator:
    def test_shape(self):
        config = DataConfig(num_days=7, sample_freq="15min")
        gen = SyntheticFeederGenerator(config)
        df = gen.generate()
        assert len(df) == 7 * 24 * 4
        assert "active_power" in df.columns
        assert "voltage" in df.columns
        assert "current" in df.columns

    def test_anomalies(self):
        config = DataConfig(num_days=30, anomaly_ratio=0.03)
        gen = SyntheticFeederGenerator(config)
        df = gen.generate()
        assert 0.01 < df["anomaly_label"].mean() < 0.35


class TestColumnMapper:
    def test_datetime(self):
        config = DataConfig(datetime_aliases=["datetime", "Date"])
        mapper = ColumnMapper(config)
        df = pd.DataFrame({"Date": ["2023-01-01"], "Value": [1.0]})
        assert mapper.detect_datetime_column(df) == "Date"

    def test_target(self):
        config = DataConfig(target_aliases=["Global_active_power", "active_power"])
        mapper = ColumnMapper(config)
        df = pd.DataFrame({"Global_active_power": [1.0]})
        assert mapper.map_target_column(df) == "Global_active_power"


class TestSARIMA:
    def test_residual_extraction(self):
        series = pd.Series(np.sin(np.linspace(0, 50, 200)) + np.random.normal(0, 0.1, 200))
        sarima = SARIMAModel()
        sarima.fit(series, use_auto=False)
        residuals = sarima.get_residuals(series)
        assert len(residuals) == len(series)
        assert np.std(residuals) >= 0.0


class TestTransformer:
    def test_forward(self):
        import torch
        model = TransformerEncoder(input_dim=11, d_model=32, nhead=4, num_layers=2)
        x = torch.randn(2, 24, 11)
        out = model(x)
        assert out.shape == (2, 1)
        assert (out >= 0).all() and (out <= 1).all()


class TestEvaluator:
    def test_eval(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])
        metrics = evaluator.evaluate_model("TestModel", y_true, y_pred, scores)
        assert "f1_score" in metrics
        assert metrics["f1_score"] > 0.0


if __name__ == "__main__":
    print("[GridGuard Unit Tests]")
    TestSyntheticGenerator().test_shape()
    TestSyntheticGenerator().test_anomalies()
    TestColumnMapper().test_datetime()
    TestColumnMapper().test_target()
    TestSARIMA().test_residual_extraction()
    TestTransformer().test_forward()
    TestEvaluator().test_eval()
    print("All unit tests passed!")
