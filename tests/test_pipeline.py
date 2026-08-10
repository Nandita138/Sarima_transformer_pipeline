import sys  
import numpy as np  
import pandas as pd  
import pytest  
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  
from src.data_loader import SyntheticFeederGenerator, ColumnMapper, DataConfig


class TestSyntheticGenerator:  
    def test_shape(self):  
        config = DataConfig(num_days=7, sample_freq="15min")  
        gen = SyntheticFeederGenerator(config)  
        df = gen.generate()  
        assert len(df) == 7 * 24 * 4  
        assert "active_power" in df.columns

    def test_anomalies(self):  
        config = DataConfig(num_days=30, anomaly_ratio=0.05)  
        gen = SyntheticFeederGenerator(config)  
        df = gen.generate()  
        assert 0.01 < df["anomaly_label"].mean() < 0.15


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


class TestTransformer:  
    def test_forward(self):  
        from src.models.transformer_model import TransformerEncoder  
        import torch  
        model = TransformerEncoder(input_dim=10, d_model=32, nhead=4, num_layers=2)  
        x = torch.randn(2, 24, 10)  
        out = model(x)  
        assert out.shape == (2, 1)  
        assert (out >= 0).all() and (out <= 1).all()


if __name__ == "__main__":  
    pytest.main([__file__, "-v"])  
