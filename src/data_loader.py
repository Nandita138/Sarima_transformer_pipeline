import os
import yaml
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    source: str = "synthetic"
    datetime_aliases: List[str] = field(default_factory=list)
    target_name: str = "active_power"
    target_aliases: List[str] = field(default_factory=list)
    exogenous: Dict[str, List[str]] = field(default_factory=dict)
    resample_freq: str = "15min"
    imputation_method: str = "interpolate"
    scaler_type: str = "standard"
    window_size: int = 96
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cyclical_features: List[str] = field(default_factory=list)
    num_days: int = 90
    sample_freq: str = "15min"
    base_load_kw: float = 150.0
    noise_std: float = 5.0
    anomaly_ratio: float = 0.03
    num_feeders: int = 1


def load_config(config_path: str = "config/dataset_config.yaml") -> DataConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    ds = raw.get("dataset", {})
    prep = raw.get("preprocessing", {})
    syn = raw.get("synthetic_data", {})

    exog_map = {}
    for key, val in ds.get("exogenous", {}).items():
        exog_map[key] = val.get("aliases", [])

    return DataConfig(
        source=ds.get("source", "synthetic"),
        datetime_aliases=ds.get("datetime_aliases", []),
        target_name=ds.get("target", {}).get("name", "active_power"),
        target_aliases=ds.get("target", {}).get("aliases", []),
        exogenous=exog_map,
        resample_freq=prep.get("resample_freq", "15min"),
        imputation_method=prep.get("imputation_method", "interpolate"),
        scaler_type=prep.get("scaler_type", "standard"),
        window_size=prep.get("window_size", 96),
        horizon=prep.get("horizon", 1),
        train_ratio=prep.get("train_ratio", 0.7),
        val_ratio=prep.get("val_ratio", 0.15),
        test_ratio=prep.get("test_ratio", 0.15),
        cyclical_features=prep.get("cyclical_features", []),
        num_days=syn.get("num_days", 90),
        sample_freq=syn.get("sample_freq", "15min"),
        base_load_kw=syn.get("base_load_kw", 150.0),
        noise_std=syn.get("noise_std", 5.0),
        anomaly_ratio=syn.get("anomaly_ratio", 0.03),
        num_feeders=syn.get("num_feeders", 1),
    )


class SyntheticFeederGenerator:
    def __init__(self, config: DataConfig):
        self.config = config

    def generate(self) -> pd.DataFrame:
        cfg = self.config
        np.random.seed(42)

        periods = cfg.num_days * 24 * (60 // self._freq_to_minutes(cfg.sample_freq))
        idx = pd.date_range(start="2023-01-01", periods=periods, freq=cfg.sample_freq)

        hours = idx.hour + idx.minute / 60.0
        dow = idx.dayofweek

        daily_pattern = (
            cfg.base_load_kw
            + 50 * np.sin(2 * np.pi * (hours - 6) / 24)
            + 20 * np.sin(4 * np.pi * hours / 24)
        )

        weekly_factor = np.where(dow >= 5, 0.85, 1.0)
        load = daily_pattern * weekly_factor
        load += np.random.normal(0, cfg.noise_std, len(idx))

        voltage = 240 - 0.05 * (load - cfg.base_load_kw) + np.random.normal(0, 1.5, len(idx))
        current = (load * 1000) / (voltage * np.sqrt(3)) + np.random.normal(0, 0.3, len(idx))
        reactive_power = 0.2 * load + np.random.normal(0, 2.0, len(idx))

        df = pd.DataFrame({
            "datetime": idx,
            "active_power": load,
            "voltage": voltage,
            "current": current,
            "reactive_power": reactive_power,
        })
        df.set_index("datetime", inplace=True)
        df = self._inject_anomalies(df)
        return df

    def _inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        num_anomalies = int(n * self.config.anomaly_ratio)
        anomaly_labels = np.zeros(n, dtype=int)

        anomaly_types = ["spike", "dip", "drift", "outage"]
        anomaly_indices = np.random.choice(range(50, n - 50), size=num_anomalies, replace=False)

        for idx in anomaly_indices:
            atype = np.random.choice(anomaly_types, p=[0.3, 0.25, 0.25, 0.2])

            if atype == "spike":
                duration = np.random.randint(1, 5)
                end = min(idx + duration, n)
                mult = np.random.uniform(1.8, 3.0)
                df.iloc[idx:end, df.columns.get_loc("active_power")] *= mult
                # Physics coupling: Current increases, voltage drops under high load
                df.iloc[idx:end, df.columns.get_loc("current")] *= (mult * 0.9)
                df.iloc[idx:end, df.columns.get_loc("voltage")] *= 0.92
                anomaly_labels[idx:end] = 1

            elif atype == "dip":
                duration = np.random.randint(1, 5)
                end = min(idx + duration, n)
                # Voltage sag / power drop
                df.iloc[idx:end, df.columns.get_loc("voltage")] *= np.random.uniform(0.6, 0.8)
                df.iloc[idx:end, df.columns.get_loc("active_power")] *= np.random.uniform(0.2, 0.5)
                df.iloc[idx:end, df.columns.get_loc("current")] *= 0.6
                anomaly_labels[idx:end] = 1

            elif atype == "drift":
                duration = np.random.randint(10, 30)
                end = min(idx + duration, n)
                actual_len = end - idx
                drift = np.linspace(0, np.random.uniform(30, 80), actual_len)
                df.iloc[idx:end, df.columns.get_loc("active_power")] += drift
                df.iloc[idx:end, df.columns.get_loc("reactive_power")] += drift * 0.5
                anomaly_labels[idx:end] = 1

            elif atype == "outage":
                duration = np.random.randint(5, 20)
                end = min(idx + duration, n)
                df.iloc[idx:end, df.columns.get_loc("active_power")] = 0.0
                df.iloc[idx:end, df.columns.get_loc("voltage")] = 0.0
                df.iloc[idx:end, df.columns.get_loc("current")] = 0.0
                df.iloc[idx:end, df.columns.get_loc("reactive_power")] = 0.0
                anomaly_labels[idx:end] = 1

        df["anomaly_label"] = anomaly_labels
        return df

    @staticmethod
    def _freq_to_minutes(freq: str) -> int:
        if "T" in freq or "min" in freq:
            return int("".join(filter(str.isdigit, freq)) or "15")
        elif "H" in freq:
            return int("".join(filter(str.isdigit, freq)) or "1") * 60
        return 15


class ColumnMapper:
    def __init__(self, config: DataConfig):
        self.config = config

    def detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        columns_lower = {col.lower().strip(): col for col in df.columns}
        for alias in self.config.datetime_aliases:
            if alias.lower() in columns_lower:
                return columns_lower[alias.lower()]
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        try:
            pd.to_datetime(df.iloc[:5, 0])
            return df.columns[0]
        except (ValueError, TypeError):
            pass
        return None

    def map_target_column(self, df: pd.DataFrame) -> Optional[str]:
        return self._find_column(df, self.config.target_aliases)

    def map_exogenous_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapped = {}
        for feature_name, aliases in self.config.exogenous.items():
            col = self._find_column(df, aliases)
            if col:
                mapped[feature_name] = col
        return mapped

    @staticmethod
    def _find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
        columns_lower = {col.lower().strip(): col for col in df.columns}
        for alias in aliases:
            if alias.lower().strip() in columns_lower:
                return columns_lower[alias.lower().strip()]
        return None


class GridGuardDataLoader:
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.mapper = ColumnMapper(self.config)
        self.scaler = None
        self.feature_names: List[str] = []
        self._target_col: Optional[str] = None
        self._exog_cols: Dict[str, str] = {}

    def load_data(self, source_override: Optional[str] = None) -> pd.DataFrame:
        source = source_override or self.config.source
        if source.lower() == "synthetic":
            print("[GridGuard] Generating synthetic feeder data...")
            generator = SyntheticFeederGenerator(self.config)
            df = generator.generate()
        else:
            print(f"[GridGuard] Loading data from: {source}")
            df = self._load_csv(source)
        return df

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(filepath, sep=sep, low_memory=False)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
        dt_col = self.mapper.detect_datetime_column(df)
        if dt_col:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            df.set_index(dt_col, inplace=True)
            df.sort_index(inplace=True)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._map_columns(df)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex after loading.")
        df = self._resample(df)
        df = self._impute(df)
        df = self._add_cyclical_features(df)
        df.dropna(inplace=True)
        print(f"[GridGuard] Preprocessed shape: {df.shape}")
        return df

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.mapper.map_target_column(df)
        if target_col and target_col != self.config.target_name:
            df.rename(columns={target_col: self.config.target_name}, inplace=True)
        self._target_col = self.config.target_name
        exog_mapping = self.mapper.map_exogenous_columns(df)
        rename_map = {}
        for std_name, original_col in exog_mapping.items():
            if original_col != std_name:
                rename_map[original_col] = std_name
        df.rename(columns=rename_map, inplace=True)
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_resampled = df[numeric_cols].resample(self.config.resample_freq).mean()
        return df_resampled

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.config.imputation_method
        if method == "ffill":
            df.ffill(inplace=True)
            df.bfill(inplace=True)
        elif method == "interpolate":
            df.interpolate(method="time", inplace=True)
            df.bfill(inplace=True)
        elif method == "mean":
            df.fillna(df.mean(), inplace=True)
        else:
            df.ffill(inplace=True)
        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if "hour_sin" in self.config.cyclical_features:
            hours = idx.hour + idx.minute / 60.0
            df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        if "dow_sin" in self.config.cyclical_features:
            dow = idx.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        if "month_sin" in self.config.cyclical_features:
            month = idx.month
            df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
            df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        return df

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c != "anomaly_label"]
        self.feature_names = feature_cols
        if self.config.scaler_type == "standard":
            scaler_cls = StandardScaler
        else:
            scaler_cls = MinMaxScaler
        if fit:
            self.scaler = scaler_cls()
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            if self.scaler is None:
                raise ValueError("No scaler loaded. Run run_pipeline.py first.")
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def save_scaler(self, filepath: str = "models_saved/scaler.pkl") -> None:
        if self.scaler is None:
            raise ValueError("Scaler is not initialized. Normalize data with fit=True before saving.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, filepath)

    def load_scaler(self, filepath: str = "models_saved/scaler.pkl") -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}. Run run_pipeline.py first.")
        data = joblib.load(filepath)
        self.scaler = data.get("scaler")
        self.feature_names = data.get("feature_names", [])

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or not self.feature_names:
            raise ValueError("Scaler and feature names not loaded. Run run_pipeline.py first.")
        # Fill missing cyclical or feature columns if needed
        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0.0
        values = df[self.feature_names].astype(float).values
        return self.scaler.transform(values)

    def create_windows(self, df: pd.DataFrame, include_raw_only: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_cols = [c for c in df.columns if c != "anomaly_label"]
        if include_raw_only and "sarima_residual" in feature_cols:
            feature_cols = [c for c in feature_cols if c != "sarima_residual"]
        has_labels = "anomaly_label" in df.columns
        data = df[feature_cols].values
        labels = df["anomaly_label"].values if has_labels else np.zeros(len(df))
        window_size = self.config.window_size
        X, y, lbl = [], [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size: i])
            y.append(data[i, 0])
            lbl.append(labels[i])
        return np.array(X), np.array(y), np.array(lbl)

    def split_chronological(self, X, y, labels):
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return {
            "train": (X[:train_end], y[:train_end], labels[:train_end]),
            "val": (X[train_end:val_end], y[train_end:val_end], labels[train_end:val_end]),
            "test": (X[val_end:], y[val_end:], labels[val_end:]),
        }

    def get_full_pipeline(self, source_override=None, sarima_model=None):
        df = self.load_data(source_override)
        df = self.preprocess(df)
        df_processed = df.copy()

        # If SARIMA model is provided, extract residuals and attach channel
        if sarima_model is not None and self.config.target_name in df.columns:
            target_series = df[self.config.target_name]
            residuals = sarima_model.get_residuals(target_series)
            # Normalize residuals
            res_std = np.std(residuals) + 1e-8
            df["sarima_residual"] = (residuals - np.mean(residuals)) / res_std
        
        df = self.normalize(df, fit=True)
        X, y, labels = self.create_windows(df)
        splits = self.split_chronological(X, y, labels)
        return splits, df_processed
