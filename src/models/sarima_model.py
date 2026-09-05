import numpy as np
import pandas as pd
import warnings
import joblib
from typing import Optional, Tuple, Dict, Union
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")


class SARIMAModel:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.fitted_model = None
        self.residuals: Optional[np.ndarray] = None
        self.predictions: Optional[np.ndarray] = None

    @staticmethod
    def _load_config(config_path: Union[str, Dict]) -> Dict:
        if isinstance(config_path, dict):
            return config_path.get("sarima", {})
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("sarima", {})
        return {}

    def fit(self, series: pd.Series, use_auto: bool = False):
        print(f"[SARIMA] Fitting model on series (length: {len(series)})...")
        if use_auto and self.config.get("auto_arima", False):
            self._fit_auto_arima(series)
        else:
            self._fit_manual(series)
        
        # Calculate fitted values and residuals
        if hasattr(self.fitted_model, "fittedvalues"):
            self.predictions = np.asarray(self.fitted_model.fittedvalues)
            self.residuals = series.values - self.predictions
        else:
            self.predictions = series.values
            self.residuals = np.zeros_like(series.values)

        print(f"[SARIMA] Fit complete. Residual std: {np.std(self.residuals):.4f}")
        return self

    def _fit_auto_arima(self, series: pd.Series):
        try:
            import pmdarima as pm
            m_val = self.config.get("m", 24)
            self.model = pm.auto_arima(
                series,
                seasonal=self.config.get("seasonal", True),
                m=m_val,
                max_p=self.config.get("max_p", 2),
                max_q=self.config.get("max_q", 2),
                stepwise=self.config.get("stepwise", True),
                suppress_warnings=self.config.get("suppress_warnings", True),
                error_action="ignore",
                trace=False,
            )
            self.fitted_model = self.model
            print(f"[SARIMA] Auto-ARIMA order: {self.model.order}, seasonal: {self.model.seasonal_order}")
        except Exception as e:
            print(f"[SARIMA] Auto-ARIMA failed ({e}), falling back to manual SARIMAX...")
            self._fit_manual(series)

    def _fit_manual(self, series: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = tuple(self.config.get("order", [1, 1, 0]))
        seasonal_order = tuple(self.config.get("seasonal_order", [1, 0, 0, 24]))
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = model.fit(disp=False, maxiter=30)
        self.model = model

    def predict(self, steps: int) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")
        if hasattr(self.fitted_model, "predict"):
            try:
                forecast = self.fitted_model.predict(n_periods=steps)
            except Exception:
                forecast = self.fitted_model.forecast(steps=steps)
        else:
            forecast = self.fitted_model.forecast(steps=steps)
        return np.asarray(forecast)

    def get_residuals(self, series: Optional[pd.Series] = None) -> np.ndarray:
        if series is None:
            if self.residuals is None:
                raise ValueError("No residuals available.")
            return self.residuals

        # 1. Primary path: Use fitted SARIMAX / pmdarima state-space filter model
        if self.fitted_model is not None:
            try:
                if hasattr(self.fitted_model, "apply"):
                    applied = self.fitted_model.apply(series)
                    return np.asarray(applied.resid)
                elif hasattr(self.fitted_model, "arima_res_") and hasattr(self.fitted_model.arima_res_, "apply"):
                    applied = self.fitted_model.arima_res_.apply(series)
                    return np.asarray(applied.resid)
            except Exception as e:
                print(f"[SARIMA] Notice: fitted_model.apply() failed ({e}), falling back to seasonal heuristic.")

        # 2. Fallback path: Heuristic seasonal difference calculation
        m = self.config.get("m", 24)
        arr = series.values
        if len(arr) >= m:
            seasonal_lag = pd.Series(arr).shift(m).bfill().values
            trend_component = pd.Series(arr).rolling(window=m, min_periods=1).mean().values
            baseline = 0.5 * (seasonal_lag + trend_component)
            return arr - baseline

        return arr - np.mean(arr)

    def compute_anomaly_scores(self, residuals=None, threshold_std=3.0) -> Tuple[np.ndarray, np.ndarray]:
        if residuals is None:
            residuals = self.residuals
        if residuals is None:
            raise ValueError("No residuals available.")
        mean = np.mean(residuals)
        std = np.std(residuals) + 1e-8
        scores = np.abs((residuals - mean) / std)
        labels = (scores > threshold_std).astype(int)
        return scores, labels

    def save(self, filepath: str = "models_saved/sarima_model.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        if self.fitted_model is not None and hasattr(self.fitted_model, "remove_data"):
            try:
                self.fitted_model.remove_data()
            except Exception:
                pass
        joblib.dump(self.fitted_model, filepath)
        print(f"[SARIMA] Lightweight model saved to {filepath}")

    def load(self, filepath: str = "models_saved/sarima_model.pkl"):
        if Path(filepath).exists():
            self.fitted_model = joblib.load(filepath)
            print(f"[SARIMA] Model loaded from {filepath}")
