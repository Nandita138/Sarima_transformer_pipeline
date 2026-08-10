import numpy as np  
import pandas as pd  
import warnings  
import joblib  
from typing import Optional, Tuple, Dict  
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
    def _load_config(config_path: str) -> Dict:  
        with open(config_path, "r") as f:  
            cfg = yaml.safe_load(f)  
        return cfg.get("sarima", {})

    def fit(self, series: pd.Series, use_auto: bool = True):  
        print("[SARIMA] Fitting model...")  
        if use_auto and self.config.get("auto_arima", True):  
            self._fit_auto_arima(series)  
        else:  
            self._fit_manual(series)  
        self.predictions = self.fitted_model.fittedvalues  
        self.residuals = series.values[:len(self.predictions)] - self.predictions.values  
        print(f"[SARIMA] Fit complete. Residual std: {np.std(self.residuals):.4f}")  
        return self

    def _fit_auto_arima(self, series: pd.Series):  
        import pmdarima as pm  
        self.model = pm.auto_arima(  
            series,  
            seasonal=self.config.get("seasonal", True),  
            m=self.config.get("m", 96),  
            max_p=self.config.get("max_p", 3),  
            max_q=self.config.get("max_q", 3),  
            stepwise=self.config.get("stepwise", True),  
            suppress_warnings=self.config.get("suppress_warnings", True),  
            error_action="ignore",  
            trace=False,  
        )  
        self.fitted_model = self.model  
        print(f"[SARIMA] Auto-ARIMA order: {self.model.order}, seasonal: {self.model.seasonal_order}")

    def _fit_manual(self, series: pd.Series):  
        from statsmodels.tsa.statespace.sarimax import SARIMAX  
        order = tuple(self.config.get("order", [1, 1, 1]))  
        seasonal_order = tuple(self.config.get("seasonal_order", [1, 1, 1, 96]))  
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,  
                        enforce_stationarity=False, enforce_invertibility=False)  
        self.fitted_model = model.fit(disp=False, maxiter=200)  
        self.model = model

    def predict(self, steps: int) -> np.ndarray:  
        if self.fitted_model is None:  
            raise ValueError("Model not fitted.")  
        forecast = self.fitted_model.predict(n_periods=steps) if hasattr(  
            self.fitted_model, "predict") else self.fitted_model.forecast(steps=steps)  
        return np.array(forecast)

    def get_residuals(self, series: Optional[pd.Series] = None) -> np.ndarray:  
        if series is not None and self.fitted_model is not None:  
            predictions = self._rolling_forecast(series)  
            return series.values - predictions  
        if self.residuals is None:  
            raise ValueError("No residuals available.")  
        return self.residuals

    def _rolling_forecast(self, series: pd.Series) -> np.ndarray:  
        predictions = []  
        for i in range(len(series)):  
            pred = self.fitted_model.predict(n_periods=1)  
            predictions.append(pred[0] if hasattr(pred, "__len__") else pred)  
            try:  
                self.fitted_model.update(series.iloc[i:i+1])  
            except Exception:  
                pass  
        return np.array(predictions)

    def compute_anomaly_scores(self, residuals=None, threshold_std=3.0):  
        if residuals is None:  
            residuals = self.residuals  
        if residuals is None:  
            raise ValueError("No residuals available.")  
        mean = np.mean(residuals)  
        std = np.std(residuals)  
        scores = np.abs((residuals - mean) / (std + 1e-8))  
        labels = (scores > threshold_std).astype(int)  
        return scores, labels

    def save(self, filepath: str = "models_saved/sarima_model.pkl"):  
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)  
        joblib.dump(self.fitted_model, filepath)  
        print(f"[SARIMA] Model saved to {filepath}")

    def load(self, filepath: str = "models_saved/sarima_model.pkl"):  
        self.fitted_model = joblib.load(filepath)  
