import numpy as np
import yaml
import joblib
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class IsolationForestBaseline:
    def __init__(self, config_path="config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None

    @staticmethod
    def _load_config(config_path):
        if isinstance(config_path, dict):
            return config_path.get("isolation_forest", {})
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("isolation_forest", {})
        return {}

    def fit(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        self.model = IsolationForest(
            n_estimators=self.config.get("n_estimators", 200),
            contamination=self.config.get("contamination", 0.03),
            max_samples=self.config.get("max_samples", "auto"),
            random_state=self.config.get("random_state", 42),
            n_jobs=-1
        )
        self.model.fit(X_flat)
        print(f"[IsolationForest] Fitted on {X_flat.shape[0]} samples")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted.")
        X_flat = X.reshape(X.shape[0], -1)
        raw_labels = self.model.predict(X_flat)
        labels = (raw_labels == -1).astype(int)
        scores = -self.model.decision_function(X_flat)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return labels, scores

    def evaluate(self, X, true_labels):
        pred_labels, scores = self.predict(X)
        return {
            "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
            "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
            "f1_score": float(f1_score(true_labels, pred_labels, zero_division=0)),
            "roc_auc": float(roc_auc_score(true_labels, scores)) if true_labels.sum() > 0 else 0.0,
        }

    def save(self, filepath="models_saved/isolation_forest.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath="models_saved/isolation_forest.pkl"):
        self.model = joblib.load(filepath)


class StandaloneTransformerBaseline:
    def __init__(self, config_path="config/model_config.yaml"):
        from .transformer_model import TransformerAnomalyDetector
        self.detector = TransformerAnomalyDetector(config_path)

    def fit(self, train_data, val_data):
        return self.detector.train(train_data, val_data)

    def predict(self, X):
        return self.detector.predict_with_threshold(X)

    def evaluate(self, X, true_labels):
        scores, pred_labels = self.predict(X)
        return {
            "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
            "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
            "f1_score": float(f1_score(true_labels, pred_labels, zero_division=0)),
            "roc_auc": float(roc_auc_score(true_labels, scores)) if true_labels.sum() > 0 else 0.0,
        }


class SARIMABaseline:
    def __init__(self, config_path="config/model_config.yaml"):
        from .sarima_model import SARIMAModel
        self.sarima = SARIMAModel(config_path)
        self.threshold_std = 3.0

    def fit(self, series):
        self.sarima.fit(series)

    def predict(self, series=None):
        residuals = self.sarima.get_residuals(series)
        scores, labels = self.sarima.compute_anomaly_scores(residuals, self.threshold_std)
        return scores, labels

    def evaluate(self, true_labels, series=None):
        scores, pred_labels = self.predict(series)
        min_len = min(len(true_labels), len(pred_labels))
        return {
            "precision": float(precision_score(true_labels[:min_len], pred_labels[:min_len], zero_division=0)),
            "recall": float(recall_score(true_labels[:min_len], pred_labels[:min_len], zero_division=0)),
            "f1_score": float(f1_score(true_labels[:min_len], pred_labels[:min_len], zero_division=0)),
            "roc_auc": float(roc_auc_score(true_labels[:min_len], scores[:min_len])) if true_labels[:min_len].sum() > 0 else 0.0,
        }
