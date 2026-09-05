import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Callable
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, model_name, true_labels, pred_labels, scores,
                       predict_fn=None, X_test=None):
        min_len = min(len(true_labels), len(pred_labels), len(scores))
        t_labels = true_labels[:min_len]
        p_labels = pred_labels[:min_len]
        s_scores = scores[:min_len]

        metrics = {
            "precision": float(precision_score(t_labels, p_labels, zero_division=0)),
            "recall": float(recall_score(t_labels, p_labels, zero_division=0)),
            "f1_score": float(f1_score(t_labels, p_labels, zero_division=0)),
            "roc_auc": float(self._safe_roc_auc(t_labels, s_scores)),
            "avg_precision": float(self._safe_avg_precision(t_labels, s_scores)),
        }
        cm = confusion_matrix(t_labels, p_labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["false_alarm_rate"] = float(fp / (fp + tn + 1e-8))
        else:
            metrics["true_positives"] = 0
            metrics["false_positives"] = 0
            metrics["false_alarm_rate"] = 0.0

        if predict_fn and X_test is not None:
            metrics["latency_ms"] = self._measure_latency(predict_fn, X_test)
        else:
            metrics["latency_ms"] = 0.0

        self.results[model_name] = metrics
        return metrics

    def _measure_latency(self, predict_fn, X, num_runs=10):
        try:
            predict_fn(X[:1])
            latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                predict_fn(X[:1])
                latencies.append((time.perf_counter() - start) * 1000)
            return float(np.mean(latencies))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_roc_auc(y_true, scores):
        try:
            if len(np.unique(y_true)) < 2:
                return 0.0
            return float(roc_auc_score(y_true, scores))
        except ValueError:
            return 0.0

    @staticmethod
    def _safe_avg_precision(y_true, scores):
        try:
            if len(np.unique(y_true)) < 2:
                return 0.0
            return float(average_precision_score(y_true, scores))
        except ValueError:
            return 0.0

    def compare_models(self):
        df = pd.DataFrame(self.results).T
        df.index.name = "Model"
        if "f1_score" in df.columns:
            df = df.sort_values("f1_score", ascending=False)
        return df.round(4)

    def print_report(self):
        df = self.compare_models()
        print("\n" + "=" * 70)
        print("          GridGuard - Model Comparison Report")
        print("=" * 70)
        print(df.to_string())
        print("=" * 70)

    def save_results(self, filepath="models_saved/evaluation_results.json"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[Evaluator] Results exported to {filepath}")

    def get_best_model(self, metric="f1_score"):
        df = self.compare_models()
        if df.empty:
            return "GridGuard Hybrid"
        return df[metric].idxmax()
