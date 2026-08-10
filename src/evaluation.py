import time  
import numpy as np  
import pandas as pd  
from typing import Dict, Callable  
from sklearn.metrics import (precision_score, recall_score, f1_score,  
                             roc_auc_score, average_precision_score, confusion_matrix)


class ModelEvaluator:  
    def __init__(self):  
        self.results = {}

    def evaluate_model(self, model_name, true_labels, pred_labels, scores,  
                       predict_fn=None, X_test=None):  
        metrics = {  
            "precision": precision_score(true_labels, pred_labels, zero_division=0),  
            "recall": recall_score(true_labels, pred_labels, zero_division=0),  
            "f1_score": f1_score(true_labels, pred_labels, zero_division=0),  
            "roc_auc": self._safe_roc_auc(true_labels, scores),  
            "avg_precision": self._safe_avg_precision(true_labels, scores),  
        }  
        cm = confusion_matrix(true_labels, pred_labels)  
        if cm.shape == (2, 2):  
            tn, fp, fn, tp = cm.ravel()  
            metrics["true_positives"] = int(tp)  
            metrics["false_positives"] = int(fp)  
            metrics["false_alarm_rate"] = fp / (fp + tn + 1e-8)  
        if predict_fn and X_test is not None:  
            metrics["latency_ms"] = self._measure_latency(predict_fn, X_test)  
        self.results[model_name] = metrics  
        return metrics

    def _measure_latency(self, predict_fn, X, num_runs=10):  
        predict_fn(X[:1])  
        latencies = []  
        for _ in range(num_runs):  
            start = time.perf_counter()  
            predict_fn(X[:1])  
            latencies.append((time.perf_counter() - start) * 1000)  
        return float(np.mean(latencies))

    @staticmethod  
    def _safe_roc_auc(y_true, scores):  
        try:  
            if len(np.unique(y_true)) < 2:  
                return 0.0  
            return roc_auc_score(y_true, scores)  
        except ValueError:  
            return 0.0

    @staticmethod  
    def _safe_avg_precision(y_true, scores):  
        try:  
            if len(np.unique(y_true)) < 2:  
                return 0.0  
            return average_precision_score(y_true, scores)  
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

    def get_best_model(self, metric="f1_score"):  
        df = self.compare_models()  
        return df[metric].idxmax()  
