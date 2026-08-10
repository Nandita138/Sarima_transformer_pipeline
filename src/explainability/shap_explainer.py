import numpy as np  
import shap  
import yaml  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
from typing import Dict, List, Optional  
from pathlib import Path


class GridGuardExplainer:  
    def __init__(self, model, feature_names, config_path="config/model_config.yaml"):  
        self.model = model  
        self.feature_names = feature_names  
        self.config = self._load_config(config_path)  
        self.explainer = None  
        self.background_data = None

    @staticmethod  
    def _load_config(config_path):  
        with open(config_path, "r") as f:  
            cfg = yaml.safe_load(f)  
        return cfg.get("shap", {})

    def setup(self, background_data):  
        num_bg = self.config.get("num_background_samples", 100)  
        if len(background_data) > num_bg:  
            indices = np.random.choice(len(background_data), num_bg, replace=False)  
            self.background_data = background_data[indices]  
        else:  
            self.background_data = background_data

        bg_flat = self.background_data.reshape(len(self.background_data), -1)

        def predict_fn(X_flat):  
            window_size = self.background_data.shape[1]  
            num_features = self.background_data.shape[2]  
            X_3d = X_flat.reshape(-1, window_size, num_features)  
            return self.model.predict(X_3d)

        self.explainer = shap.KernelExplainer(predict_fn, bg_flat)  
        print(f"[SHAP] Explainer initialized with {len(self.background_data)} background samples.")

    def explain(self, X, top_k=5):  
        if self.explainer is None:  
            raise ValueError("Call setup() first.")  
        X_flat = X.reshape(len(X), -1)  
        shap_values = self.explainer.shap_values(X_flat, nsamples=200)

        window_size = X.shape[1]  
        num_features = X.shape[2]  
        shap_3d = np.array(shap_values).reshape(len(X), window_size, num_features)  
        feature_importance = np.mean(np.abs(shap_3d), axis=1)

        text_explanations = self._generate_text(shap_3d, feature_importance, top_k)  
        return {  
            "shap_values": shap_values,  
            "shap_3d": shap_3d,  
            "feature_importance": feature_importance,  
            "text_explanations": text_explanations,  
        }

    def _generate_text(self, shap_3d, feature_importance, top_k):  
        explanations = []  
        top_k = min(top_k, len(self.feature_names))  
        for i in range(len(feature_importance)):  
            ranked = np.argsort(-feature_importance[i])[:top_k]  
            lines = [f"--- Anomaly Explanation (Sample {i}) ---"]  
            for rank, feat_idx in enumerate(ranked):  
                fname = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"  
                imp = feature_importance[i, feat_idx]  
                tc = shap_3d[i, :, feat_idx]  
                peak = np.argmax(np.abs(tc))  
                direction = "increased" if tc[peak] > 0 else "decreased"  
                lines.append(f"  {rank+1}. {fname} contributed {imp:+.4f} ({direction})")  
            explanations.append("\n".join(lines))  
        return explanations

    def plot_summary(self, X, save_path=None):  
        if self.explainer is None:  
            raise ValueError("Call setup() first.")  
        X_flat = X.reshape(len(X), -1)  
        shap_values = self.explainer.shap_values(X_flat, nsamples=100)  
        plt.figure(figsize=(12, 8))  
        shap.summary_plot(shap_values, X_flat, plot_type="bar", max_display=20, show=False)  
        if save_path:  
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)  
            plt.savefig(save_path, dpi=150, bbox_inches="tight")  
            print(f"[SHAP] Plot saved to {save_path}")  
        plt.close()  
