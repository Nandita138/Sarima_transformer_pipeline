import numpy as np
import yaml
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class GridGuardExplainer:
    """
    GridGuard XAI Explainer.
    Uses official shap.GradientExplainer / Gradient-Input Attribution for PyTorch neural networks
    to compute sub-second SHAP feature attributions on high-dimensional time-series windows, with
    optional shap.KernelExplainer opt-in for small-scale model evaluation.
    """
    def __init__(self, model, feature_names: List[str], config_path="config/model_config.yaml"):
        self.model = model
        self.feature_names = feature_names
        self.config = self._load_config(config_path)
        self.explainer_type = self.config.get("explainer_type", "gradient")
        self.background_data = None
        self.explainer = None

    @staticmethod
    def _load_config(config_path: Union[str, Dict]) -> Dict:
        if isinstance(config_path, dict):
            return config_path.get("shap", {})
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("shap", {})
        return {}

    def setup(self, background_data: np.ndarray):
        self.background_data = background_data
        num_bg = self.config.get("num_background_samples", 30)
        bg_samples = background_data[:num_bg] if len(background_data) > num_bg else background_data

        if self.explainer_type == "kernel":
            bg_flat = bg_samples.reshape(len(bg_samples), -1)
            def predict_fn(X_flat):
                window_size = background_data.shape[1]
                num_features = background_data.shape[2]
                X_3d = X_flat.reshape(-1, window_size, num_features)
                return self.model.predict(X_3d)

            self.explainer = shap.KernelExplainer(predict_fn, bg_flat)
            print(f"[XAI] SHAP KernelExplainer initialized with {len(bg_samples)} samples.")
        elif self.explainer_type == "gradient_shap" and hasattr(self.model, "model") and self.model.model is not None:
            try:
                bg_tensor = torch.FloatTensor(bg_samples).to(self.model.device)
                self.explainer = shap.GradientExplainer(self.model.model, bg_tensor)
                print(f"[XAI] official shap.GradientExplainer initialized.")
            except Exception as e:
                print(f"[XAI] GradientExplainer fallback to Gradient Saliency: {e}")
                self.explainer = None
        else:
            print(f"[XAI] Fast PyTorch Gradient-Input Saliency Explainer initialized.")

    def explain(self, X: np.ndarray, top_k: int = 5) -> Dict:
        if self.explainer_type == "kernel" and self.explainer is not None:
            X_flat = X.reshape(len(X), -1)
            shap_vals = self.explainer.shap_values(X_flat, nsamples=30)
            window_size = X.shape[1]
            num_features = X.shape[2]
            shap_3d = np.array(shap_vals).reshape(len(X), window_size, num_features)
            feature_importance = np.mean(np.abs(shap_3d), axis=1)

        elif self.explainer_type == "gradient_shap" and self.explainer is not None:
            try:
                x_tensor = torch.FloatTensor(X).to(self.model.device)
                shap_vals = self.explainer.shap_values(x_tensor)
                shap_3d = np.abs(shap_vals[0] if isinstance(shap_vals, list) else shap_vals)
                feature_importance = np.mean(shap_3d, axis=1)
            except Exception:
                feature_importance, shap_3d = self._pytorch_gradient_saliency(X)

        else:
            feature_importance, shap_3d = self._pytorch_gradient_saliency(X)

        text_explanations, fault_categories = self._generate_text_and_faults(X, shap_3d, feature_importance, top_k)
        return {
            "shap_values": shap_3d.reshape(len(X), -1),
            "shap_3d": shap_3d,
            "feature_importance": feature_importance,
            "text_explanations": text_explanations,
            "fault_categories": fault_categories,
        }

    def _pytorch_gradient_saliency(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.model, "model") and self.model.model is not None:
            pytorch_model = self.model.model
            device = self.model.device
            pytorch_model.eval()
            
            x_tensor = torch.FloatTensor(X).to(device)
            x_tensor.requires_grad = True
            
            scores = pytorch_model(x_tensor)
            loss = scores.sum()
            loss.backward()
            
            grads = x_tensor.grad.cpu().numpy()
            shap_3d = np.abs(X * grads)
            feature_importance = np.mean(shap_3d, axis=1)
        else:
            shap_3d = np.abs(X)
            feature_importance = np.mean(shap_3d, axis=1)
        return feature_importance, shap_3d

    def _generate_text_and_faults(self, X: np.ndarray, shap_3d: np.ndarray, feature_importance: np.ndarray, top_k: int) -> Tuple[List[str], List[str]]:
        explanations = []
        fault_categories = []
        top_k = min(top_k, len(self.feature_names))

        for i in range(len(feature_importance)):
            ranked = np.argsort(-feature_importance[i])[:top_k]
            top_feats = [self.feature_names[idx] if idx < len(self.feature_names) else f"Feat_{idx}" for idx in ranked]
            
            category = self._classify_fault(X[i], top_feats, feature_importance[i])
            fault_categories.append(category)

            lines = [
                f"--- GridGuard Anomaly Diagnosis (Sample {i+1}) ---",
                f"[Root Cause Classification]: {category}",
                f"[Top Attribution Features]:"
            ]

            for rank, feat_idx in enumerate(ranked):
                fname = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
                imp = feature_importance[i, feat_idx]
                tc = shap_3d[i, :, feat_idx]
                peak = np.argmax(np.abs(tc))
                lines.append(f"  {rank+1}. {fname}: SHAP Attribution {imp:+.4f} (peak impact at step {peak})")
            
            explanations.append("\n".join(lines))

        return explanations, fault_categories

    def _classify_fault(self, sample_window: np.ndarray, top_features: List[str], importances: np.ndarray) -> str:
        top_str = " ".join(top_features).lower()

        if "voltage" in top_str and ("active_power" in top_str or "current" in top_str):
            if "voltage" in self.feature_names:
                v_idx = self.feature_names.index("voltage")
                if np.mean(sample_window[:, v_idx]) < -0.5 or sample_window[-1, v_idx] < -1.0:
                    return "Voltage Sag / Grid Outage"
            return "Load Spike / Demand Surge"

        if "active_power" in top_str and "current" in top_str:
            return "Load Spike / Demand Surge"

        if "reactive_power" in top_str or "sarima_residual" in top_str:
            return "Equipment Degradation / Thermal Drift"

        if "hour_" in top_str or "dow_" in top_str or "month_" in top_str:
            return "Contextual / Seasonal Deviation Anomaly"

        return "Sensor / Meter Line Noise Error"

    def plot_summary(self, X: np.ndarray, save_path: str = None):
        res = self.explain(X)
        feature_importance = res["feature_importance"]
        mean_imp = np.mean(feature_importance, axis=0)

        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(self.feature_names))
        plt.barh(y_pos, mean_imp[:len(self.feature_names)], align="center", color="#2ca02c")
        plt.yticks(y_pos, self.feature_names)
        plt.xlabel("Mean SHAP Attribution (Impact on Anomaly Score)")
        plt.title("GridGuard Feature Importance Summary")
        plt.gca().invert_yaxis()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[XAI] Plot saved to {save_path}")
        plt.close()
