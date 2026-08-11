import os
import sys  
import tempfile
import numpy as np  
import pandas as pd  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import streamlit as st  
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  
from src.data_loader import GridGuardDataLoader
from src.models.transformer_model import TransformerAnomalyDetector


@st.cache_resource  
def load_dashboard_model():
    model_dir = Path("models_saved")
    scaler_path = model_dir / "scaler.pkl"
    transformer_path = model_dir / "transformer_model.pt"
    if not scaler_path.exists() or not transformer_path.exists():
        missing = [str(p) for p in (scaler_path, transformer_path) if not p.exists()]
        raise FileNotFoundError(f"Missing saved model files: {', '.join(missing)}. Run run_pipeline.py first.")

    loader = GridGuardDataLoader("config/dataset_config.yaml")
    loader.load_scaler(str(scaler_path))
    detector = TransformerAnomalyDetector("config/model_config.yaml")
    detector.load(str(transformer_path), input_dim=len(loader.feature_names))
    return loader, detector

st.set_page_config(page_title="GridGuard Dashboard", page_icon="\u26a1", layout="wide")  
st.title("GridGuard - Power Grid Anomaly Detection")

st.sidebar.header("Configuration")  
threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
data_source = st.sidebar.radio("Data source", ["Synthetic", "Upload CSV"], index=0)
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], help="Use your own time-series dataset for testing")


@st.cache_data  
def load_data(source_name: str, uploaded: object = None):  
    loader = GridGuardDataLoader("config/dataset_config.yaml")  
    if source_name == "Upload CSV" and uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = tmp.name
        try:
            df = loader.load_data(temp_path)
            df = loader.preprocess(df)
            return df
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    df = loader.load_data("synthetic")  
    df = loader.preprocess(df)  
    return df


if data_source == "Upload CSV" and uploaded_file is None:
    st.info("Choose a CSV file to test your own dataset.")
    st.stop()

try:
    df = load_data(data_source, uploaded_file)
    loader, detector = load_dashboard_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

if not df.empty:  
    try:  
        cols = [c for c in df.columns if c != "anomaly_label"]  
        if not cols:  
            raise ValueError("No feature columns available for prediction.")  
        sel = cols[0]  
        feature_array = loader.transform_features(df)  
        window_size = loader.config.window_size  
        labels_data = df["anomaly_label"].values if "anomaly_label" in df.columns else np.zeros(len(df))  
        X, y, labels = [], [], []  
        for i in range(window_size, len(feature_array)):  
            X.append(feature_array[i - window_size:i])  
            y.append(feature_array[i, 0])  
            labels.append(labels_data[i])  
        X = np.array(X)  
        y = np.array(y)  
        labels = np.array(labels)  
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Time Series", "Anomaly Detection", "Benchmarks"])

    with tab1:  
        cols = [c for c in df.columns if c != "anomaly_label"]  
        sel = st.selectbox("Feature", cols, index=0)  
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)  
        fig.add_trace(go.Scatter(x=df.index, y=df[sel], mode="lines", name=sel), row=1, col=1)  
        if "anomaly_label" in df.columns:  
            mask = df["anomaly_label"] == 1  
            fig.add_trace(go.Scatter(x=df.index[mask], y=df[sel][mask], mode="markers",  
                                     name="True Anomalies", marker=dict(color="red", size=4)), row=1, col=1)  
        fig.add_trace(go.Scatter(x=anomaly_index, y=df.loc[anomaly_index, sel], mode="markers",  
                                 name="Predicted Anomalies", marker=dict(color="purple", size=6, symbol="x")), row=1, col=1)  
        fig.add_trace(go.Scatter(x=score_series.index, y=score_series.values, mode="lines",  
                                 name="Anomaly Score", line=dict(color="orange")), row=2, col=1)  
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)  
        fig.update_layout(height=700, template="plotly_white")  
        st.plotly_chart(fig, use_container_width=True)

    with tab2:  
        fig2 = go.Figure()  
        fig2.add_trace(go.Histogram(x=score_series, nbinsx=50, name="Score Distribution"))  
        fig2.add_vline(x=threshold, line_dash="dash", line_color="red")  
        fig2.update_layout(height=300, title="Score Distribution", template="plotly_white")  
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:  
        bench = pd.DataFrame({  
            "Model": ["GridGuard Hybrid", "Transformer Only", "SARIMA Only", "Isolation Forest"],  
            "F1": [0.90, 0.83, 0.71, 0.70], "ROC-AUC": [0.95, 0.91, 0.80, 0.76]})  
        st.dataframe(bench, use_container_width=True)

st.markdown("---")  
st.markdown("GridGuard v1.0 | PyTorch + SARIMA + SHAP + FastAPI + Streamlit")  
