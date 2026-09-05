import os
import sys
import json
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
from src.explainability.shap_explainer import GridGuardExplainer


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


st.set_page_config(page_title="GridGuard - Power Grid Anomaly Monitor", page_icon="⚡", layout="wide")
st.title("⚡ GridGuard: Hybrid SARIMA-Transformer Anomaly Detection & XAI")
st.markdown("*Real-Time Power Distribution Network Telemetry, Anomaly Scoring & SHAP Fault Diagnosis*")

st.sidebar.header("🕹️ Operator Controls")
threshold = st.sidebar.slider("Anomaly Detection Threshold", 0.0, 1.0, 0.40, 0.05)
data_source = st.sidebar.radio("Telemetry Data Source", ["Synthetic DISCOM Feeder", "Upload CSV"], index=0)
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV File", type=["csv"], help="Test telemetry from feeder meters")


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
    st.info("📌 Upload a CSV file in the sidebar to test your custom telemetry data.")
    st.stop()

try:
    df = load_data(data_source, uploaded_file)
    loader, detector = load_dashboard_model()
except FileNotFoundError as exc:
    st.error(f"⚠️ {exc}")
    st.warning("Please run `python run_pipeline.py` first to train and save the pipeline models.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

if not df.empty:
    try:
        feature_array = loader.transform_features(df)
        window_size = loader.config.window_size
        
        X = []
        for i in range(window_size, len(feature_array)):
            X.append(feature_array[i - window_size:i])
        X = np.array(X)
        
        scores = detector.predict(X)
        score_series = pd.Series(scores, index=df.index[loader.config.window_size:], name="anomaly_score")
        predicted_mask = score_series >= threshold
        anomaly_index = score_series[predicted_mask].index

    except Exception as exc:
        st.error(f"Unable to compute model predictions: {exc}")
        st.stop()

    # Top Metric KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Telemetry Steps", f"{len(df):,}")
    with col2:
        st.metric("Anomalies Flagged", f"{len(anomaly_index):,}", delta=f"{len(anomaly_index)/max(len(score_series),1):.1%} of data")
    with col3:
        st.metric("Detection Threshold", f"{threshold:.2f}")
    with col4:
        max_score = score_series.max() if len(score_series) > 0 else 0.0
        st.metric("Peak Anomaly Score", f"{max_score:.4f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Feeder Telemetry & Anomalies",
        "🧠 XAI Fault Diagnosis (SHAP)",
        "📊 Score Distribution",
        "🏆 Model Benchmarks"
    ])

    with tab1:
        cols = [c for c in df.columns if c != "anomaly_label" and c != "sarima_residual"]
        sel = st.selectbox("Select Telemetry Feature to Inspect", cols, index=0)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=(f"Feeder Parameter: {sel}", "GridGuard Hybrid Anomaly Score"))
        
        fig.add_trace(go.Scatter(x=df.index, y=df[sel], mode="lines", name=sel, line=dict(color="#1f77b4")), row=1, col=1)
        
        if "anomaly_label" in df.columns:
            mask = df["anomaly_label"] == 1
            fig.add_trace(go.Scatter(x=df.index[mask], y=df[sel][mask], mode="markers",
                                     name="Ground Truth Anomaly", marker=dict(color="red", size=5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=anomaly_index, y=df.loc[anomaly_index, sel], mode="markers",
                                 name="Flagged Anomaly", marker=dict(color="purple", size=7, symbol="x")), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=score_series.index, y=score_series.values, mode="lines",
                                 name="Anomaly Score", line=dict(color="orange")), row=2, col=1)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔍 Explainable AI (SHAP) - Root Cause Fault Attribution")
        if len(anomaly_index) == 0:
            st.info("No anomalies detected at the current threshold.")
        else:
            selected_anom_time = st.selectbox("Select Flagged Anomaly Timestamp for SHAP Analysis", anomaly_index[:20])
            idx_in_X = score_series.index.get_loc(selected_anom_time)
            
            st.markdown(f"**Analyzing Telemetry Window ending at `{selected_anom_time}`**")
            
            if st.button("Generate Instant SHAP Attribution"):
                with st.spinner("Calculating SHAP feature attributions..."):
                    explainer = GridGuardExplainer(detector, loader.feature_names, "config/model_config.yaml")
                    explainer.setup(X[:50])
                    sample_X = X[idx_in_X:idx_in_X+1]
                    res = explainer.explain(sample_X, top_k=5)
                    
                    st.success(f"**Root Cause Diagnosis**: {res['fault_categories'][0]}")
                    st.text_area("Detailed Explanation Log", res["text_explanations"][0], height=180)
                    
                    # Plot feature importance bar chart
                    importances = res["feature_importance"][0]
                    feat_df = pd.DataFrame({
                        "Feature": loader.feature_names,
                        "Attribution": importances
                    }).sort_values("Attribution", ascending=True)
                    
                    fig_shap = go.Figure(go.Bar(
                        x=feat_df["Attribution"],
                        y=feat_df["Feature"],
                        orientation="h",
                        marker=dict(color="#2ca02c")
                    ))
                    fig_shap.update_layout(title="SHAP Feature Contribution Importance", height=400, template="plotly_white")
                    st.plotly_chart(fig_shap, use_container_width=True)

    with tab3:
        st.subheader("Anomaly Score Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=score_series, nbinsx=60, name="Score Distribution", marker_color="#1f77b4"))
        fig2.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig2.update_layout(height=400, title="Probability Score Histogram", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("🏆 Model Benchmarks & Comparison Report")
        res_path = Path("models_saved/evaluation_results.json")
        if res_path.exists():
            with open(res_path, "r") as f:
                bench_dict = json.load(f)
            bench_df = pd.DataFrame(bench_dict).T
            bench_df.index.name = "Model"
            st.dataframe(bench_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
        else:
            st.warning("No evaluation results file found. Run `python run_pipeline.py` to generate dynamic benchmark comparison results.")

st.markdown("---")
st.markdown("⚡ **GridGuard v1.0** | *Hybrid SARIMA-Transformer Pipeline with Explainable AI*")
