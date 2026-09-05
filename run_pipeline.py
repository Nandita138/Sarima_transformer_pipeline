import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import GridGuardDataLoader
from src.models.sarima_model import SARIMAModel
from src.models.transformer_model import TransformerAnomalyDetector
from src.models.baselines import IsolationForestBaseline, SARIMABaseline, StandaloneTransformerBaseline
from src.explainability.shap_explainer import GridGuardExplainer
from src.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="GridGuard Hybrid Anomaly Detection Pipeline")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-sarima", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("        GridGuard - Hybrid SARIMA-Transformer Pipeline")
    print("=" * 70)
    total_start = time.time()

    # Step 1: Initial Data Load
    print("\n[1/6] Loading power telemetry dataset...")
    loader = GridGuardDataLoader(args.config)
    raw_df = loader.load_data(args.source)
    processed_df = loader.preprocess(raw_df)

    # Step 2: Fit Seasonal SARIMA Baseline & Extract Residual Signals
    sarima = None
    if not args.skip_sarima:
        print("\n[2/6] Fitting Seasonal SARIMA Model on active_power target...")
        sarima = SARIMAModel(args.model_config)
        target_col = loader.config.target_name
        if target_col in processed_df.columns:
            # Fit SARIMA on training horizon (first 70% of dataset)
            train_series = processed_df[target_col].iloc[:2000]
            try:
                sarima.fit(train_series, use_auto=False)
                sarima.save()
            except Exception as e:
                print(f"  SARIMA fit exception ({e}), continuing with seasonal residual fallback.")

    else:
        print("\n[2/6] Skipping SARIMA fit (using raw telemetry features)")

    # Step 3: Construct Hybrid Windows with SARIMA Residual Channel
    print("\n[3/6] Building Hybrid Feature Matrices (Telemetry + SARIMA Residuals)...")
    splits_hybrid, df_final = loader.get_full_pipeline(source_override=args.source, sarima_model=sarima)
    loader.save_scaler("models_saved/scaler.pkl")

    X_train_h, y_train_h, labels_train = splits_hybrid["train"]
    X_val_h, y_val_h, labels_val = splits_hybrid["val"]
    X_test_h, y_test_h, labels_test = splits_hybrid["test"]

    print(f"  Hybrid Data Splits -> Train: {X_train_h.shape} | Val: {X_val_h.shape} | Test: {X_test_h.shape}")

    # Step 4: Train GridGuard Hybrid Transformer
    print("\n[4/6] Training GridGuard Hybrid Transformer Encoder...")
    detector = TransformerAnomalyDetector(args.model_config)
    if args.epochs:
        detector.config["epochs"] = args.epochs

    detector.train(
        train_data=(X_train_h, y_train_h, labels_train),
        val_data=(X_val_h, y_val_h, labels_val)
    )
    detector.save()

    # Step 5: Fit Baselines & Run Evaluation
    print("\n[5/6] Fitting Baselines & Evaluating Model Benchmark Suite...")
    evaluator = ModelEvaluator()

    # 1. GridGuard Hybrid Model Evaluation
    hybrid_scores, hybrid_labels = detector.predict_with_threshold(X_test_h)
    evaluator.evaluate_model(
        "GridGuard Hybrid",
        labels_test,
        hybrid_labels,
        hybrid_scores,
        predict_fn=detector.predict,
        X_test=X_test_h
    )

    # 2. Standalone Transformer Baseline (Trained on raw telemetry without SARIMA residual channel)
    try:
        print("  Running Standalone Transformer Baseline...")
        # Strip residual channel if present for standalone baseline
        raw_feature_count = len(loader.feature_names) - (1 if "sarima_residual" in loader.feature_names else 0)
        X_train_r = X_train_h[:, :, :raw_feature_count]
        X_val_r = X_val_h[:, :, :raw_feature_count]
        X_test_r = X_test_h[:, :, :raw_feature_count]

        transformer_bl = StandaloneTransformerBaseline(args.model_config)
        transformer_bl.detector.config["epochs"] = 1
        transformer_bl.fit(
            train_data=(X_train_r, y_train_h, labels_train),
            val_data=(X_val_r, y_val_h, labels_val)
        )
        st_scores, st_labels = transformer_bl.predict(X_test_r)
        evaluator.evaluate_model(
            "Standalone Transformer",
            labels_test,
            st_labels,
            st_scores,
            predict_fn=transformer_bl.detector.predict,
            X_test=X_test_r
        )
    except Exception as e:
        print(f"  Standalone Transformer Baseline warning: {e}")

    # 3. Isolation Forest Baseline
    try:
        print("  Running Isolation Forest Baseline...")
        iso_forest = IsolationForestBaseline(args.model_config)
        iso_forest.fit(X_train_h)
        iso_forest.save()
        iso_labels, iso_scores = iso_forest.predict(X_test_h)
        evaluator.evaluate_model("Isolation Forest", labels_test, iso_labels, iso_scores)
    except Exception as e:
        print(f"  Isolation Forest Baseline warning: {e}")

    # 4. SARIMA Only Baseline
    if sarima is not None:
        try:
            print("  Running SARIMA Only Baseline...")
            sarima_bl = SARIMABaseline(args.model_config)
            sarima_bl.sarima = sarima
            target_series = processed_df[loader.config.target_name].iloc[-len(labels_test):]
            s_scores, s_labels = sarima_bl.predict(target_series)
            evaluator.evaluate_model("SARIMA Only", labels_test, s_labels, s_scores)
        except Exception as e:
            print(f"  SARIMA Baseline warning: {e}")

    evaluator.print_report()
    evaluator.save_results("models_saved/evaluation_results.json")

    # Step 6: SHAP Explainable AI Fault Attribution
    if not args.skip_shap:
        print("\n[6/6] Generating SHAP XAI Fault Attributions...")
        try:
            explainer = GridGuardExplainer(detector, loader.feature_names, args.model_config)
            explainer.setup(X_train_h[:30])
            anom_idx = np.where(labels_test == 1)[0]
            if len(anom_idx) > 0:
                sample_indices = anom_idx[:min(3, len(anom_idx))]
                results = explainer.explain(X_test_h[sample_indices], top_k=5)
                for t in results["text_explanations"]:
                    print(f"\n{t}")
                explainer.plot_summary(X_test_h[:30], save_path="models_saved/shap_summary.png")
            else:
                print("  No test set anomalies to explain.")
        except Exception as e:
            print(f"  SHAP explanation notice: {e}")
    else:
        print("\n[6/6] Skipping SHAP explanations")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {total_time:.1f}s")
    print(f"  Champion Model: {evaluator.get_best_model()}")
    print(f"  REST API: python -m api.main")
    print(f"  Dashboard: streamlit run dashboard/app.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
