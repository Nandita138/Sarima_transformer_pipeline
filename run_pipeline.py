import argparse  
import time  
import numpy as np  
from src.data_loader import GridGuardDataLoader  
from src.models.sarima_model import SARIMAModel  
from src.models.transformer_model import TransformerAnomalyDetector  
from src.models.baselines import IsolationForestBaseline, SARIMABaseline  
from src.explainability.shap_explainer import GridGuardExplainer  
from src.evaluation import ModelEvaluator


def main():  
    parser = argparse.ArgumentParser(description="GridGuard Pipeline")  
    parser.add_argument("--source", type=str, default=None)  
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml")  
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")  
    parser.add_argument("--epochs", type=int, default=None)  
    parser.add_argument("--skip-sarima", action="store_true")  
    parser.add_argument("--skip-shap", action="store_true")  
    args = parser.parse_args()

    print("=" * 70)  
    print("        GridGuard - Anomaly Detection Pipeline")  
    print("=" * 70)  
    total_start = time.time()

    # Step 1: Data  
    print("\n[1/6] Loading data...")  
    loader = GridGuardDataLoader(args.config)  
    splits, df_processed = loader.get_full_pipeline(source_override=args.source)  
    loader.save_scaler("models_saved/scaler.pkl")  
    X_train, y_train, labels_train = splits["train"]  
    X_val, y_val, labels_val = splits["val"]  
    X_test, y_test, labels_test = splits["test"]  
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Step 2: SARIMA  
    sarima = None  
    if not args.skip_sarima:  
        print("\n[2/6] Fitting SARIMA...")  
        sarima = SARIMAModel(args.model_config)  
        target_col = loader.config.target_name  
        if target_col in df_processed.columns:  
            series = df_processed[target_col].iloc[:5000]  
            try:  
                sarima.fit(series, use_auto=True)  
                sarima.save()  
            except Exception as e:  
                print(f"  SARIMA failed: {e}")  
                sarima = None  
    else:  
        print("\n[2/6] Skipping SARIMA")

    # Step 3: Transformer  
    print("\n[3/6] Training Transformer...")  
    detector = TransformerAnomalyDetector(args.model_config)  
    if args.epochs:  
        detector.config["epochs"] = args.epochs  
    detector.train(train_data=(X_train, y_train, labels_train),  
                   val_data=(X_val, y_val, labels_val))  
    detector.save()

    # Step 4: Baselines  
    print("\n[4/6] Running baselines...")  
    iso_forest = IsolationForestBaseline(args.model_config)  
    iso_forest.fit(X_train)  
    iso_forest.save()

    # Step 5: Evaluate  
    print("\n[5/6] Evaluating...")  
    evaluator = ModelEvaluator()  
    hybrid_scores = detector.predict(X_test)  
    hybrid_labels = (hybrid_scores >= 0.5).astype(int)  
    evaluator.evaluate_model("GridGuard Hybrid", labels_test, hybrid_labels, hybrid_scores,  
                             predict_fn=detector.predict, X_test=X_test)  
    iso_labels, iso_scores = iso_forest.predict(X_test)  
    evaluator.evaluate_model("Isolation Forest", labels_test, iso_labels, iso_scores)

    if sarima is not None:  
        sarima_bl = SARIMABaseline(args.model_config)  
        sarima_bl.sarima = sarima  
        try:  
            s_scores, s_labels = sarima_bl.predict()  
            ml = min(len(labels_test), len(s_labels))  
            if ml > 0:  
                evaluator.evaluate_model("SARIMA Only", labels_test[:ml], s_labels[:ml], s_scores[:ml])  
        except Exception:  
            pass

    evaluator.print_report()

    # Step 6: SHAP  
    if not args.skip_shap:  
        print("\n[6/6] SHAP explanations...")  
        try:  
            explainer = GridGuardExplainer(detector, loader.feature_names, args.model_config)  
            explainer.setup(X_train[:100])  
            anom_idx = np.where(labels_test == 1)[0]  
            if len(anom_idx) > 0:  
                results = explainer.explain(X_test[anom_idx[:3]], top_k=5)  
                for t in results["text_explanations"][:2]:  
                    print(f"  {t}\n")  
                explainer.plot_summary(X_test[:50], save_path="models_saved/shap_summary.png")  
        except Exception as e:  
            print(f"  SHAP failed: {e}")  
    else:  
        print("\n[6/6] Skipping SHAP")

    total_time = time.time() - total_start  
    print(f"\n{'=' * 70}")  
    print(f"  Pipeline complete in {total_time:.1f}s")  
    print(f"  Best model: {evaluator.get_best_model()}")  
    print(f"  API: python -m api.main")  
    print(f"  Dashboard: streamlit run dashboard/app.py")  
    print(f"{'=' * 70}")


if __name__ == "__main__":  
    main()  
