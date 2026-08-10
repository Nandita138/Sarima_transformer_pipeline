# GridGuard

Real-Time Power Grid Anomaly Detection with Explainable AI

## Quick Start

    pip install -r requirements.txt
    python run_pipeline.py
    python -m api.main
    streamlit run dashboard/app.py

## Custom Dataset

1. Place CSV in data/ folder
2. Edit config/dataset_config.yaml
3. Run: python run_pipeline.py --source data/your_file.csv

## API Endpoints

- GET /health - Health check
- POST /predict - Anomaly scoring
- POST /explain - SHAP attributions
- GET /metrics - Performance metrics

## CLI Options

- python run_pipeline.py --skip-sarima (faster)
- python run_pipeline.py --epochs 10 (quick test)
- python run_pipeline.py --skip-shap (skip explanations)

## License

MIT
