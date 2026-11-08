# Breast Cancer Diagnostic Classifier

An end-to-end machine learning project that classifies breast tumour measurements as **malignant** or **benign**. It compares several models using cross-validation, saves the best one as a reusable pipeline, and serves predictions through a REST API.

> **Disclaimer:** This is an educational project built on the public Wisconsin Diagnostic Breast Cancer dataset. It is not a medical device and must not be used for clinical decisions.

## Features

- Reproducible training pipeline (fixed seeds, stratified split, no test-set leakage)
- Model comparison: logistic regression, RBF SVM, random forest, gradient boosting
- Evaluation with accuracy, precision/recall/F1 for the malignant class, and ROC-AUC
- Saved artifact bundling the pipeline, feature names, class names and metrics
- FastAPI service with input validation
- Automated tests (unit, integration, API)

## Project structure

```
bc-diagnostic-classifier/
├── src/bcclf/
│   ├── config.py          # settings and paths (env-overridable)
│   ├── logging_config.py  # logging setup
│   ├── data.py            # loading, validation, splitting
│   ├── model.py           # candidate pipelines
│   ├── evaluate.py        # metrics and plots
│   ├── train.py           # training CLI
│   ├── predict.py         # inference helpers
│   └── api.py             # FastAPI application
├── tests/
├── models/                # trained artifact (git-ignored)
├── reports/               # metrics and plots (git-ignored)
├── requirements.txt
├── pytest.ini
└── .env.example
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional
```

## Usage

Train (writes `models/model.joblib` and files in `reports/`):

```bash
PYTHONPATH=src python -m bcclf.train
PYTHONPATH=src python -m bcclf.train --models logistic_regression svm_rbf
```

Serve the API:

```bash
PYTHONPATH=src uvicorn bcclf.api:app --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`.

| Method | Path        | Description                              |
|--------|-------------|------------------------------------------|
| GET    | `/health`   | Service status and loaded model name     |
| GET    | `/features` | The 30 required feature names            |
| POST   | `/predict`  | Body: `{"features": {"mean radius": ...}}` |

## Testing

```bash
python -m pytest -q
```

## Configuration

| Variable       | Default                | Purpose                        |
|----------------|------------------------|--------------------------------|
| `MODEL_PATH`   | `models/model.joblib`  | Where the artifact is stored   |
| `REPORTS_DIR`  | `reports`              | Where metrics/plots are written|
| `RANDOM_STATE` | `42`                   | Seed                           |
| `TEST_SIZE`    | `0.2`                  | Hold-out fraction              |
| `CV_FOLDS`     | `5`                    | Cross-validation folds         |
| `LOG_LEVEL`    | `INFO`                 | Logging verbosity              |

## Results

On the 20% hold-out set (114 samples), the selected model (logistic regression) achieved accuracy 0.982, malignant recall 0.976 and ROC-AUC 0.995. Results come from a single split of a small dataset and should be read as indicative, not definitive.

## Limitations and next steps

- Small dataset (569 samples); add repeated CV or an external validation set
- Tune the decision threshold to favour malignant recall
- Add feature-importance or SHAP explanations
- Containerise with Docker and add CI
