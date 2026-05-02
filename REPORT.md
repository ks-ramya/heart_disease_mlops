# Heart Disease MLOps — Final Report

> **Course:** MLOps (S2-25_AMLCSZG523) — Assignment I
> **Dataset:** UCI Heart Disease (Cleveland) — 14 attributes, binary target
> **Goal:** Build a production-grade, monitored, cloud-ready ML service.

---

## 1. Setup & Installation

### Prerequisites
* Python 3.10+
* Docker 24+ (for containerised deployment)
* `kubectl` + Minikube/GKE/EKS/AKS (for K8s deployment)
* `helm` (optional, for chart-based deployment)

### Quick install
```bash
git clone <your-repo-url>
cd heart_disease_mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### End-to-end run (local)
```bash
make data        # download UCI heart disease CSV
make train       # train LR + RF, log to MLflow, save best model
make test        # run pytest
make serve       # start FastAPI on :8080
make compose-up  # API + Prometheus + Grafana stack
```

---

## 2. Data Acquisition & EDA  *(Task 1 — 5 marks)*

* **Source:** UCI ML Repository, dataset id 45 (Heart Disease, Cleveland).
* **Download script:** `src/data/download.py` — three-tier fallback
  (`ucimlrepo` package → UCI raw `.data` → public GitHub mirror).
* **Cleaning:** missing values represented as `?` are coerced to NaN and the
  affected rows dropped (~6 rows out of 303). The original 0–4 severity
  target is binarised to `{0, 1}` (presence vs absence of disease).
* **EDA notebook:** `notebooks/01_eda.ipynb` produces:
  - `class_balance.png` — class distribution
  - `numeric_distributions.png` — histograms of `age`, `trestbps`, `chol`,
    `thalach`, `oldpeak`, coloured by target
  - `categorical_vs_target.png` — stacked bars per categorical feature
  - `correlation_heatmap.png` — Pearson correlation matrix

**Findings:** the dataset is mildly imbalanced (~46% positives), `cp`, `thalach`,
and `oldpeak` are the strongest individual predictors, and the mixed
numeric/categorical structure motivates a `ColumnTransformer` preprocessing
pipeline.

---

## 3. Feature Engineering & Model Development  *(Task 2 — 8 marks)*

* **Preprocessing pipeline** (`src/data/preprocess.py`):
  `ColumnTransformer( StandardScaler(numeric) + OneHotEncoder(categorical) )`.
  This keeps preprocessing **inside** the saved sklearn pipeline, eliminating
  train/serve skew.
* **Models compared** (`src/models/train.py`):
  | Model | Hyperparameter grid |
  |-------|---------------------|
  | Logistic Regression | `C ∈ {0.1, 1, 10}` |
  | Random Forest       | `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 5, 10}`, `min_samples_split ∈ {2, 5}` |
* **Selection:** `GridSearchCV` with **5-fold StratifiedKFold**, optimised on
  **ROC-AUC**, refit on the full training set.
* **Evaluation metrics** logged for each model:
  Accuracy, Precision, Recall, F1, ROC-AUC, plus confusion-matrix and
  ROC-curve PNGs under `reports/figures/`.

Typical results on the held-out test set:

| Model | CV ROC-AUC | Test Acc | Test Precision | Test Recall | Test ROC-AUC |
|-------|-----------:|---------:|---------------:|------------:|-------------:|
| Logistic Regression | ~0.90 | ~0.85 | ~0.85 | ~0.86 | ~0.91 |
| Random Forest       | ~0.89 | ~0.83 | ~0.83 | ~0.83 | ~0.90 |

(Numbers will vary slightly between runs; exact values land in
`models/metrics.json` and the MLflow UI.)

---

## 4. Experiment Tracking  *(Task 3 — 5 marks)*

`src/models/train.py` integrates **MLflow**:
* Every model gets its own `mlflow.start_run`.
* Logged: hyperparameters from the grid search, CV/test metrics,
  confusion-matrix + ROC-curve PNGs, and the full sklearn pipeline as a
  reusable MLflow model.
* Tracking URI defaults to `file:./mlruns` so the workflow runs **without
  external infra**, but is overridable via `MLFLOW_TRACKING_URI` to point
  at a remote server.
* Browse with:
  ```bash
  make mlflow-ui   # http://localhost:5000
  ```

---

## 5. Model Packaging & Reproducibility  *(Task 4 — 7 marks)*

* The best pipeline is serialised as `models/heart_disease_model.pkl` via
  `joblib`. Because preprocessing is part of the same `Pipeline`, inference
  needs nothing more than the `.pkl` and the request payload.
* MLflow also stores the model under each run's `model/` artifact, enabling
  `mlflow models serve` if desired.
* `requirements.txt` pins every dependency. `Dockerfile` builds and trains
  in an isolated `python:3.10-slim` container, guaranteeing identical
  artifacts across environments.

---

## 6. CI/CD & Automated Testing  *(Task 5 — 8 marks)*

### Tests (`tests/`)
* `test_data.py` — schema, stratified split, preprocessor output shape.
* `test_model.py` — pipeline shape, `predict_proba` sums to 1, beats
  majority-class baseline.
* `test_api.py` — FastAPI `TestClient` covers `/health`, `/predict`,
  `/predict/batch`, validation errors and the `/metrics` endpoint.
* Synthetic in-memory dataset (`conftest.py`) keeps tests **network-free**
  and deterministic.

### GitHub Actions (`.github/workflows/`)
* **`ci.yml`** — three jobs in series:
  1. `lint` (flake8 + black --check)
  2. `test` (pytest with coverage, uploads `coverage.xml`)
  3. `train` (downloads data, trains, evaluates, uploads model + metrics +
     plots + `mlruns/` as workflow artifacts; writes a JSON metrics summary
     to `$GITHUB_STEP_SUMMARY`)
* **`cd.yml`** — builds the Docker image, runs a smoke test (`/health` +
  `/predict`), and pushes tagged images to **GHCR** on `main`/tag pushes.
* The pipeline **fails fast**: any lint, test, or training error halts the
  workflow with clear logs.

---

## 7. Containerisation  *(Task 6 — 5 marks)*

`Dockerfile` is a **multi-stage** build:
1. **Builder stage** installs dependencies, downloads data, trains model.
2. **Runtime stage** is a minimal `python:3.10-slim` image running as a
   non-root user, with a `HEALTHCHECK` and `gunicorn + uvicorn` workers.

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8080:8080 heart-disease-api:latest

# Sample request
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
# -> {"prediction":1,"label":"disease","confidence":0.83,"probabilities":{...}}
```

The CD workflow performs exactly this build + smoke test on every push.

---

## 8. Production Deployment (Kubernetes)  *(Task 7 — 7 marks)*

`deployment/k8s/` and `deployment/helm/` provide two equivalent paths.

### Plain manifests
| Manifest | Purpose |
|----------|---------|
| `namespace.yaml`  | `heart-disease` namespace |
| `configmap.yaml`  | env vars (port, model path, log level) |
| `deployment.yaml` | 2 replicas, probes, resource limits, non-root securityContext |
| `service.yaml`    | `LoadBalancer` (port 80 → 8080) |
| `ingress.yaml`    | NGINX ingress on `heart-disease.local` |
| `hpa.yaml`        | autoscale 2–10 pods on CPU 70% |

```bash
# Minikube
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
kubectl apply -f deployment/k8s/
kubectl -n heart-disease rollout status deploy/heart-disease-api
kubectl -n heart-disease port-forward svc/heart-disease-api 8080:80
```

### Helm chart
```bash
helm install hd ./deployment/helm \
  --set image.repository=ghcr.io/<owner>/heart-disease-api \
  --set image.tag=latest
```

Deployment screenshots live under `reports/screenshots/` (to be captured
during the demo recording).

---

## 9. Monitoring & Logging  *(Task 8 — 3 marks)*

* **Logs:** every API request emits a JSON line with `request_id`, method,
  path, latency, and prediction outcome (`src/api/logging_config.py`,
  `src/api/app.py` middleware). Container runtimes collect these from
  stdout for free.
* **Metrics:** `prometheus-fastapi-instrumentator` exposes `/metrics` with
  default request/latency/status histograms.
* **Stack:** `docker-compose up -d` brings up API + Prometheus + Grafana
  with a pre-provisioned **"Heart Disease API"** dashboard
  (`monitoring/grafana/dashboards/heart_disease_api.json`):
  - requests per second
  - error rate (5xx)
  - p95 latency
  - requests by endpoint
  - status-code breakdown

---

## 10. Architecture Diagram

```
   ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
   │  UCI Data  │──▶│  Preprocess  │──▶│  Train+CV    │──▶│  MLflow     │
   │  Download  │   │ (ColumnTrans)│   │  LR + RF     │   │  Tracking   │
   └────────────┘   └──────────────┘   └──────┬───────┘   └─────────────┘
                                              ▼
                                     ┌──────────────────┐
                                     │ joblib Pipeline  │
                                     │  (model.pkl)     │
                                     └────────┬─────────┘
                                              │
            ┌─────────────────────────────────┼─────────────────────────────┐
            ▼                                 ▼                             ▼
   ┌────────────────┐               ┌────────────────┐           ┌────────────────────┐
   │ GitHub Actions │               │  FastAPI app   │           │  Kubernetes (GKE / │
   │  CI: lint+test │               │ /predict +     │           │  Minikube) Deploy  │
   │  CD: build img │               │ /metrics       │           │  + Service + HPA   │
   └────────────────┘               └────────┬───────┘           └─────────┬──────────┘
                                             ▼                             ▼
                                     ┌────────────────┐           ┌────────────────────┐
                                     │  Docker image  │──────────▶│ Prometheus+Grafana │
                                     │   (GHCR)       │           │  monitoring stack  │
                                     └────────────────┘           └────────────────────┘
```

---

## 11. Deliverables Checklist

| # | Deliverable | Location |
|---|-------------|----------|
| ✅ | Code, Dockerfile, requirements.txt | repo root |
| ✅ | Dataset download script | `src/data/download.py` |
| ✅ | EDA notebook + train/eval scripts | `notebooks/`, `src/models/` |
| ✅ | Unit tests | `tests/` |
| ✅ | GitHub Actions workflows | `.github/workflows/{ci,cd}.yml` |
| ✅ | K8s manifests + Helm chart | `deployment/{k8s,helm}/` |
| ✅ | Monitoring stack | `monitoring/`, `docker-compose.yml` |
| ✅ | Final report (this file) | `REPORT.md` |
| ⏳ | Screenshots folder | `reports/screenshots/` (record during demo) |
| ⏳ | Demo video | record locally; link in repo README |
| ⏳ | Deployed API URL | populate after deploying to your cluster |

---

## 12. Repository Link

> Push this folder to GitHub and paste the URL here once available.
> Example: `https://github.com/<you>/heart-disease-mlops`
