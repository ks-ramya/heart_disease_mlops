# Heart Disease MLOps Pipeline

[![CI - Lint, Test, Train](https://github.com/ks-ramya/heart_disease_mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ks-ramya/heart_disease_mlops/actions/workflows/ci.yml)
[![CD - Build and Push Docker Image](https://github.com/ks-ramya/heart_disease_mlops/actions/workflows/cd.yml/badge.svg?branch=main)](https://github.com/ks-ramya/heart_disease_mlops/actions/workflows/cd.yml)
[![Container Image](https://img.shields.io/badge/ghcr.io-heart--disease--api-blue?logo=docker)](https://github.com/ks-ramya/heart_disease_mlops/pkgs/container/heart-disease-api)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end MLOps solution for predicting heart disease risk using the **UCI Heart Disease (Cleveland)** dataset. The project covers the full lifecycle: data acquisition, EDA, training, experiment tracking, containerised serving, CI/CD, Kubernetes deployment, and monitoring.

> Course: **MLOps (S2-25_AMLCSZG523)** – Assignment I

## Pull the prebuilt image
```bash
docker pull ghcr.io/ks-ramya/heart-disease-api:latest
docker run -d -p 8080:8080 ghcr.io/ks-ramya/heart-disease-api:latest
open http://localhost:8080/ui/    # browser UI
curl http://localhost:8080/health  # API health
```

## Web UI

A built-in single-page UI is served at **`/ui/`** (also opens automatically when a browser hits `/`):

* Form for all 13 features with sensible defaults & validation
* One-click **High-risk** / **Low-risk** sample patients
* Live result card with prediction badge + probability bars
* Health pill in the top-right tracks `/health`

Try it locally with `make serve` then open <http://localhost:8080/ui/>.

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data + run EDA
python -m src.data.download
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb

# 3. Train (logs to MLflow + saves model)
python -m src.models.train

# 4. Serve API + UI locally
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
# UI:  http://localhost:8080/ui/
# API: http://localhost:8080/predict

# 5. Test endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## Deploy to Minikube

One command builds the image inside the Minikube docker daemon, applies all
manifests in `deployment/k8s/`, waits for the rollout, and opens the UI:

```bash
make minikube-up
# or, equivalently:
bash scripts/deploy_minikube.sh
```

Useful follow-ups:
```bash
make minikube-url     # print the NodePort URL
kubectl -n heart-disease get pods,svc,hpa
make minikube-down    # tear down the namespace
```

The script:
1. Starts Minikube if it is not running (4 GB / 2 CPU).
2. Enables the `metrics-server` (HPA) and `ingress` add-ons.
3. Builds `heart-disease-api:latest` directly inside Minikube's Docker (no registry round-trip).
4. Applies `namespace`, `configmap`, `deployment`, `service` (NodePort 30080), `hpa`, `ingress`.
5. Waits for the deployment to become Ready, smoke-tests `/health`, and opens `/ui/` in your browser.

See [REPORT.md](REPORT.md) for the full report and [docs](#docs) section below for deeper guides.

## Architecture

```
   ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
   │  UCI Data  │──▶│  Preprocess  │──▶│   Training   │──▶│   MLflow    │
   │  Download  │   │  (Pipeline)  │   │  LR  +  RF   │   │  Tracking   │
   └────────────┘   └──────────────┘   └──────┬───────┘   └─────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  joblib pipeline │
                                     │  (model.pkl)     │
                                     └────────┬─────────┘
                                              │
            ┌─────────────────────────────────┼─────────────────────────────┐
            ▼                                 ▼                             ▼
   ┌────────────────┐               ┌────────────────┐           ┌────────────────────┐
   │ GitHub Actions │               │  FastAPI app   │           │  Kubernetes (GKE / │
   │  CI: lint+test │               │ /predict +     │           │  Minikube) Deploy  │
   │  CD: build img │               │ /metrics       │           │  + Service + Ingr. │
   └────────────────┘               └────────┬───────┘           └─────────┬──────────┘
                                             ▼                             ▼
                                     ┌────────────────┐           ┌────────────────────┐
                                     │  Docker Image  │──────────▶│ Prometheus+Grafana │
                                     └────────────────┘           │  Monitoring stack  │
                                                                  └────────────────────┘
```

## Project Structure

```
heart_disease_mlops/
├── src/
│   ├── data/             # download.py, preprocess.py
│   ├── models/           # train.py, evaluate.py
│   └── api/              # FastAPI app + Pydantic schemas
├── notebooks/            # 01_eda.ipynb
├── tests/                # pytest unit tests
├── deployment/
│   ├── k8s/              # Kubernetes manifests
│   └── helm/             # Helm chart
├── monitoring/           # Prometheus + Grafana configs
├── .github/workflows/    # CI / CD pipelines
├── data/                 # raw + processed CSVs (generated)
├── models/               # Saved models (generated)
├── reports/figures/      # EDA plots (generated)
├── Dockerfile
├── docker-compose.yml    # API + Prometheus + Grafana
├── requirements.txt
├── Makefile
└── REPORT.md
```

## Docs

- **Setup & Reproducibility:** see "Quick Start" above and `requirements.txt`.
- **EDA:** `notebooks/01_eda.ipynb` — class balance, correlations, distributions.
- **Modelling choices:** `REPORT.md` § Modelling.
- **CI/CD:** `.github/workflows/ci.yml`, `.github/workflows/cd.yml`.
- **Deployment:** `deployment/k8s/README.md`, `deployment/helm/`.
- **Monitoring:** `monitoring/README.md`.

## License

Educational use only. Dataset © UCI Machine Learning Repository.
