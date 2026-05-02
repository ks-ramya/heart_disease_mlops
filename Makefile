# Heart Disease MLOps - Make targets
# Usage: make <target>

PY ?= python
IMAGE ?= heart-disease-api
TAG   ?= latest
PORT  ?= 8080

.PHONY: help install data train evaluate test lint format \
        serve docker-build docker-run docker-stop \
        mlflow-ui clean compose-up compose-down k8s-apply k8s-delete \
        minikube-up minikube-down minikube-url

help:
	@echo "Targets:"
	@echo "  install       Install Python dependencies"
	@echo "  data          Download dataset to data/raw/"
	@echo "  train         Train models + log to MLflow + save best model"
	@echo "  evaluate      Evaluate saved model on test set"
	@echo "  test          Run pytest"
	@echo "  lint          Run flake8"
	@echo "  format        Format with black"
	@echo "  serve         Run FastAPI locally on :$(PORT)"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run container, mapping port :$(PORT)"
	@echo "  docker-stop   Stop running container"
	@echo "  compose-up    Start API + Prometheus + Grafana"
	@echo "  compose-down  Stop the stack"
	@echo "  k8s-apply     Apply Kubernetes manifests"
	@echo "  k8s-delete    Delete Kubernetes resources"
	@echo "  minikube-up   End-to-end deploy to Minikube (build + apply + open UI)"
	@echo "  minikube-down Tear down the heart-disease namespace from Minikube"
	@echo "  minikube-url  Print the Minikube service URL"
	@echo "  mlflow-ui     Launch MLflow UI on :5000"
	@echo "  clean         Remove generated artifacts"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data.download

train: data
	$(PY) -m src.models.train

evaluate:
	$(PY) -m src.models.evaluate

test:
	$(PY) -m pytest tests/ -v --tb=short

lint:
	$(PY) -m flake8 src tests --max-line-length=110 --extend-ignore=E203,W503

format:
	$(PY) -m black src tests

serve:
	$(PY) -m uvicorn src.api.app:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -d --name $(IMAGE) -p $(PORT):8080 $(IMAGE):$(TAG)

docker-stop:
	docker stop $(IMAGE) || true

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down -v

k8s-apply:
	kubectl apply -f deployment/k8s/

k8s-delete:
	kubectl delete -f deployment/k8s/ || true

minikube-up:
	bash scripts/deploy_minikube.sh

minikube-down:
	bash scripts/deploy_minikube.sh --teardown

minikube-url:
	@minikube service heart-disease-api -n heart-disease --url

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --port 5000

clean:
	rm -rf mlruns/ mlartifacts/ models/*.pkl models/*.joblib \
	       data/processed/* reports/figures/*.png \
	       __pycache__ .pytest_cache .coverage
