# Monitoring & Logging

This project ships with two complementary observability layers:

## 1. Structured JSON logs

Every API request is logged as a single JSON line via `python-json-logger`,
including `request_id`, `method`, `path`, `client`, `duration_ms`, plus
prediction outcome / confidence on `/predict` calls. Configure the formatter
in `src/api/logging_config.py`. Logs go to **stdout** so any container
runtime (Docker, Kubernetes, Cloud Run) collects them automatically.

## 2. Prometheus + Grafana

The API auto-exposes a Prometheus-formatted endpoint at **`/metrics`** thanks
to `prometheus-fastapi-instrumentator`. The default counters cover request
rate, latency histograms, status codes, and in-flight requests.

On top of those, `src/api/app.py` publishes two **model-level** metrics:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `predictions_total` | Counter | `label` (`disease` / `no_disease`) | Number of `/predict` responses by predicted class |
| `prediction_confidence` | Histogram | — | Distribution of the model's max class probability |

### Local stack (docker-compose)

```bash
cd heart_disease_mlops
docker compose up -d --build

# Services
#   API         http://localhost:8080  (docs: /docs)
#   Prometheus  http://localhost:9090
#   Grafana     http://localhost:3000  (admin / admin)
```

### Generate some traffic

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}' > /dev/null
done
```

### Dashboard

A pre-provisioned dashboard "**Heart Disease API**" is auto-loaded into
Grafana from `monitoring/grafana/dashboards/heart_disease_api.json` with 9 panels:

1. Requests / sec (stat, `reqps`)
2. Error rate 4xx (stat, `reqps`, thresholds)
3. Error rate 5xx (stat, `reqps`, thresholds)
4. Predictions total (stat)
5. Latency p50 / p95 (timeseries, seconds)
6. Requests by endpoint (timeseries, by `handler`)
7. Status codes (timeseries, by `status`)
8. Predictions by class — rps (timeseries, by `label`)
9. Prediction confidence p50 / p95 (timeseries, percent unit)

### Kubernetes

Deployment pods are annotated with `prometheus.io/scrape: "true"`, so a
Prometheus instance configured for pod-discovery (e.g. kube-prometheus-stack)
will pick them up automatically. Enable the Helm `serviceMonitor` flag if
the prometheus-operator is installed.
