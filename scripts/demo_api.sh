#!/usr/bin/env bash
# Quick demo of all API endpoints. Assumes the server is listening on :8080.
set -uo pipefail
URL="${URL:-http://127.0.0.1:8080}"

banner() { printf '\n============================================\n %s\n============================================\n' "$1"; }

banner "GET /"
curl -s "$URL/" | python3 -m json.tool

banner "GET /health"
curl -s "$URL/health" | python3 -m json.tool

banner "POST /predict   (high-risk patient)"
curl -s -X POST "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":3,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' \
  | python3 -m json.tool

banner "POST /predict   (low-risk patient)"
curl -s -X POST "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":41,"sex":0,"cp":1,"trestbps":130,"chol":204,"fbs":0,"restecg":2,"thalach":172,"exang":0,"oldpeak":1.4,"slope":1,"ca":0,"thal":2}' \
  | python3 -m json.tool

banner "POST /predict/batch  (3 patients)"
curl -s -X POST "$URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"instances":[
    {"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1},
    {"age":67,"sex":1,"cp":3,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3},
    {"age":41,"sex":0,"cp":1,"trestbps":130,"chol":204,"fbs":0,"restecg":2,"thalach":172,"exang":0,"oldpeak":1.4,"slope":1,"ca":0,"thal":2}
  ]}' | python3 -m json.tool

banner "POST /predict   (validation error: age=-5)"
code=$(curl -s -o /tmp/err.json -w "%{http_code}" -X POST "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":-5,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}')
echo "[HTTP $code]"; python3 -m json.tool < /tmp/err.json

banner "GET /metrics  (Prometheus, sample lines)"
curl -s "$URL/metrics" | grep -E '^http_requests_total|^http_request_duration_seconds_count|^http_request_size_bytes_count' | head -15
