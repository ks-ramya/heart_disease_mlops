#!/usr/bin/env bash
# One-off evidence-capture script for the Task-7 audit.
# Spins a port-forward to svc/heart-disease-api and curls /health, /predict,
# /metrics; writes the result to reports/screenshots/25_k8s_endpoints.txt.
set -u

OUT="reports/screenshots/25_k8s_endpoints.txt"

pkill -f "kubectl.*port-forward.*heart-disease" 2>/dev/null
sleep 1
kubectl -n heart-disease port-forward svc/heart-disease-api 18080:80 \
  > /tmp/pf.log 2>&1 &
PF=$!
sleep 4

{
  echo "# Heart-Disease-API in Minikube — verified live endpoints"
  echo "# Captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Source:   curl http://127.0.0.1:18080/* via"
  echo "#           kubectl -n heart-disease port-forward svc/heart-disease-api 18080:80"
  echo "# ============================================================"
  echo
  echo "--- GET /health ---"
  curl -sS -m 5 http://127.0.0.1:18080/health | python3 -m json.tool
  echo
  echo "--- POST /predict (high-risk patient) ---"
  curl -sS -m 8 -X POST http://127.0.0.1:18080/predict \
    -H "Content-Type: application/json" \
    -d '{"age":67,"sex":1,"cp":3,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' \
    | python3 -m json.tool
  echo
  echo "--- POST /predict (low-risk patient) ---"
  curl -sS -m 8 -X POST http://127.0.0.1:18080/predict \
    -H "Content-Type: application/json" \
    -d '{"age":45,"sex":0,"cp":1,"trestbps":120,"chol":200,"fbs":0,"restecg":0,"thalach":170,"exang":0,"oldpeak":0.5,"slope":1,"ca":0,"thal":2}' \
    | python3 -m json.tool
  echo
  echo "--- GET /metrics (first 12 lines) ---"
  curl -sS -m 5 http://127.0.0.1:18080/metrics | head -12
} > "$OUT" 2>&1

kill $PF 2>/dev/null
echo "wrote: $OUT ($(wc -l < $OUT) lines)"
