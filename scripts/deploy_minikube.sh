#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Deploy Heart Disease API to Minikube end-to-end.
#
# What it does:
#   1. Starts minikube if not running
#   2. Enables ingress + metrics-server addons (idempotent)
#   3. Builds the Docker image *inside* minikube's docker daemon
#      (so the deployment can `imagePullPolicy: IfNotPresent` without a registry)
#   4. Applies all manifests in deployment/k8s/
#   5. Waits for the rollout to become Ready
#   6. Prints the service URL and opens the UI in a browser
#
# Usage:
#   bash scripts/deploy_minikube.sh           # build + deploy + open UI
#   bash scripts/deploy_minikube.sh --no-open # don't open browser
#   bash scripts/deploy_minikube.sh --teardown
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-minikube}"
NAMESPACE="heart-disease"
IMAGE="heart-disease-api:latest"
DEPLOY="heart-disease-api"
SVC="heart-disease-api"
OPEN_BROWSER=1

cd "$(dirname "$0")/.."

log()  { printf "\033[1;36m[deploy]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
    --teardown)
      log "Tearing down namespace $NAMESPACE…"
      kubectl delete ns "$NAMESPACE" --ignore-not-found
      log "Done."
      exit 0
      ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) die "Unknown arg: $arg" ;;
  esac
done

command -v minikube >/dev/null || die "minikube not found in PATH"
command -v kubectl  >/dev/null || die "kubectl not found in PATH"
command -v docker   >/dev/null || die "docker not found in PATH"

# 1. Start cluster
if ! minikube -p "$PROFILE" status --format='{{.Host}}' 2>/dev/null | grep -q Running; then
  log "Starting minikube profile '$PROFILE' (this can take ~1 minute)…"
  minikube start -p "$PROFILE" --memory=4096 --cpus=2
else
  log "Minikube profile '$PROFILE' already running."
fi

# 2. Addons
log "Ensuring addons (metrics-server, ingress) are enabled…"
minikube -p "$PROFILE" addons enable metrics-server >/dev/null 2>&1 || warn "metrics-server enable failed (HPA may not work)"
minikube -p "$PROFILE" addons enable ingress        >/dev/null 2>&1 || warn "ingress enable failed (ingress.yaml will be skipped)"

# 3. Build image inside minikube's docker
log "Building '$IMAGE' inside minikube's docker daemon…"
eval "$(minikube -p "$PROFILE" docker-env)"
DOCKER_BUILDKIT=1 docker build -t "$IMAGE" .
log "Image built. Sizes:"
docker image ls "$IMAGE" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}'

# 4. Apply manifests
log "Applying manifests in deployment/k8s/…"
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/hpa.yaml          || warn "HPA apply failed (likely metrics-server not ready yet)"
kubectl apply -f deployment/k8s/ingress.yaml      || warn "Ingress apply failed (addon may be missing)"

# 5. Wait for rollout
log "Waiting for deployment/$DEPLOY to be Ready (timeout 180s)…"
kubectl -n "$NAMESPACE" rollout status "deploy/$DEPLOY" --timeout=180s

# 6. Show URL and open
log "Pods:"
kubectl -n "$NAMESPACE" get pods -o wide
log "Service:"
kubectl -n "$NAMESPACE" get svc "$SVC"

URL=$(minikube -p "$PROFILE" service "$SVC" -n "$NAMESPACE" --url | head -1)
log "🌐 Service URL: $URL"
log "    Health:    $URL/health"
log "    UI:        $URL/ui/"
log "    Predict:   $URL/predict (POST JSON)"

# Smoke test
log "Smoke-testing /health…"
if curl -fsS --max-time 10 "$URL/health" >/tmp/hd_health.json 2>&1; then
  cat /tmp/hd_health.json; echo
  log "✅ /health OK"
else
  warn "/health did not respond yet; pod may still be warming up."
fi

if [ "$OPEN_BROWSER" -eq 1 ]; then
  log "Opening UI in your default browser…"
  ( open "$URL/ui/" 2>/dev/null || xdg-open "$URL/ui/" 2>/dev/null || true ) &
fi

log "Done. To tear down: bash scripts/deploy_minikube.sh --teardown"
