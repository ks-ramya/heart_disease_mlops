# Kubernetes Deployment

Manifests in this folder deploy the Heart Disease API to any Kubernetes cluster
(Minikube, Docker Desktop, GKE, EKS, AKS).

## Files
| File | Purpose |
|------|---------|
| `namespace.yaml`  | Creates the `heart-disease` namespace |
| `configmap.yaml`  | Non-secret env (port, model path, log level) |
| `deployment.yaml` | 2-replica Deployment with probes, limits, security context |
| `service.yaml`    | `LoadBalancer` service exposing port 80 → 8080 |
| `ingress.yaml`    | Optional NGINX Ingress on host `heart-disease.local` |
| `hpa.yaml`        | Horizontal Pod Autoscaler (CPU 70% / Mem 80%) |

## Quick Start (Minikube)

```bash
# 1. Build the image inside Minikube's Docker daemon
eval $(minikube docker-env)
docker build -t heart-disease-api:latest ../..

# 2. Apply manifests
kubectl apply -f .

# 3. Wait for rollout
kubectl -n heart-disease rollout status deploy/heart-disease-api

# 4. Access the service
minikube service heart-disease-api -n heart-disease --url
# OR
kubectl -n heart-disease port-forward svc/heart-disease-api 8080:80
curl http://localhost:8080/health
```

## Quick Start (GKE / EKS / AKS)

```bash
# 1. Push image (example with GHCR)
docker tag heart-disease-api:latest ghcr.io/<owner>/heart-disease-api:latest
docker push ghcr.io/<owner>/heart-disease-api:latest

# 2. Update deployment.yaml `image:` to that path

# 3. Apply
kubectl apply -f .

# 4. Get external IP (LoadBalancer)
kubectl -n heart-disease get svc heart-disease-api -w
```

## Tear-down

```bash
kubectl delete -f .
# or
kubectl delete namespace heart-disease
```
