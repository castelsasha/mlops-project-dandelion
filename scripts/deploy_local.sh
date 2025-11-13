#!/usr/bin/env bash
set -euo pipefail

# --- Helper pretty print ---
hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '-'; }
msg() { echo -e "\n$1"; }

# --- Cleanup any previous port-forward on 8000 ---
cleanup_pf() {
  pkill -f "kubectl.*port-forward.*dandelion-api.*8000:8000" >/dev/null 2>&1 || true
}
trap cleanup_pf EXIT

# --- 0) Ensure required CLIs ---
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl not found"; exit 1; }

# --- 1) Build local API image ---
hr; msg "▶ Building API image: dandelion-api:latest"
docker build -f docker/api.Dockerfile -t dandelion-api:latest .

# --- 2) Apply k8s manifests (idempotent) ---
hr; msg "▶ Applying Kubernetes manifests (namespace, secret, configmap, deployment, service)…"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml || true
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# --- 3) Rollout restart & wait ---
hr; msg "▶ Rolling restart + waiting for readiness…"
kubectl -n dandelion rollout restart deployment dandelion-api
kubectl -n dandelion rollout status  deployment dandelion-api --timeout=180s

# --- 4) Show pods ---
hr; msg "▶ Pods:"
kubectl -n dandelion get pods -l app=dandelion-api -o wide

# --- 5) Port-forward the Service to localhost:8000 ---
hr; msg "▶ Port-forward svc/dandelion-api → localhost:8000"
cleanup_pf
kubectl -n dandelion port-forward svc/dandelion-api 8000:8000 >/dev/null 2>&1 &
PF_PID=$!

# --- 6) Wait for /health to respond ---
hr; msg "▶ Healthcheck"
set +e
for i in {1..30}; do
  RESP=$(curl -sf http://127.0.0.1:8000/health) && OK=1 || OK=0
  if [ "$OK" -eq 1 ]; then
    echo "$RESP" | python -m json.tool 2>/dev/null || echo "$RESP"
    break
  fi
  sleep 1
done
set -e
if [ "${OK:-0}" -ne 1 ]; then
  echo "❌ Health endpoint did not respond. Recent logs:"
  kubectl -n dandelion logs -l app=dandelion-api --tail=100
  exit 1
fi

# --- 7) Optional sample /predict (if sample images exist) ---
hr; msg "▶ Sample /predict (optional)"
if [ -f "data/raw/dandelion_00000010.jpg" ]; then
  echo "→ dandelion sample:"
  curl -s -X POST "http://127.0.0.1:8000/predict" \
    -F "file=@data/raw/dandelion_00000010.jpg" | python -m json.tool || true
fi
if [ -f "data/raw/grass_00000010.jpg" ]; then
  echo "→ grass sample:"
  curl -s -X POST "http://127.0.0.1:8000/predict" \
    -F "file=@data/raw/grass_00000010.jpg" | python -m json.tool || true
fi

hr
echo "✅ Done. API available at http://127.0.0.1:8000"
echo "   Swagger UI: http://127.0.0.1:8000/docs"