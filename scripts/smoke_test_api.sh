#!/usr/bin/env bash
# Smoke-test a running API instance:  ./scripts/smoke_test_api.sh [http://localhost:8000]
set -euo pipefail
URL=${1:-http://localhost:8000}
IMG=$(ls examples/images/*.jpeg | head -1)
echo "--- GET /ready";   curl -fsS "$URL/ready"; echo
echo "--- POST /predict ($IMG)"; curl -fsS -F "file=@${IMG};type=image/jpeg" "$URL/predict"; echo
echo "--- POST /predict with a text file (expect 415)"
curl -sS -o /dev/stderr -w "HTTP %{http_code}\n" -F "file=@README.md;type=text/plain" "$URL/predict"
echo "--- POST /predict/batch"
args=(); for f in $(ls examples/images/*.jpeg | head -3); do args+=(-F "files=@${f};type=image/jpeg"); done
curl -fsS "${args[@]}" "$URL/predict/batch"; echo
