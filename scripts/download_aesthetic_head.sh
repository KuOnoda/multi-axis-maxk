#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PATH="${AESTHETIC_MODEL_PATH:-${REPO_ROOT}/models/sac+logos+ava1-l14-linearMSE.pth}"
URL="https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/fe88a163f4661b4ddabba0751ff645e2e620746e/sac+logos+ava1-l14-linearMSE.pth"

mkdir -p "$(dirname "${OUT_PATH}")"
curl --fail --location --retry 3 "${URL}" --output "${OUT_PATH}"
echo "Downloaded aesthetic predictor head to ${OUT_PATH}"
