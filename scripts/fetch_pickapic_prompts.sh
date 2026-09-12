#!/usr/bin/env bash
# Fetch the Pick-a-Pic prompt list used for the paper's C_pix@12 evaluation.
#
# The reported color-coverage numbers sample 50 prompts (seed 0) from the
# 2,048-line Pick-a-Pic prompt file distributed with the MIT-licensed Flow-GRPO
# repository (dataset/pickscore/test.txt). The file is user-written text from
# Pick-a-Pic and contains explicit and otherwise objectionable prompts, so it is
# not redistributed here. The download is verified against the checksum of the
# copy used for the paper.
set -euo pipefail

DEST="${1:-data/pickapic/test.txt}"
URL="https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/test.txt"
EXPECTED_SHA256="d75c025fa451866e70f7e36243f7cdd93095523cb51f5f9651f2ce3236d20262"

mkdir -p "$(dirname "${DEST}")"
curl -fsSL "${URL}" -o "${DEST}"
ACTUAL_SHA256="$(sha256sum "${DEST}" | cut -d' ' -f1)"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "checksum mismatch for ${DEST}: ${ACTUAL_SHA256}" >&2
  echo "expected ${EXPECTED_SHA256}; the upstream file may have changed" >&2
  exit 1
fi
echo "saved $(wc -l < "${DEST}") prompts to ${DEST} (sha256 verified)"
