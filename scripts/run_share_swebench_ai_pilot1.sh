#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PREDICTION="datasets/share_swebench_verified_pilot5/generated_ai/astropy__astropy-14309/gpt-5.4-mini-r2/prediction.jsonl"
QUARANTINE="datasets/share_swebench_verified_pilot5/generated_ai/astropy__astropy-14309/gpt-5.4-mini-r2/QUARANTINED.json"
LOG_FILE="logs/share-ai-eval-astropy-gpt54mini-r2.log"
PID_FILE="logs/share-ai-eval-astropy-gpt54mini-r2.pid"

if [[ ! -f "${PREDICTION}" ]]; then
  echo "Missing prediction file: ${PREDICTION}" >&2
  exit 1
fi
if [[ -e "${QUARANTINE}" ]]; then
  echo "Refusing to evaluate quarantined generation: ${QUARANTINE}" >&2
  exit 1
fi

mkdir -p logs
nohup .venv-swebench/bin/swebench eval \
  /tmp/swebench_verified.parquet \
  --predictions "${PREDICTION}" \
  --instance astropy__astropy-14309 \
  --run-id share-ai-pilot1-gpt54mini-r2 \
  --workers 1 \
  --timeout 1800 \
  --report-dir results/share_swebench_ai_pilot1_gpt54mini_r2 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Started PID ${PID}"
echo "Log: ${LOG_FILE}"
echo "Follow with: tail -f ${LOG_FILE}"
