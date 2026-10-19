#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PILOT="datasets/share_swebench_verified_replacement_xarray4075"
WORKFLOWS="datasets/share_swebench_verified_staging/workflows.jsonl"
DATASET="/tmp/swebench_verified.parquet"
INSTANCE="pydata__xarray-4075"
MODEL="gpt-5.4-mini"
LOG_FILE="logs/share-swebench-replacement-xarray4075.log"
PID_FILE="logs/share-swebench-replacement-xarray4075.pid"
BRANCH_ROOT="${PILOT}/observed_branches/${INSTANCE}"
GENERATION="${PILOT}/generated_ai/${INSTANCE}/gpt-5.4-mini-r1"

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p logs
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}")"
    if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
      echo "Xarray replacement worker is already running as PID ${OLD_PID}" >&2
      exit 1
    fi
  fi
  nohup "${BASH_SOURCE[0]}" --worker >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > "${PID_FILE}"
  echo "Started Xarray replacement as PID ${PID}"
  echo "Log: ${LOG_FILE}"
  echo "Follow with: tail -f ${LOG_FILE}"
  exit 0
fi

for required in "${DATASET}" "${WORKFLOWS}" "${PILOT}/execution_specs.jsonl" .env; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done
docker info >/dev/null
echo "[$(date --iso-8601=seconds)] ${INSTANCE}: replacement begin"

if [[ ! -f "${PILOT}/image_provenance/${INSTANCE}.json" ]]; then
  python scripts/pull_share_swebench_image.py --pilot "${PILOT}" --instance "${INSTANCE}"
fi

if [[ ! -f "${BRANCH_ROOT}/human/artifact.json" ]]; then
  if [[ ! -f "results/share_swebench_human_xarray4075_r1/gold.share-human-xarray4075-r1.json" ]]; then
    .venv-swebench/bin/swebench eval "${DATASET}" \
      --gold \
      --instance "${INSTANCE}" \
      --run-id share-human-xarray4075-r1 \
      --workers 1 \
      --timeout 1800 \
      --report-dir results/share_swebench_human_xarray4075_r1
  fi
  python scripts/archive_share_swebench_human_result.py \
    --pilot "${PILOT}" \
    --instance "${INSTANCE}" \
    --run-id share-human-xarray4075-r1 \
    --report-dir results/share_swebench_human_xarray4075_r1 \
    --out "${BRANCH_ROOT}/human"
fi

if [[ ! -f "${BRANCH_ROOT}/stop/artifact.json" ]]; then
  if [[ ! -f "results/share_swebench_stop_xarray4075_r1/share-stop-no-solution-patch.share-stop-xarray4075-r1.binding.json" ]]; then
    .venv-swebench/bin/python scripts/run_share_swebench_stop.py \
      --dataset "${DATASET}" \
      --instance "${INSTANCE}" \
      --run-id share-stop-xarray4075-r1 \
      --report-dir results/share_swebench_stop_xarray4075_r1
  fi
  python scripts/archive_share_swebench_stop_result.py \
    --pilot "${PILOT}" \
    --instance "${INSTANCE}" \
    --run-id share-stop-xarray4075-r1 \
    --report-dir results/share_swebench_stop_xarray4075_r1 \
    --out "${BRANCH_ROOT}/stop"
fi

if [[ ! -f "${GENERATION}/artifact.json" ]]; then
  if [[ -e "${GENERATION}" ]]; then
    echo "Incomplete AI generation directory requires inspection: ${GENERATION}" >&2
    exit 1
  fi
  python scripts/generate_share_swebench_ai_patch.py \
    --pilot "${PILOT}" \
    --workflows "${WORKFLOWS}" \
    --instance "${INSTANCE}" \
    --profile primary \
    --env .env \
    --out "${GENERATION}"
fi
if [[ -e "${GENERATION}/QUARANTINED.json" ]]; then
  echo "Refusing quarantined AI generation: ${GENERATION}" >&2
  exit 1
fi

if [[ ! -f "${BRANCH_ROOT}/ai/artifact.json" ]]; then
  if [[ ! -f "results/share_swebench_ai_xarray4075_gpt54mini_r1/${MODEL}.share-ai-xarray4075-gpt54mini-r1.json" ]]; then
    .venv-swebench/bin/swebench eval "${DATASET}" \
      --predictions "${GENERATION}/prediction.jsonl" \
      --instance "${INSTANCE}" \
      --run-id share-ai-xarray4075-gpt54mini-r1 \
      --workers 1 \
      --timeout 1800 \
      --report-dir results/share_swebench_ai_xarray4075_gpt54mini_r1
  fi
  python scripts/archive_share_swebench_ai_result.py \
    --pilot "${PILOT}" \
    --instance "${INSTANCE}" \
    --generation "${GENERATION}" \
    --model "${MODEL}" \
    --run-id share-ai-xarray4075-gpt54mini-r1 \
    --report-dir results/share_swebench_ai_xarray4075_gpt54mini_r1 \
    --out "${BRANCH_ROOT}/ai"
fi

if [[ ! -f "${BRANCH_ROOT}/triplet_manifest.json" ]]; then
  python scripts/finalize_share_swebench_triplet.py \
    --pilot "${PILOT}" \
    --instance "${INSTANCE}" \
    --out "${BRANCH_ROOT}/triplet_manifest.json"
fi

echo "[$(date --iso-8601=seconds)] ${INSTANCE}: replacement complete"
