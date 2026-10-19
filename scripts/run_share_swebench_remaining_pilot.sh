#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PILOT="datasets/share_swebench_verified_pilot5"
WORKFLOWS="datasets/share_swebench_verified_staging/workflows.jsonl"
DATASET="/tmp/swebench_verified.parquet"
MODEL="gpt-5.4-mini"
LOG_FILE="logs/share-swebench-remaining-pilot.log"
PID_FILE="logs/share-swebench-remaining-pilot.pid"

ENTRIES=(
  "django__django-10097:django10097"
  "matplotlib__matplotlib-13989:matplotlib13989"
  "pallets__flask-5014:flask5014"
  "psf__requests-1142:requests1142"
)

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p logs
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}")"
    if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
      echo "Pilot worker is already running as PID ${OLD_PID}" >&2
      exit 1
    fi
  fi
  nohup "${BASH_SOURCE[0]}" --worker >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > "${PID_FILE}"
  echo "Started remaining pilot as PID ${PID}"
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

echo "[$(date --iso-8601=seconds)] Starting four-instance pilot worker"

for entry in "${ENTRIES[@]}"; do
  INSTANCE="${entry%%:*}"
  SUFFIX="${entry##*:}"
  BRANCH_ROOT="${PILOT}/observed_branches/${INSTANCE}"
  PROVENANCE="${PILOT}/image_provenance/${INSTANCE}.json"
  EXCLUSION="${PILOT}/exclusions/${INSTANCE}.json"
  HUMAN_RUN="share-human-${SUFFIX}-r1"
  HUMAN_RESULTS="results/share_swebench_human_${SUFFIX}_r1"
  STOP_RUN="share-stop-${SUFFIX}-r1"
  STOP_RESULTS="results/share_swebench_stop_${SUFFIX}_r1"
  GENERATION="${PILOT}/generated_ai/${INSTANCE}/gpt-5.4-mini-r1"
  AI_RUN="share-ai-${SUFFIX}-gpt54mini-r1"
  AI_RESULTS="results/share_swebench_ai_${SUFFIX}_gpt54mini_r1"

  echo "[$(date --iso-8601=seconds)] ${INSTANCE}: begin"

  if [[ -f "${EXCLUSION}" ]]; then
    echo "${INSTANCE}: excluded by ${EXCLUSION}; skipping"
    continue
  fi

  if [[ ! -f "${PROVENANCE}" ]]; then
    python scripts/pull_share_swebench_image.py --pilot "${PILOT}" --instance "${INSTANCE}"
  else
    echo "${INSTANCE}: image provenance already archived"
  fi

  if [[ ! -f "${BRANCH_ROOT}/human/artifact.json" ]]; then
    if [[ ! -f "${HUMAN_RESULTS}/gold.${HUMAN_RUN}.json" ]]; then
      .venv-swebench/bin/swebench eval "${DATASET}" \
        --gold \
        --instance "${INSTANCE}" \
        --run-id "${HUMAN_RUN}" \
        --workers 1 \
        --timeout 1800 \
        --report-dir "${HUMAN_RESULTS}"
    fi
    python scripts/archive_share_swebench_human_result.py \
      --pilot "${PILOT}" \
      --instance "${INSTANCE}" \
      --run-id "${HUMAN_RUN}" \
      --report-dir "${HUMAN_RESULTS}" \
      --out "${BRANCH_ROOT}/human"
  else
    echo "${INSTANCE}: Human branch already archived"
  fi

  if [[ ! -f "${BRANCH_ROOT}/stop/artifact.json" ]]; then
    if [[ ! -f "${STOP_RESULTS}/share-stop-no-solution-patch.${STOP_RUN}.binding.json" ]]; then
      .venv-swebench/bin/python scripts/run_share_swebench_stop.py \
        --dataset "${DATASET}" \
        --instance "${INSTANCE}" \
        --run-id "${STOP_RUN}" \
        --report-dir "${STOP_RESULTS}"
    fi
    python scripts/archive_share_swebench_stop_result.py \
      --pilot "${PILOT}" \
      --instance "${INSTANCE}" \
      --run-id "${STOP_RUN}" \
      --report-dir "${STOP_RESULTS}" \
      --out "${BRANCH_ROOT}/stop"
  else
    echo "${INSTANCE}: Stop branch already archived"
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
    if [[ ! -f "${AI_RESULTS}/${MODEL}.${AI_RUN}.json" ]]; then
      .venv-swebench/bin/swebench eval "${DATASET}" \
        --predictions "${GENERATION}/prediction.jsonl" \
        --instance "${INSTANCE}" \
        --run-id "${AI_RUN}" \
        --workers 1 \
        --timeout 1800 \
        --report-dir "${AI_RESULTS}"
    fi
    python scripts/archive_share_swebench_ai_result.py \
      --pilot "${PILOT}" \
      --instance "${INSTANCE}" \
      --generation "${GENERATION}" \
      --model "${MODEL}" \
      --run-id "${AI_RUN}" \
      --report-dir "${AI_RESULTS}" \
      --out "${BRANCH_ROOT}/ai"
  else
    echo "${INSTANCE}: AI branch already archived"
  fi

  if [[ ! -f "${BRANCH_ROOT}/triplet_manifest.json" ]]; then
    python scripts/finalize_share_swebench_triplet.py \
      --pilot "${PILOT}" \
      --instance "${INSTANCE}" \
      --out "${BRANCH_ROOT}/triplet_manifest.json"
  fi
  echo "[$(date --iso-8601=seconds)] ${INSTANCE}: complete"
done

echo "[$(date --iso-8601=seconds)] All remaining pilot instances complete"
