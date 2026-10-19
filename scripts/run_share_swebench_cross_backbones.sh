#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET="/tmp/swebench_verified.parquet"
WORKFLOWS="datasets/share_swebench_verified_staging/workflows.jsonl"
LOG_FILE="logs/share-swebench-cross-backbones.log"
PID_FILE="logs/share-swebench-cross-backbones.pid"
PREFLIGHT="logs/share-swebench-cross-backbones-profiles.json"
PROFILES=("backbone_2" "backbone_3")
MAX_GENERATION_ATTEMPTS=8
ENTRIES=(
  "datasets/share_swebench_verified_pilot5:astropy__astropy-14309:astropy14309"
  "datasets/share_swebench_verified_pilot5:matplotlib__matplotlib-13989:matplotlib13989"
  "datasets/share_swebench_verified_pilot5:pallets__flask-5014:flask5014"
  "datasets/share_swebench_verified_pilot5:psf__requests-1142:requests1142"
  "datasets/share_swebench_verified_replacement_xarray4075:pydata__xarray-4075:xarray4075"
)

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p logs
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}")"
    if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
      echo "Cross-backbone worker is already running as PID ${OLD_PID}" >&2
      exit 1
    fi
  fi
  nohup "${BASH_SOURCE[0]}" --worker >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > "${PID_FILE}"
  echo "Started cross-backbone collection as PID ${PID}"
  echo "Log: ${LOG_FILE}"
  echo "Follow with: tail -f ${LOG_FILE}"
  exit 0
fi

for required in "${DATASET}" "${WORKFLOWS}" .env; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done
docker info >/dev/null
python scripts/check_share_llm_profiles.py --env .env --out "${PREFLIGHT}"

echo "[$(date --iso-8601=seconds)] Starting same-task cross-backbone collection"
for PROFILE in "${PROFILES[@]}"; do
  MODEL="$(jq -r --arg profile "${PROFILE}" '.profiles[] | select(.name == $profile) | .model' "${PREFLIGHT}")"
  if [[ -z "${MODEL}" || "${MODEL}" == "null" || ! "${MODEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Profile ${PROFILE} has a missing or path-unsafe model identifier: ${MODEL}" >&2
    exit 1
  fi
  PROFILE_NUMBER="${PROFILE#backbone_}"

  for entry in "${ENTRIES[@]}"; do
    PILOT="${entry%%:*}"
    remainder="${entry#*:}"
    INSTANCE="${remainder%%:*}"
    SUFFIX="${remainder##*:}"
    ARCHIVE="${PILOT}/observed_ai_backbones/${INSTANCE}/${PROFILE}"

    echo "[$(date --iso-8601=seconds)] ${INSTANCE}/${PROFILE}/${MODEL}: begin"
    if [[ -f "${ARCHIVE}/artifact.json" ]]; then
      echo "[$(date --iso-8601=seconds)] ${INSTANCE}/${PROFILE}/${MODEL}: complete (already archived)"
      continue
    fi

    ARCHIVED=false
    for candidate_attempt in $(seq 1 "${MAX_GENERATION_ATTEMPTS}"); do
      candidate="${PILOT}/generated_ai/${INSTANCE}/${MODEL}-${PROFILE}-r${candidate_attempt}"
      if [[ -e "${candidate}/QUARANTINED.json" ]]; then
        echo "Retaining quarantined generation and trying a fresh attempt: ${candidate}" >&2
        continue
      fi
      if [[ -f "${candidate}/artifact.json" ]]; then
        if jq -e \
          --arg instance "${INSTANCE}" \
          --arg model "${MODEL}" \
          --arg profile "${PROFILE}" \
          '.instance_id == $instance and .model == $model and .profile == $profile and .generation_error == null and .patch_empty == false and .clean_tracked_baseline_verified == true' \
          "${candidate}/artifact.json" >/dev/null; then
          :
        else
          echo "Retaining unusable generation and trying a fresh attempt: ${candidate}" >&2
          continue
        fi
      elif [[ -e "${candidate}" ]]; then
        echo "Incomplete AI generation directory requires inspection: ${candidate}" >&2
        exit 1
      else
        if ! python scripts/generate_share_swebench_ai_patch.py \
          --pilot "${PILOT}" \
          --workflows "${WORKFLOWS}" \
          --instance "${INSTANCE}" \
          --profile "${PROFILE}" \
          --env .env \
          --max-turns 30 \
          --out "${candidate}"; then
          echo "Generation attempt r${candidate_attempt} did not yield a usable patch: ${candidate}" >&2
          continue
        fi
      fi

      RUN_ID="share-ai-${SUFFIX}-backbone${PROFILE_NUMBER}-r${candidate_attempt}"
      RESULTS="results/share_swebench_ai_${SUFFIX}_backbone${PROFILE_NUMBER}_r${candidate_attempt}"
      if [[ ! -f "${RESULTS}/${MODEL}.${RUN_ID}.json" ]]; then
        .venv-swebench/bin/swebench eval "${DATASET}" \
          --predictions "${candidate}/prediction.jsonl" \
          --instance "${INSTANCE}" \
          --run-id "${RUN_ID}" \
          --workers 1 \
          --timeout 1800 \
          --report-dir "${RESULTS}"
      fi

      DETAIL="logs/run_evaluation/${RUN_ID}/${MODEL}/${INSTANCE}/report.json"
      if [[ ! -f "${DETAIL}" ]]; then
        echo "Completed evaluation summary lacks its detailed report: ${DETAIL}" >&2
        exit 1
      fi
      if ! jq -e --arg instance "${INSTANCE}" '.[$instance].patch_successfully_applied == true' "${DETAIL}" >/dev/null; then
        if jq -e --arg instance "${INSTANCE}" '.[$instance].patch_successfully_applied == false and .[$instance].infra_failure == false' "${DETAIL}" >/dev/null; then
          echo "Retaining non-applicable evaluated patch and trying a fresh attempt: ${candidate}" >&2
          continue
        fi
        echo "Evaluation detail is invalid or indicates infrastructure failure: ${DETAIL}" >&2
        exit 1
      fi

      python scripts/archive_share_swebench_ai_result.py \
        --pilot "${PILOT}" \
        --instance "${INSTANCE}" \
        --generation "${candidate}" \
        --model "${MODEL}" \
        --run-id "${RUN_ID}" \
        --report-dir "${RESULTS}" \
        --out "${ARCHIVE}"
      ARCHIVED=true
      break
    done
    if [[ "${ARCHIVED}" != true ]]; then
      echo "No archivable generation after ${MAX_GENERATION_ATTEMPTS} attempts for ${INSTANCE}/${PROFILE}" >&2
      exit 1
    fi
    echo "[$(date --iso-8601=seconds)] ${INSTANCE}/${PROFILE}/${MODEL}: complete"
  done
done

echo "[$(date --iso-8601=seconds)] All ten cross-backbone AI branches complete"
