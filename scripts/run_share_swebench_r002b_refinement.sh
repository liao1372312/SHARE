#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PILOT="datasets/share_swebench_r002_protocol_v1"
OUT_ROOT="datasets/share_swebench_r002b_verified_refinement_v2"
DATASET="/tmp/swebench_verified.parquet"
PROFILES="logs/share-r002-profiles.json"
PROTOCOL="${OUT_ROOT}/protocol_manifest.json"
LOG_FILE="logs/share-swebench-r002b-refinement.log"
PID_FILE="logs/share-swebench-r002b-refinement.pid"

verify_protocol() {
  python scripts/freeze_share_swebench_r002b_protocol.py \
    --r002-pilot "${PILOT}" --profiles "${PROFILES}" --out "${PROTOCOL}" --verify
}

if [[ "${1:-}" == "--preflight" ]]; then
  verify_protocol
  python scripts/check_share_swebench_runtime.py --pilot "${PILOT}" --python .venv-swebench/bin/python
  exit
fi

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p "${OUT_ROOT}" logs
  if [[ ! -f "${PROTOCOL}" ]]; then
    python scripts/freeze_share_swebench_r002b_protocol.py \
      --r002-pilot "${PILOT}" --profiles "${PROFILES}" --out "${PROTOCOL}"
  fi
  verify_protocol
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "R002b worker is already running as PID ${old_pid}" >&2
      exit 1
    fi
  fi
  nohup bash "${BASH_SOURCE[0]}" --worker >>"${LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" >"${PID_FILE}"
  echo "Started R002b verified refinement as PID ${pid}"
  echo "Log: ${LOG_FILE}"
  exit
fi

verify_protocol
echo "[$(date --iso-8601=seconds)] R002b uniform verified refinement begins"
while IFS=$'\t' read -r instance image; do
  docker image inspect "${image}" >/dev/null
  suffix="$(printf '%s' "${instance}" | tr -cs 'A-Za-z0-9' '-')"
  initial_patch="${PILOT}/observed_r002_ai/${instance}/primary/raw/generation_model.patch"
  generation="${OUT_ROOT}/generated/${instance}/primary"
  run_id="share-r002b-${suffix}-primary-r1"
  report_dir="results/share_swebench_r002b_${suffix}_primary_r1"
  echo "[$(date --iso-8601=seconds)] ${instance}/primary: begin"
  if [[ ! -f "${generation}/artifact.json" ]]; then
    if [[ -e "${generation}" ]]; then
      echo "Incomplete R002b generation directory: ${generation}" >&2
      exit 1
    fi
    python scripts/refine_share_swebench_ai_patch.py \
      --pilot "${PILOT}" --workflows "${PILOT}/workflows.jsonl" \
      --instance "${instance}" --profile primary --initial-patch "${initial_patch}" \
      --env .env --max-turns 20 --max-output-tokens 4096 \
      --api-timeout 180 --command-timeout 300 --out "${generation}" || true
  fi
  if jq -e '.generation_error == null and .patch_empty == false and .test_verified_after_latest_edit == true and .diff_inspected_after_latest_edit == true' "${generation}/artifact.json" >/dev/null; then
    if [[ ! -f "${report_dir}/gpt-5.4-mini+verified-refinement.${run_id}.json" ]]; then
      .venv-swebench/bin/swebench eval "${DATASET}" \
        --predictions "${generation}/prediction.jsonl" --instance "${instance}" \
        --run-id "${run_id}" --workers 1 --timeout 1800 --report-dir "${report_dir}"
    fi
  fi
  echo "[$(date --iso-8601=seconds)] ${instance}/primary: complete"
done < <(jq -r '.bindings.tasks[] | [.instance_id,.image] | @tsv' "${PROTOCOL}")
echo "[$(date --iso-8601=seconds)] All 11 R002b branches complete"
