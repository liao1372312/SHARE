#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PILOT="datasets/share_swebench_r002_protocol_v1"
DATASET="/tmp/swebench_verified.parquet"
PROFILES="logs/share-r002-profiles.json"
PROTOCOL="${PILOT}/protocol_manifest.json"
LOG_FILE="logs/share-swebench-r002-controlled.log"
PID_FILE="logs/share-swebench-r002-controlled.pid"

verify_protocol() {
  python scripts/freeze_share_swebench_r002_protocol.py \
    --pilot "${PILOT}" --dataset "${DATASET}" --profiles "${PROFILES}" \
    --out "${PROTOCOL}" --verify
}

if [[ "${1:-}" == "--preflight" ]]; then
  verify_protocol
  python scripts/check_share_swebench_runtime.py --pilot "${PILOT}" --python .venv-swebench/bin/python
  missing=0
  while IFS=$'\t' read -r instance image; do
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
      echo "Missing image: ${instance} ${image}" >&2
      missing=$((missing + 1))
    fi
  done < <(jq -r '.bindings.tasks[] | [.instance_id,.image] | @tsv' "${PROTOCOL}")
  echo "Missing images: ${missing}"
  [[ "${missing}" -eq 0 ]]
  exit
fi

if [[ "${1:-}" != "--worker" ]]; then
  verify_protocol
  mkdir -p logs
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "R002 worker is already running as PID ${old_pid}" >&2
      exit 1
    fi
  fi
  nohup "${BASH_SOURCE[0]}" --worker >>"${LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" >"${PID_FILE}"
  echo "Started controlled R002 collection as PID ${pid}"
  echo "Log: ${LOG_FILE}"
  echo "Follow with: tail -f ${LOG_FILE}"
  exit
fi

verify_protocol
echo "[$(date --iso-8601=seconds)] R002 fixed single-attempt collection begins"
while IFS=$'\t' read -r instance image; do
  docker image inspect "${image}" >/dev/null
  suffix="$(printf '%s' "${instance}" | tr -cs 'A-Za-z0-9' '-')"
  while IFS=$'\t' read -r profile model; do
    archive="${PILOT}/observed_r002_ai/${instance}/${profile}"
    generation="${PILOT}/generated_r002_ai/${instance}/${profile}/generation"
    run_id="share-r002-${suffix}-${profile}-r1"
    report_dir="results/share_swebench_r002_${suffix}_${profile}_r1"
    echo "[$(date --iso-8601=seconds)] ${instance}/${profile}/${model}: begin"
    if [[ -f "${archive}/artifact.json" ]]; then
      echo "[$(date --iso-8601=seconds)] ${instance}/${profile}/${model}: already archived"
      continue
    fi
    if [[ ! -f "${generation}/artifact.json" ]]; then
      if [[ -e "${generation}" ]]; then
        echo "Incomplete generation directory: ${generation}" >&2
        exit 1
      fi
      python scripts/generate_share_swebench_ai_patch.py \
        --pilot "${PILOT}" --workflows "${PILOT}/workflows.jsonl" \
        --instance "${instance}" --profile "${profile}" --env .env \
        --max-turns 20 --max-output-tokens 2048 --api-timeout 180 --command-timeout 180 \
        --out "${generation}" || true
    fi
    if jq -e '.generation_error == null and .patch_empty == false and .clean_tracked_baseline_verified == true' "${generation}/artifact.json" >/dev/null; then
      if [[ ! -f "${report_dir}/${model}.${run_id}.json" ]]; then
        .venv-swebench/bin/swebench eval "${DATASET}" \
          --predictions "${generation}/prediction.jsonl" --instance "${instance}" \
          --run-id "${run_id}" --workers 1 --timeout 1800 --report-dir "${report_dir}"
      fi
    fi
    python scripts/archive_share_swebench_r002_branch.py \
      --protocol "${PROTOCOL}" --pilot "${PILOT}" --instance "${instance}" \
      --profile "${profile}" --generation "${generation}" --run-id "${run_id}" \
      --report-dir "${report_dir}" --out "${archive}"
    echo "[$(date --iso-8601=seconds)] ${instance}/${profile}/${model}: complete"
  done < <(jq -r '.bindings.profiles[] | [.name,.model] | @tsv' "${PROTOCOL}")
done < <(jq -r '.bindings.tasks[] | [.instance_id,.image] | @tsv' "${PROTOCOL}")
echo "[$(date --iso-8601=seconds)] All 33 R002 branches complete"
