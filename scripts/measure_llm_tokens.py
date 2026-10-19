from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

METHOD_PROMPTS = {
    "No-Repair": "You are evaluating a failed service-composition trace. Do not repair it. Return a short JSON status.",
    "Retry-only": "Given the failed tool invocation below, decide whether to retry the same call. Return a short JSON repair plan.",
    "Full Re-execution": "Given the original user request and failed trace, regenerate the full multi-agent service-composition workflow from scratch. Return a concise JSON plan.",
    "Last-Step Repair": "Given the failed trace, repair only the last failed execution step. Return a concise JSON patch.",
    "Supervisor-only Repair": "Act as a supervisor agent. Diagnose the failure from the trace and choose a repair action. Return JSON.",
    "LLM-Diagnose-and-Repair": "Given the complete execution trace, identify the root cause and generate a repair plan. Return JSON with faulty_node, failure_type, action, and rationale.",
    "EPF+Repair": "Given the provenance-style execution trace, attribute the failure and choose a repair action. Return concise JSON.",
    "FLARE": "Given the localized faulty decision and affected execution slice, generate the minimal repair action. Return concise JSON.",
}
METHOD_KEYS = {
    "No-Repair": "no_repair",
    "Retry-only": "retry_only",
    "Full Re-execution": "full_reexecution",
    "Last-Step Repair": "last_step_repair",
    "Supervisor-only Repair": "supervisor_only",
    "LLM-Diagnose-and-Repair": "llm_diagnose_and_repair",
    "EPF+Repair": "epf_repair",
    "FLARE": "flare",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real token usage with an OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--data", type=Path, default=Path("datasets/flare_synthetic"))
    parser.add_argument("--out", type=Path, default=Path("results/flare_experiments_realistic/token_usage"))
    parser.add_argument("--env", type=Path, default=Path(".env"), help="Path to dotenv-style LLM configuration file.")
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-trace-chars", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    load_env_file(args.env)
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "GPT-5.4 Mini")
    if not base_url or not api_key:
        raise SystemExit("Please set LLM_BASE_URL and LLM_API_KEY in the environment. Do not put secrets in source files.")

    args.out.mkdir(parents=True, exist_ok=True)
    samples = load_failed_samples(args.data, args.samples_per_dataset, args.seed)
    raw_path = args.out / "raw_token_usage.jsonl"
    rows: List[Dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw:
        for sample in samples:
            for method in METHOD_PROMPTS:
                prompt = build_prompt(method, sample, args.max_trace_chars)
                usage, response_id = call_chat_completion(base_url, api_key, model, prompt, args.temperature, args.max_tokens)
                row = {
                    "dataset": sample["dataset"],
                    "sample_id": sample["id"],
                    "method": method,
                    "response_id": response_id,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                }
                rows.append(row)
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()
                time.sleep(0.05)
    summary = summarize(rows)
    write_csv(args.out / "token_usage_summary.csv", summary)
    (args.out / "token_usage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex_rows(args.out / "token_latex_rows.tex", summary)
    print(json.dumps({"out": str(args.out), "samples": len(samples), "methods": len(METHOD_PROMPTS)}, ensure_ascii=False, indent=2))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_failed_samples(root: Path, per_dataset: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failed = [row for row in rows if row.get("repair_label", {}).get("failed")]
        selected.extend(rng.sample(failed, min(per_dataset, len(failed))))
    return selected


def build_prompt(method: str, sample: Dict[str, Any], max_trace_chars: int) -> str:
    label = sample.get("repair_label", {})
    if method == "FLARE":
        payload = {
            "task": sample["task"],
            "localized_faulty_node": label.get("faulty_decision_node"),
            "failure_type": label.get("failure_type"),
            "expected_action_space": ["AddSubtask", "Replan", "ReselectAPI", "RebindParam", "RetryTool", "ReplaceTool", "ReviseRepair", "ReviseAnswer", "MarkUncertain", "Terminate"],
            "affected_slice": label.get("affected_slice", []),
        }
    elif method == "EPF+Repair":
        payload = {
            "task": sample["task"],
            "provenance_events": compact_events(sample["events"], include_outputs=False),
        }
    elif method == "Full Re-execution":
        payload = {"task": sample["task"], "api_pool_size": sample.get("api_pool_size"), "candidate_api_ids": sample.get("candidate_api_ids", [])}
    elif method == "No-Repair":
        payload = {"task": sample["task"], "observed_failure": label.get("failure_type")}
    else:
        payload = {"task": sample["task"], "trace": compact_events(sample["events"], include_outputs=True)}
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > max_trace_chars:
        text = text[:max_trace_chars] + "...<truncated>"
    return METHOD_PROMPTS[method] + "\n\nInput:\n" + text


def compact_events(events: Iterable[Dict[str, Any]], include_outputs: bool) -> List[Dict[str, Any]]:
    compact = []
    for event in events:
        item = {"id": event.get("id"), "agent": event.get("agent"), "type": event.get("type"), "status": event.get("status"), "meta": event.get("meta", {})}
        if include_outputs:
            item["output"] = event.get("output")
        compact.append(item)
    return compact


def call_chat_completion(base_url: str, api_key: str, model: str, prompt: str, temperature: float, max_tokens: int) -> tuple[Dict[str, int], str]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise evaluator for service-composition repair experiments."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail[:500]}") from exc
    usage = data.get("usage") or {}
    return usage, str(data.get("id", ""))


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = []
    for method, items in by_method.items():
        summary.append(
            {
                "method": method,
                "method_key": METHOD_KEYS[method],
                "n": len(items),
                "prompt_tokens": round(sum(item["prompt_tokens"] for item in items) / len(items), 2),
                "completion_tokens": round(sum(item["completion_tokens"] for item in items) / len(items), 2),
                "total_tokens": round(sum(item["total_tokens"] for item in items) / len(items), 2),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [f"% {row['method']}: avg total tokens = {row['total_tokens']}" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
