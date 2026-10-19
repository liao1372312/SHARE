from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flare.core import Agent, Event, NodeStatus, NodeType
from scripts.generate_flare_datasets import (
    FAILURE_TYPES,
    affected_ids,
    build_labels,
    event_to_dict,
    inject_answer_grounding_failure,
    inject_planning_failure,
    inject_service_selection_failure,
    inject_supervisor_failure,
)

DOMAIN_SOURCES = {
    "finance": Path("dataset/Finance"),
    "medical": Path("dataset/Medical"),
    "technology": Path("dataset/Technology"),
    "tourism": Path("dataset/Tourism"),
    "cross_domain": Path("dataset/Cross_Domain"),
}
API_POOL_SIZE = {
    "finance": 725,
    "medical": 254,
    "technology": 2793,
    "tourism": 1093,
    "cross_domain": 837,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert existing domain service-composition tasks into FLARE-compatible benchmark JSONL.")
    parser.add_argument("--out", type=Path, default=Path("datasets/flare_domain"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--tasks-per-domain", type=int, default=500)
    parser.add_argument("--failure-ratio", type=float, default=0.35)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for domain, source in DOMAIN_SOURCES.items():
        items = load_domain_tasks(source, domain, args.tasks_per_domain, rng)
        dataset = build_dataset(domain, items, args.failure_ratio, rng)
        path = args.out / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries.append(summarize(domain, dataset, path))
    (args.out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_schema(args.out / "SCHEMA.md")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def load_domain_tasks(source: Path, domain: str, limit: int, rng: random.Random) -> List[Dict[str, Any]]:
    files = sorted(source.glob("*.json"))
    rng.shuffle(files)
    selected = []
    for path in files:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_list = item.get("TaskList") or []
        if not item.get("user_query") or not isinstance(task_list, list) or not task_list:
            continue
        selected.append({"source_file": str(path), **item})
        if len(selected) >= limit:
            break
    return selected


def build_dataset(domain: str, tasks: Sequence[Dict[str, Any]], failure_ratio: float, rng: random.Random) -> List[Dict[str, Any]]:
    failure_count = min(len(tasks), round(len(tasks) * failure_ratio))
    failure_indices = set(rng.sample(range(len(tasks)), failure_count))
    failure_cycle = cycle_domain_failures(domain)
    rows = []
    for index, item in enumerate(tasks):
        failed = index in failure_indices
        failure_type = next(failure_cycle) if failed else None
        events = build_domain_trace(domain, item, failure_type)
        labels = build_labels(events, failure_type)
        row = {
            "id": f"{domain}_domain_{index + 1:06d}",
            "dataset": domain,
            "synthetic": True,
            "domain_benchmark": True,
            "task": item["user_query"],
            "api_pool_size": API_POOL_SIZE.get(domain, len(tasks)),
            "candidate_api_ids": candidate_api_ids(domain, item.get("TaskList") or [], index),
            "events": [event_to_dict(event) for event in events],
            "repair_label": labels,
            "metadata": {
                "source": "existing_domain_service_composition_tasks",
                "source_file": item.get("source_file"),
                "original_domain": item.get("domain"),
                "primary_failure_injected": failed,
                "failure_type": failure_type,
                "expected_repair_action": labels.get("expected_repair_action"),
                "generator": "scripts/generate_flare_domain_datasets.py",
            },
        }
        rows.append(row)
    return rows


def build_domain_trace(domain: str, item: Dict[str, Any], failure_type: str | None) -> List[Event]:
    task_text = str(item["user_query"])
    task_list = normalize_task_list(item.get("TaskList") or [], domain)
    events: List[Event] = [Event("q1", Agent.SYSTEM, NodeType.REQUEST, output=task_text, status=NodeStatus.SUCCESS, time=1.0)]
    prior = "q1"
    observations = []
    time = 2.0
    for step in task_list:
        sid = int(step["id"])
        plan_id = f"p{sid}"
        service_id = f"s{sid}"
        bind_id = f"b{sid}"
        exec_id = f"e{sid}"
        obs_id = f"o{sid}"
        events.append(Event(plan_id, Agent.PLANNER, NodeType.PLAN, output=step, meta={"depends_on": [prior]}, status=NodeStatus.SUCCESS, time=time)); time += 1
        service = service_spec(domain, step, sid)
        events.append(Event(service_id, Agent.SERVICE_PROVIDER, NodeType.SERVICE_SELECTION, output=service, meta={"depends_on": [plan_id], "subtask": step}, status=NodeStatus.SUCCESS, time=time)); time += 1
        params = bind_inputs(step, task_text, observations)
        events.append(Event(bind_id, Agent.EXECUTOR, NodeType.PARAMETER_BINDING, output=params, meta={"depends_on": [service_id], "service_id": service["record_id"]}, status=NodeStatus.SUCCESS, time=time)); time += 1
        output = invoke_output(domain, step, params)
        events.append(Event(exec_id, Agent.EXECUTOR, NodeType.API_INVOCATION, output=output, meta={"depends_on": [bind_id], "service_id": service["record_id"], "parameters": params}, status=NodeStatus.SUCCESS, time=time)); time += 1
        events.append(Event(obs_id, Agent.EXECUTOR, NodeType.OBSERVATION, output=output, meta={"depends_on": [exec_id], "service_id": service["record_id"]}, status=NodeStatus.SUCCESS, time=time)); time += 1
        observations.append(output)
        prior = obs_id
    for idx, observation in enumerate(observations, start=1):
        events.append(Event(f"claim_{idx}", Agent.SUMMARIZER, NodeType.CLAIM, output={"claim": claim_text(domain, observation)}, meta={"depends_on": [prior], "supported_by": [f"o{idx}"]}, status=NodeStatus.SUCCESS, time=time + idx / 1000))
    events = inject_domain_failure(events, failure_type)
    return events


def normalize_task_list(task_list: Sequence[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    normalized = []
    for idx, raw in enumerate(task_list, start=1):
        step_id = raw.get("id") if isinstance(raw.get("id"), int) else idx
        deps = raw.get("depends_on") or ([step_id - 1] if step_id > 1 else [])
        normalized.append(
            {
                "id": step_id,
                "description": str(raw.get("description") or raw.get("task") or "Perform service step"),
                "domain": str(raw.get("domain") or domain),
                "depends_on": deps,
                "required_inputs": [str(x) for x in raw.get("required_inputs") or []],
            }
        )
    return normalized[:6]


def service_spec(domain: str, step: Dict[str, Any], sid: int) -> Dict[str, Any]:
    required = [clean_param(x) for x in step.get("required_inputs", [])]
    return {
        "record_id": f"{domain}.service_{sid:02d}.{abs(hash(step['description'])) % 100000}",
        "api_name": f"{domain.replace('_', ' ').title()} Service {sid}",
        "description": step["description"],
        "domain": step.get("domain", domain),
        "category": domain,
        "endpoint": f"/{domain}/service/{sid}",
        "method": "GET",
        "parameters": [{"name": name, "required": True, "type": "str", "default": None, "description": ""} for name in required[:4]],
        "outputs": ["status", "data", "evidence"],
        "executable": True,
    }


def bind_inputs(step: Dict[str, Any], task_text: str, observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    params = {}
    for raw in step.get("required_inputs", [])[:4]:
        name = clean_param(raw)
        value = infer_value(raw, task_text, observations)
        params[name] = value
    if not params:
        params["query"] = task_text[:120]
    return params


def clean_param(raw: Any) -> str:
    text = str(raw).strip()
    text = text.split(":", 1)[0].strip() if ":" in text else text
    text = text.lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch == "_") or "query"


def infer_value(raw: Any, task_text: str, observations: Sequence[Dict[str, Any]]) -> str:
    text = str(raw)
    if ":" in text:
        return text.split(":", 1)[1].strip()
    for obs in reversed(observations):
        if obs.get("data"):
            return str(obs["data"])
    return task_text[:80]


def invoke_output(domain: str, step: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "status_code": 200,
        "domain": domain,
        "step_id": step["id"],
        "data": f"{domain} result for {step['description'][:60]}",
        "evidence": list(params.keys()),
    }


def claim_text(domain: str, observation: Dict[str, Any]) -> str:
    return f"The {domain} service returned evidence for step {observation.get('step_id')}."


def inject_domain_failure(events: List[Event], failure_type: str | None) -> List[Event]:
    if failure_type is None:
        return events
    if failure_type == "parameter_binding_failure":
        for event in events:
            if event.type == NodeType.PARAMETER_BINDING:
                event.status = NodeStatus.FAILED
                event.meta["missing_parameters"] = [next(iter((event.output or {"query": None}).keys()))]
                break
    elif failure_type == "tool_execution_failure":
        for event in events:
            if event.type == NodeType.API_INVOCATION:
                event.status = NodeStatus.FAILED
                event.output = {"ok": False, "status_code": 503, "error": "domain service unavailable"}
                event.meta["api_error"] = True
                break
    elif failure_type == "planning_failure":
        events = inject_planning_failure(events)
    elif failure_type == "service_selection_failure":
        events = inject_service_selection_failure(events)
    elif failure_type == "supervisor_failure":
        events = inject_supervisor_failure(events)
    elif failure_type == "answer_grounding_failure":
        events = inject_answer_grounding_failure(events)
    return events


def candidate_api_ids(domain: str, task_list: Sequence[Dict[str, Any]], index: int) -> List[str]:
    ids = [f"{domain}.service_{idx + 1:02d}.{abs(hash(str(step))) % 100000}" for idx, step in enumerate(task_list[:8])]
    while len(ids) < 12:
        ids.append(f"{domain}.background_api_{index}_{len(ids)}")
    return ids[:12]


def cycle_domain_failures(domain: str):
    order = FAILURE_TYPES if domain == "cross_domain" else FAILURE_TYPES[:4] + ["answer_grounding_failure"]
    while True:
        for failure in order:
            yield failure


def summarize(domain: str, dataset: Sequence[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    failures = [item for item in dataset if item["repair_label"]["failed"]]
    return {
        "dataset": domain,
        "path": str(path),
        "tasks": len(dataset),
        "apis": API_POOL_SIZE.get(domain, len(dataset)),
        "failures": len(failures),
        "faulty_decisions": len(failures),
        "failure_type_counts": dict(Counter(item["repair_label"]["failure_type"] for item in failures)),
        "repair_action_counts": dict(Counter(item["repair_label"]["expected_repair_action"] for item in failures)),
        "source_domains": True,
        "synthetic": True,
    }


def write_schema(path: Path) -> None:
    path.write_text(
        """# FLARE Domain Benchmark Schema\n\nThis benchmark is converted from existing domain service-composition task files under `dataset/`.\nThe user requests and task decompositions come from the existing domain datasets; execution traces and injected primary failures are generated in the FLARE schema for controlled localization and repair experiments.\n\nDomains: finance, medical, technology, tourism, cross_domain.\n\nEach JSONL row is compatible with `scripts/run_flare_experiments.py` and contains `events`, `repair_label`, and source metadata.\n""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
