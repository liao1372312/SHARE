from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flare.backbone import MultiAgentComposer
from flare.core import Agent, Event, NodeStatus, NodeType
from flare.services import ParameterSpec, ServiceRepository, ServiceSpec
from flare.slicing import affected_downstream_slice

PUBLIC_TASKS = [
    ("Get current weather for city {city}", "weather", {"city": "Beijing"}),
    ("Get weather forecast for city {city}", "weather", {"city": "Shanghai"}),
    ("Get stock quote for symbol {symbol}", "finance", {"symbol": "AAPL"}),
    ("Translate '{text}' to target language {target_language}", "translation", {"text": "hello world", "target_language": "English"}),
]
ENTERPRISE_TASKS = [
    "查询项目初始信息并返回项目状态",
    "查询规划报告信息并汇总报告状态",
    "查询合规检测状态并反馈ERP建项结果",
    "查询流程实例状态并推送流程交互信息",
    "根据单位和年份计算项目评价平均指标值",
]
STRESS_TASKS = [
    ("Get current weather for city {city}", "weather", {"city": "Beijing"}),
    ("Get stock quote for symbol {symbol}", "finance", {"symbol": "AAPL"}),
    ("Translate '{text}' to target language {target_language}", "translation", {"text": "repair me", "target_language": "French"}),
]
FAILURE_TYPES = [
    "planning_failure",
    "service_selection_failure",
    "parameter_binding_failure",
    "tool_execution_failure",
    "supervisor_failure",
    "answer_grounding_failure",
]
FAILURE_TO_AGENT = {
    "planning_failure": "planner",
    "service_selection_failure": "service_provider",
    "parameter_binding_failure": "executor",
    "tool_execution_failure": "external_tool",
    "supervisor_failure": "supervisor",
    "answer_grounding_failure": "summarizer",
}
FAILURE_TO_ACTION = {
    "planning_failure": "AddSubtask",
    "service_selection_failure": "ReselectAPI",
    "parameter_binding_failure": "RebindParam",
    "tool_execution_failure": "RetryTool",
    "supervisor_failure": "ReviseRepair",
    "answer_grounding_failure": "ReviseAnswer",
}
DATASET_CONFIG = {
    "public_api": {"tasks": 1200, "apis": 428, "failures": 216, "templates": PUBLIC_TASKS},
    "enterprise_service": {"tasks": 1500, "apis": 7479, "failures": 315, "templates": ENTERPRISE_TASKS},
    "mixed_stress": {"tasks": 600, "apis": 7907, "failures": 198, "templates": STRESS_TASKS},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FLARE experiment datasets with repair labels.")
    parser.add_argument("--out", type=Path, default=Path("datasets/flare_synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale task/failure counts for smoke generation.")
    args = parser.parse_args()
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, config in DATASET_CONFIG.items():
        dataset = generate_dataset(name, config, args.scale)
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in dataset:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        summaries.append(summarize_dataset(name, dataset, config, path))
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_schema(args.out / "SCHEMA.md")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def generate_dataset(name: str, config: Dict[str, Any], scale: float) -> List[Dict[str, Any]]:
    task_count = max(1, round(config["tasks"] * scale))
    failure_count = min(task_count, max(0, round(config["failures"] * scale)))
    failure_indices = set(random.sample(range(task_count), failure_count))
    repository = make_repository(name, min(config["apis"], 300 if scale < 1 else config["apis"]))
    composer = MultiAgentComposer(repository=repository)
    items = []
    failure_cycle = cycle_failures(name)
    for index in range(task_count):
        failed = index in failure_indices
        failure_type = next(failure_cycle) if failed else None
        task = make_task(name, config["templates"], index)
        events = build_trace(composer, task, failure_type, index)
        graph = composer.build_graph(events) if hasattr(composer, "build_graph") else None
        labels = build_labels(events, failure_type)
        items.append(
            {
                "id": f"{name}_synth_{index + 1:06d}",
                "dataset": name,
                "synthetic": True,
                "task": task,
                "api_pool_size": config["apis"],
                "candidate_api_ids": sample_api_ids(repository, 12),
                "events": [event_to_dict(event) for event in events],
                "repair_label": labels,
                "metadata": {
                    "source": "synthetic_generated_for_flare_experiments",
                    "primary_failure_injected": failed,
                    "failure_type": failure_type,
                    "expected_repair_action": labels.get("expected_repair_action"),
                    "generator": "scripts/generate_flare_datasets.py",
                },
            }
        )
    return items


def build_trace(composer: MultiAgentComposer, task: str, failure_type: str | None, index: int) -> List[Event]:
    if failure_type == "parameter_binding_failure":
        events, _ = composer.execute_initial(task, inject_failure="missing_param")
    elif failure_type == "tool_execution_failure":
        events, _ = composer.execute_initial(task, inject_failure="tool_error")
    else:
        events, _ = composer.execute_initial(task)
    if failure_type == "planning_failure":
        events = inject_planning_failure(events)
    elif failure_type == "service_selection_failure":
        events = inject_service_selection_failure(events)
    elif failure_type == "supervisor_failure":
        events = inject_supervisor_failure(events)
    elif failure_type == "answer_grounding_failure":
        events = inject_answer_grounding_failure(events)
    return events


def build_labels(events: Sequence[Event], failure_type: str | None) -> Dict[str, Any]:
    if failure_type is None:
        return {
            "failed": False,
            "faulty_agent": None,
            "failure_type": None,
            "faulty_decision_node": None,
            "expected_repair_action": None,
            "affected_slice": [],
            "annotation_source": "synthetic_no_failure",
        }
    faulty_node = choose_faulty_node(events, failure_type)
    affected = affected_ids(events, faulty_node)
    return {
        "failed": True,
        "faulty_agent": FAILURE_TO_AGENT[failure_type],
        "failure_type": failure_type,
        "faulty_decision_node": faulty_node,
        "expected_repair_action": FAILURE_TO_ACTION[failure_type],
        "affected_slice": affected,
        "annotation_source": "synthetic_single_primary_failure",
    }


def choose_faulty_node(events: Sequence[Event], failure_type: str) -> str:
    target_types = {
        "planning_failure": NodeType.PLAN,
        "service_selection_failure": NodeType.SERVICE_SELECTION,
        "parameter_binding_failure": NodeType.PARAMETER_BINDING,
        "tool_execution_failure": NodeType.API_INVOCATION,
        "supervisor_failure": NodeType.REPAIR,
        "answer_grounding_failure": NodeType.CLAIM,
    }
    target = target_types[failure_type]
    for event in events:
        if event.type == target:
            return event.id
    return events[-1].id


def affected_ids(events: Sequence[Event], faulty_node: str) -> List[str]:
    children = defaultdict(list)
    for event in events:
        for parent in event.meta.get("depends_on", []):
            children[str(parent)].append(event.id)
    seen, stack = set(), list(children[faulty_node])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children[current])
    return sorted(seen)


def inject_planning_failure(events: List[Event]) -> List[Event]:
    for event in events:
        if event.type == NodeType.PLAN:
            event.status = NodeStatus.FAILED
            event.meta["missing_dependency"] = "missing prerequisite subtask"
            event.meta["invalid_dependency"] = True
            break
    return events


def inject_service_selection_failure(events: List[Event]) -> List[Event]:
    for event in events:
        if event.type == NodeType.SERVICE_SELECTION:
            event.status = NodeStatus.FAILED
            event.meta["wrong_api"] = True
            event.meta["hallucinated_api"] = "nonexistent.service"
            break
    return events


def inject_supervisor_failure(events: List[Event]) -> List[Event]:
    anchor = next((event.id for event in events if event.status == NodeStatus.FAILED), events[-1].id)
    events.append(Event(f"r{len(events)+1}", Agent.SUPERVISOR, NodeType.REPAIR, output={"action": "RetryTool"}, meta={"depends_on": [anchor], "repair_failed": True, "repeated_retry": 3}, status=NodeStatus.FAILED, time=max(event.time for event in events) + 1))
    return events


def inject_answer_grounding_failure(events: List[Event]) -> List[Event]:
    for event in reversed(events):
        if event.type == NodeType.CLAIM:
            event.meta["unsupported"] = True
            event.meta.pop("supported_by", None)
            break
    return events


def make_repository(name: str, api_count: int) -> ServiceRepository:
    repo = ServiceRepository.builtin()
    if name == "enterprise_service":
        enterprise_services = [
            ("enterprise.project_init", "查询项目初始信息接口", "查询项目初始信息并返回项目状态 query"),
            ("enterprise.plan_report", "查询规划报告信息接口", "查询规划报告信息并汇总报告状态 query"),
            ("enterprise.compliance_erp", "合规检测与ERP反馈接口", "查询合规检测状态并反馈ERP建项结果 query"),
            ("enterprise.flow_interact", "流程实例状态与交互推送接口", "查询流程实例状态并推送流程交互信息 query"),
            ("enterprise.eval_average", "项目评价平均指标计算接口", "根据单位和年份计算项目评价平均指标值 query"),
        ]
        for record_id, api_name, description in enterprise_services:
            repo.add(
                ServiceSpec(
                    record_id=record_id,
                    api_name=api_name,
                    description=description,
                    domain="general",
                    category="enterprise",
                    endpoint=f"/enterprise/{record_id.split('.')[-1]}",
                    parameters=[ParameterSpec("query", required=True)],
                    outputs=["status", "data"],
                )
            )
    for index in range(len(repo.services), api_count):
        domain = random.choice(["travel", "medical", "technology", "billing", "analytics", "project", "messaging", "system"])
        repo.add(
            ServiceSpec(
                record_id=f"{name}.api_{index:05d}",
                api_name=f"{domain.title()} Synthetic Service {index}",
                description=f"Synthetic {domain} service for controlled FLARE experiments",
                domain=domain,
                category=domain,
                endpoint=f"/{domain}/synthetic/{index}",
                parameters=[ParameterSpec("query", required=index % 5 == 0)],
                outputs=["status", "data"],
            )
        )
    return repo


def make_task(name: str, templates: Sequence[Any], index: int) -> str:
    template = templates[index % len(templates)]
    if isinstance(template, tuple):
        text, _, values = template
        varied = dict(values)
        if "city" in varied:
            varied["city"] = ["Beijing", "Shanghai", "Tokyo", "Paris"][index % 4]
        if "symbol" in varied:
            varied["symbol"] = ["AAPL", "MSFT", "GOOG", "TSLA"][index % 4]
        return text.format(**varied)
    suffix = f" query: standard_value（样本{index + 1}）" if name == "enterprise_service" else f" sample {index + 1} query: standard_value"
    return str(template) + suffix


def cycle_failures(name: str):
    order = FAILURE_TYPES if name == "mixed_stress" else FAILURE_TYPES[:4] + ["answer_grounding_failure"]
    while True:
        for failure in order:
            yield failure


def sample_api_ids(repository: ServiceRepository, count: int) -> List[str]:
    ids = list(repository.services)
    if len(ids) <= count:
        return ids
    return random.sample(ids, count)


def event_to_dict(event: Event) -> Dict[str, Any]:
    return {
        "id": event.id,
        "agent": event.agent.value if isinstance(event.agent, Agent) else event.agent,
        "type": event.type.value if isinstance(event.type, NodeType) else event.type,
        "input": json_safe(event.input),
        "output": json_safe(event.output),
        "meta": json_safe(event.meta),
        "status": event.status.value if isinstance(event.status, NodeStatus) else event.status,
        "time": event.time,
    }


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (Agent, NodeType, NodeStatus)):
        return value.value
    return value


def summarize_dataset(name: str, dataset: Sequence[Dict[str, Any]], config: Dict[str, Any], path: Path) -> Dict[str, Any]:
    failures = [item for item in dataset if item["repair_label"]["failed"]]
    type_counts = Counter(item["repair_label"]["failure_type"] for item in failures)
    action_counts = Counter(item["repair_label"]["expected_repair_action"] for item in failures)
    return {
        "dataset": name,
        "path": str(path),
        "tasks": len(dataset),
        "apis": config["apis"],
        "failures": len(failures),
        "faulty_decisions": len(failures),
        "failure_type_counts": dict(type_counts),
        "repair_action_counts": dict(action_counts),
        "synthetic": True,
    }


def write_schema(path: Path) -> None:
    path.write_text(
        """# FLARE Synthetic Dataset Schema\n\nAll files in this directory are synthetic and generated for controlled FLARE experiments.\n\nEach JSONL row contains:\n\n- `id`: stable sample id.\n- `dataset`: one of `public_api`, `enterprise_service`, `mixed_stress`.\n- `synthetic`: always `true`.\n- `task`: natural-language service composition request.\n- `api_pool_size`: nominal API pool size used for paper statistics.\n- `candidate_api_ids`: sampled candidate APIs.\n- `events`: observable execution records matching `rt=<id,agent,type,input,output,meta,status,time>`.\n- `repair_label.failed`: whether a primary failure is injected.\n- `repair_label.faulty_agent`: planner, service_provider, executor, external_tool, supervisor, or summarizer.\n- `repair_label.failure_type`: planning_failure, service_selection_failure, parameter_binding_failure, tool_execution_failure, supervisor_failure, or answer_grounding_failure.\n- `repair_label.faulty_decision_node`: gold faulty decision node id.\n- `repair_label.expected_repair_action`: AddSubtask, Replan, ReselectAPI, RebindParam, RetryTool, ReplaceTool, ReviseRepair, ReviseAnswer, MarkUncertain, or Terminate-compatible action.\n- `repair_label.affected_slice`: downstream nodes to invalidate/re-execute.\n- `metadata`: generator provenance; never claim these are real user traces.\n""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
