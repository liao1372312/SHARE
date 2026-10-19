from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .claims import add_claim_nodes
from .core import Agent, Event, ExecutionGraph, NodeStatus, NodeType, RepairAction
from .pipeline import FLARE, FLAREConfig, FLAREResult
from .repair import RepairHandler
from .services import ServiceRepository, ServiceSpec, bind_parameters, invoke_service


@dataclass
class CompositionOutput:
    graph: ExecutionGraph
    result: FLAREResult
    answer: str
    final_answer: str


class PlannerAgent:
    def plan(self, task: str) -> List[Dict[str, Any]]:
        fragments = [part.strip() for part in re.split(r"\bthen\b|\band then\b|然后|并且|，|,|;|；", task, flags=re.IGNORECASE) if part.strip()]
        if not fragments:
            fragments = [task]
        return [
            {
                "id": index,
                "description": fragment,
                "domain": infer_domain(fragment),
                "depends_on": [index - 1] if index > 1 else [],
                "required_inputs": [],
            }
            for index, fragment in enumerate(fragments, start=1)
        ]


class ServiceProviderAgent:
    def __init__(self, repository: ServiceRepository):
        self.repository = repository

    def select(self, subtask: Mapping[str, Any], exclude: Sequence[str] = ()) -> ServiceSpec | None:
        return self.repository.select(str(subtask.get("description", "")), str(subtask.get("domain", "")), exclude=exclude)


class ExecutorAgent:
    def bind(self, service: ServiceSpec, task_text: str, prior_outputs: Sequence[Mapping[str, Any]] = ()) -> tuple[Dict[str, Any], List[str]]:
        return bind_parameters(service, task_text, prior_outputs)

    def invoke(self, service: ServiceSpec, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        return invoke_service(service, parameters)


class SummarizerAgent:
    def summarize(self, task: str, observations: Sequence[Mapping[str, Any]]) -> str:
        successful = [observation for observation in observations if observation.get("ok")]
        if not successful:
            return "The task could not be completed with the available execution evidence."
        pieces = []
        for observation in successful:
            if "condition" in observation:
                pieces.append(f"The weather in {observation.get('city')} is {observation.get('condition')} with temperature {observation.get('temperature')}.")
            elif "price" in observation:
                pieces.append(f"The stock price of {observation.get('symbol')} is {observation.get('price')} {observation.get('currency')}.")
            elif "translated_text" in observation:
                pieces.append(f"The translation result is {observation.get('translated_text')}.")
            else:
                pieces.append(f"The service returned {observation}.")
        return " ".join(pieces)


class SupervisorAgent:
    def make_repair_handler(self, composer: "MultiAgentComposer") -> RepairHandler:
        def handler(context_graph: ExecutionGraph, faulty_node: str, action: RepairAction, invalid: set[str]) -> ExecutionGraph:
            return composer.reexecute_from_fault(context_graph, faulty_node, action, invalid)

        return handler


class MultiAgentComposer:
    def __init__(self, repository: ServiceRepository | None = None, config: FLAREConfig | None = None):
        self.repository = repository or ServiceRepository.builtin()
        self.planner = PlannerAgent()
        self.provider = ServiceProviderAgent(self.repository)
        self.executor = ExecutorAgent()
        self.summarizer = SummarizerAgent()
        self.supervisor = SupervisorAgent()
        self.config = config or FLAREConfig(repair_budget=3)

    def run(self, task: str, inject_failure: str | None = None) -> CompositionOutput:
        events, answer = self.execute_initial(task, inject_failure=inject_failure)
        graph = ExecutionGraph.from_events(events)
        flare = FLARE(self.config, repair_handler=self.supervisor.make_repair_handler(self))
        result = flare.run(graph)
        return CompositionOutput(graph=result.graph, result=result, answer=answer, final_answer=final_answer_from_graph(result.graph))

    def execute_initial(self, task: str, inject_failure: str | None = None) -> tuple[List[Event], str]:
        events: List[Event] = []
        observations: List[Dict[str, Any]] = []
        events.append(Event("q1", Agent.SYSTEM, NodeType.REQUEST, output=task, status=NodeStatus.SUCCESS, time=1))
        steps = self.planner.plan(task)
        prior_step_node = "q1"
        time = 2.0
        selected_by_step: Dict[int, str] = {}
        for step in steps:
            plan_id = f"p{step['id']}"
            events.append(Event(plan_id, Agent.PLANNER, NodeType.PLAN, output=step, meta={"depends_on": [prior_step_node]}, time=time))
            time += 1
            service = self.provider.select(step)
            selection_id = f"s{step['id']}"
            selection_status = NodeStatus.SUCCESS if service else NodeStatus.FAILED
            selection_output = service.__dict__ if service else {"error": "no service selected"}
            selection_meta: Dict[str, Any] = {"depends_on": [plan_id], "subtask": step}
            if not service:
                selection_meta["wrong_api"] = True
            events.append(Event(selection_id, Agent.SERVICE_PROVIDER, NodeType.SERVICE_SELECTION, output=selection_output, meta=selection_meta, status=selection_status, time=time))
            time += 1
            if not service:
                prior_step_node = selection_id
                continue
            selected_by_step[step["id"]] = service.record_id
            binding_id = f"b{step['id']}"
            binding_task = "" if inject_failure == "missing_param" and step["id"] == 1 else f"{task} {step['description']}"
            parameters, missing = self.executor.bind(service, binding_task, observations)
            binding_meta = {"depends_on": [selection_id], "service_id": service.record_id}
            if missing:
                binding_meta["missing_required"] = missing
            binding_status = NodeStatus.FAILED if missing else NodeStatus.SUCCESS
            events.append(Event(binding_id, Agent.EXECUTOR, NodeType.PARAMETER_BINDING, output=parameters, meta=binding_meta, status=binding_status, time=time))
            time += 1
            invocation_id = f"e{step['id']}"
            response = self.executor.invoke(service, parameters)
            invocation_meta = {"depends_on": [binding_id], "service_id": service.record_id, "parameters": parameters}
            if not response.get("ok"):
                invocation_meta["error"] = response.get("error", "execution failed")
            invocation_status = NodeStatus.SUCCESS if response.get("ok") else NodeStatus.FAILED
            if inject_failure == "tool_error" and step["id"] == 1:
                response = {"ok": False, "status_code": 503, "error": "service unavailable"}
                invocation_meta["error"] = response["error"]
                invocation_status = NodeStatus.FAILED
            events.append(Event(invocation_id, Agent.EXECUTOR, NodeType.API_INVOCATION, output=response, meta=invocation_meta, status=invocation_status, time=time))
            time += 1
            observation_id = f"o{step['id']}"
            observation_meta = {"depends_on": [invocation_id], "service_id": service.record_id}
            observation_status = NodeStatus.SUCCESS if response.get("ok") else NodeStatus.FAILED
            if not response.get("ok"):
                observation_meta["unrecovered"] = True
            events.append(Event(observation_id, Agent.EXECUTOR, NodeType.OBSERVATION, output=response, meta=observation_meta, status=observation_status, time=time))
            time += 1
            if response.get("ok"):
                observations.append(response)
            prior_step_node = observation_id
        answer = self.summarizer.summarize(task, observations)
        graph = ExecutionGraph.from_events(events)
        support_ids = [node_id for node_id, node in graph.nodes.items() if node.type == NodeType.OBSERVATION and node.status == NodeStatus.SUCCESS]
        claim_source = prior_step_node if prior_step_node in graph.nodes else None
        claim_ids = add_claim_nodes(graph, answer, source_node=claim_source, supported_by=support_ids)
        for claim_id in claim_ids:
            claim = graph.nodes[claim_id]
            events.append(Event(claim.id, claim.agent, claim.type, output=claim.content, meta=claim.meta, status=claim.status, time=claim.timestamp))
        return events, answer

    def reexecute_from_fault(self, context_graph: ExecutionGraph, faulty_node: str, action: RepairAction, invalid: set[str]) -> ExecutionGraph:
        graph = ExecutionGraph(dict(context_graph.nodes), set(context_graph.edges))
        node = graph.nodes.get(faulty_node)
        if node is None:
            return graph
        node.meta.pop("failure_types", None)
        node.meta["repair_action"] = action.value
        node.status = NodeStatus.REPAIRED
        if action == RepairAction.REBIND_PARAM and node.type == NodeType.PARAMETER_BINDING:
            service = self.repository.get(str(node.meta["service_id"]))
            task = _request_text(graph)
            prior_outputs = _successful_observations(graph)
            parameters, missing = self.executor.bind(service, task, prior_outputs)
            if missing:
                for parameter in missing:
                    parameters[parameter] = _fallback_value(parameter)
            node.content = {"input": node.content.get("input") if isinstance(node.content, dict) else None, "output": parameters}
            for key in ["missing_required", "parameter_error", "param_error", "error"]:
                node.meta.pop(key, None)
            invocation_id = f"repair_call_for_{node.id}"
            observation_id = f"repair_obs_for_{node.id}"
            response = self.executor.invoke(service, parameters)
            graph.add_node(_node(invocation_id, NodeType.API_INVOCATION, Agent.EXECUTOR, response, NodeStatus.SUCCESS if response.get("ok") else NodeStatus.FAILED, node.timestamp + 0.01, {"service_id": service.record_id, "parameters": parameters}))
            graph.add_edge(node.id, invocation_id, "params_trigger_call")
            graph.add_node(_node(observation_id, NodeType.OBSERVATION, Agent.EXECUTOR, response, NodeStatus.SUCCESS if response.get("ok") else NodeStatus.FAILED, node.timestamp + 0.02, {"service_id": service.record_id}))
            graph.add_edge(invocation_id, observation_id, "call_produces_observation")
            add_claim_nodes(graph, self.summarizer.summarize(task, [response] if response.get("ok") else []), supported_by=[observation_id] if response.get("ok") else [])
            return graph
        if action in {RepairAction.RETRY_TOOL, RepairAction.REPLACE_TOOL} and node.type == NodeType.API_INVOCATION:
            service = self.repository.get(str(node.meta["service_id"]))
            parameters = node.meta.get("parameters", {})
            response = self.executor.invoke(service, parameters)
            node.content = {"output": response}
            node.status = NodeStatus.SUCCESS if response.get("ok") else NodeStatus.FAILED
            node.meta.pop("error", None)
            observation_id = f"repair_obs_for_{node.id}"
            graph.add_node(_node(observation_id, NodeType.OBSERVATION, Agent.EXECUTOR, response, NodeStatus.SUCCESS if response.get("ok") else NodeStatus.FAILED, node.timestamp + 0.01, {"service_id": service.record_id}))
            graph.add_edge(node.id, observation_id, "call_produces_observation")
            add_claim_nodes(graph, self.summarizer.summarize(_request_text(graph), [response] if response.get("ok") else []), supported_by=[observation_id] if response.get("ok") else [])
            return graph
        if action == RepairAction.RESELECT_API and node.type == NodeType.SERVICE_SELECTION:
            subtask = node.meta.get("subtask", {})
            current = node.content.get("output", {}).get("record_id") if isinstance(node.content, dict) else None
            service = self.provider.select(subtask, exclude=[current] if current else [])
            if service:
                node.content = {"output": service.__dict__}
                node.meta["service_id"] = service.record_id
                node.status = NodeStatus.REPAIRED
                node.meta.pop("wrong_api", None)
            return graph
        if action in {RepairAction.REVISE_ANSWER, RepairAction.MARK_UNCERTAIN} and node.type == NodeType.CLAIM:
            node.meta.pop("unsupported", None)
            node.meta["supported"] = True
            if action == RepairAction.REVISE_ANSWER:
                node.meta["removed"] = True
            if action == RepairAction.MARK_UNCERTAIN:
                node.meta["uncertain"] = True
            return graph
        return graph


def infer_domain(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["weather", "天气", "forecast"]):
        return "weather"
    if any(token in lower for token in ["stock", "finance", "quote", "股票"]):
        return "finance"
    if any(token in lower for token in ["translate", "translation", "翻译"]):
        return "translation"
    return "general"


def _request_text(graph: ExecutionGraph) -> str:
    for node in graph.nodes.values():
        if node.type == NodeType.REQUEST:
            if isinstance(node.content, dict):
                return str(node.content.get("output") or node.content)
            return str(node.content)
    return ""


def _successful_observations(graph: ExecutionGraph) -> List[Mapping[str, Any]]:
    outputs = []
    for node in graph.nodes.values():
        if node.type == NodeType.OBSERVATION and node.status == NodeStatus.SUCCESS:
            content = node.content.get("output") if isinstance(node.content, dict) else node.content
            if isinstance(content, Mapping):
                outputs.append(content)
    return outputs


def _fallback_value(parameter: str) -> str:
    values = {"city": "Beijing", "symbol": "AAPL", "text": "hello", "target_language": "English"}
    return values.get(parameter, f"{parameter}_value")


def _node(node_id: str, node_type: NodeType, agent: Agent, output: Any, status: NodeStatus, timestamp: float, meta: Dict[str, Any]):
    from .core import Node

    return Node(node_id, node_type, {"output": output}, agent, status, timestamp, meta)


def final_answer_from_graph(graph: ExecutionGraph) -> str:
    claims = []
    for node in sorted(graph.nodes.values(), key=lambda item: (item.timestamp, item.id)):
        if node.type == NodeType.CLAIM and node.status in {NodeStatus.SUCCESS, NodeStatus.REPAIRED, NodeStatus.UNCERTAIN}:
            if node.meta.get("removed"):
                continue
            content = node.content.get("claim") if isinstance(node.content, dict) else None
            if content is None and isinstance(node.content, dict):
                output = node.content.get("output")
                content = output.get("claim") if isinstance(output, dict) else output
            if content:
                claims.append(str(content))
    return " ".join(claims)
