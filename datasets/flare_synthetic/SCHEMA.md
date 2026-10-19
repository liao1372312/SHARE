# FLARE Synthetic Dataset Schema

All files in this directory are synthetic and generated for controlled FLARE experiments.

Each JSONL row contains:

- `id`: stable sample id.
- `dataset`: one of `public_api`, `enterprise_service`, `mixed_stress`.
- `synthetic`: always `true`.
- `task`: natural-language service composition request.
- `api_pool_size`: nominal API pool size used for paper statistics.
- `candidate_api_ids`: sampled candidate APIs.
- `events`: observable execution records matching `rt=<id,agent,type,input,output,meta,status,time>`.
- `repair_label.failed`: whether a primary failure is injected.
- `repair_label.faulty_agent`: planner, service_provider, executor, external_tool, supervisor, or summarizer.
- `repair_label.failure_type`: planning_failure, service_selection_failure, parameter_binding_failure, tool_execution_failure, supervisor_failure, or answer_grounding_failure.
- `repair_label.faulty_decision_node`: gold faulty decision node id.
- `repair_label.expected_repair_action`: AddSubtask, Replan, ReselectAPI, RebindParam, RetryTool, ReplaceTool, ReviseRepair, ReviseAnswer, MarkUncertain, or Terminate-compatible action.
- `repair_label.affected_slice`: downstream nodes to invalidate/re-execute.
- `metadata`: generator provenance; never claim these are real user traces.
