from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backbone import MultiAgentComposer
from .pipeline import FLAREConfig
from .services import ServiceRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full FLARE multi-agent service-composition prototype.")
    parser.add_argument("task", help="natural-language service composition task")
    parser.add_argument("--catalog", type=Path, help="optional JSONL service catalog")
    parser.add_argument("--catalog-limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--inject-failure", choices=["missing_param", "tool_error"], default=None)
    args = parser.parse_args()
    repository = ServiceRepository.from_jsonl(args.catalog, args.catalog_limit) if args.catalog else ServiceRepository.builtin()
    composer = MultiAgentComposer(repository=repository, config=FLAREConfig(repair_budget=args.budget))
    output = composer.run(args.task, inject_failure=args.inject_failure)
    print(json.dumps({
        "initial_answer": output.answer,
        "final_answer": output.final_answer,
        "repaired": output.result.repaired,
        "initial_failures": sorted(output.result.initial_failures),
        "final_failures": sorted(output.result.final_failures),
        "iterations": [
            {
                "action": iteration.action.value,
                "localization": None if iteration.localization is None else {
                    "node_id": iteration.localization.node_id,
                    "agent": iteration.localization.agent,
                    "node_type": iteration.localization.node_type,
                    "failure_type": iteration.localization.failure_type,
                    "affected": sorted(iteration.localization.affected),
                },
            }
            for iteration in output.result.iterations
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
