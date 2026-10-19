from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_share_swebench_ai_patch import (
    call_model,
    canonical_json,
    read_instance,
    read_spec,
    run,
    sha256_bytes,
    sha256_file,
    validate_tracked_patch,
)
from share.llm_profiles import load_llm_profiles


SYSTEM_PROMPT = (
    "You are the second-stage verifier for a coding agent in /testbed. The current git diff is an untrusted first "
    "draft for the supplied SWE-bench issue. Inspect the implementation and surrounding code, run relevant repository "
    "tests, and revise the patch when necessary. You do not have benchmark evaluation feedback or a gold patch. "
    "Prefer the smallest general fix; do not hard-code tests. Before final, run at least one relevant test command and "
    "inspect git diff after your most recent edit. Do not use network access, git history, commits, or destructive "
    "repository resets. Respond only as one JSON object: "
    '{"action":"shell","command":"...","reason":"..."}, '
    '{"action":"replace","path":"relative/path.py","old":"exact existing text","new":"replacement text","reason":"..."}, or '
    '{"action":"final","summary":"..."}. Use one non-interactive command or one exact replacement per turn.'
)
USER_PROMPT_TEMPLATE = (
    "Repository: {repository}\nBase commit: {base_commit}\n\nIssue:\n{task}\n\n"
    "A first-stage agent patch is already present in the working tree. Treat it as untrusted: inspect, test, and refine it."
)
TEST_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:python\s+-m\s+)?(?:pytest|unittest|tox|nox)(?:\s|$)|"
    r"(?:^|[;&|]\s*)(?:make|npm|yarn|pnpm|cargo|go)\s+(?:test|check)(?:\s|$)|"
    r"(?:^|[;&|]\s*)python(?:\s+-\S+)*\s+\S*tests?\S*(?:\s|$)|"
    r"(?:^|[;&|]\s*)\.?/?\S*tests?\S*(?:\s|$)"
)
DIFF_COMMAND = re.compile(r"(?:^|[;&|]\s*)git\s+(?:diff|status)(?:\s|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uniform second-stage verification of a frozen SWE-bench patch.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--profile", default="primary")
    parser.add_argument("--initial-patch", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--command-timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_turns <= 30:
        parser.error("--max-turns must be between 1 and 30")
    if not 256 <= args.max_output_tokens <= 8192:
        parser.error("--max-output-tokens must be between 256 and 8192")
    if not args.initial_patch.is_file() or not args.initial_patch.read_text(encoding="utf-8").strip():
        parser.error("--initial-patch must be a non-empty file")
    return args


def tracked_diff(container: str, base_commit: str) -> str:
    return run(
        ["docker", "exec", "-w", "/testbed", container, "git", "diff", "--binary", base_commit, "--"],
        check=True,
    ).stdout


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite refinement directory: {args.out}")
    try:
        workflow = read_instance(args.workflows, args.instance)
        spec = read_spec(args.pilot / "execution_specs.jsonl", args.instance)
        profiles = {profile.name: profile for profile in load_llm_profiles(args.env)}
        profile = profiles[args.profile]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    image = spec.get("image")
    context = workflow.get("context", {})
    base_commit = context.get("base_commit")
    repository = context.get("repository")
    if not all(isinstance(value, str) and value for value in (image, base_commit, repository, workflow.get("task"))):
        raise SystemExit("workflow or execution spec lacks required public model input")

    args.out.mkdir(parents=True)
    transcript_path = args.out / "transcript.jsonl"
    transcript_path.touch()
    container = "share-refine-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", args.instance).lower()
    started_at = datetime.now(timezone.utc).isoformat()
    initial_patch_sha256 = sha256_file(args.initial_patch)
    final_summary: str | None = None
    error_message: str | None = None
    transcript: list[dict[str, Any]] = []
    test_verified = False
    diff_inspected = False
    image_id = ""
    patch_result = subprocess.CompletedProcess([], 1, "", "not started")
    try:
        image_id = run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], check=True).stdout.strip()
        create = run(
            ["docker", "run", "--detach", "--name", container, "--network", "none", "--entrypoint", "/bin/bash", image, "-lc", "sleep infinity"],
            timeout=120,
        )
        if create.returncode != 0:
            raise ValueError(create.stderr.strip() or "could not start refinement container")
        head = run(["docker", "exec", "-w", "/testbed", container, "git", "rev-parse", "HEAD"], check=True)
        if head.stdout.strip() != base_commit:
            raise ValueError("container HEAD does not match the bound base commit")
        run(["docker", "exec", "-w", "/testbed", container, "git", "reset", "--hard", base_commit], check=True)
        container_patch = "/tmp/share-initial.patch"
        copy = run(["docker", "cp", str(args.initial_patch.resolve()), f"{container}:{container_patch}"])
        if copy.returncode != 0:
            raise ValueError(copy.stderr.strip() or "could not copy initial patch")
        apply = run(["docker", "exec", "-w", "/testbed", container, "git", "apply", "--whitespace=nowarn", container_patch])
        if apply.returncode != 0:
            raise ValueError(apply.stderr.strip() or "initial patch did not apply")
        if validate_tracked_patch(container, base_commit) is not None:
            raise ValueError("initial patch failed tracked-patch validation")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(repository=repository, base_commit=base_commit, task=workflow["task"])
                + f"\nYou have at most {args.max_turns} turns. Reserve the final two turns for post-edit testing and diff inspection.",
            },
        ]
        with transcript_path.open("w", encoding="utf-8") as transcript_file:
            for turn in range(1, args.max_turns + 1):
                content, response_audit = call_model(profile, messages, args.max_output_tokens, args.api_timeout)
                try:
                    decision = json.loads(content.strip())
                    if not isinstance(decision, Mapping):
                        raise ValueError("response is not an object")
                except (json.JSONDecodeError, ValueError) as error:
                    decision = {"action": "invalid", "error": str(error), "raw_content": content[:4000]}
                record: dict[str, Any] = {"turn": turn, "response": decision, "response_audit": response_audit}
                action = decision.get("action")
                if action == "final":
                    missing = []
                    if not test_verified:
                        missing.append("a successful test command after the latest edit")
                    if not diff_inspected:
                        missing.append("git diff/status inspection after the latest edit")
                    validation_error = validate_tracked_patch(container, base_commit)
                    if validation_error is not None:
                        missing.append("a valid tracked patch")
                    if missing:
                        observation_value = {"final_rejected": True, "missing": missing, "instruction": "Complete the missing verification before final."}
                    else:
                        final_summary = str(decision.get("summary", ""))
                        record["observation"] = {"final_accepted": True}
                        transcript.append(record)
                        transcript_file.write(canonical_json(record) + "\n")
                        transcript_file.flush()
                        break
                elif action == "replace":
                    path_value, old_value, new_value = decision.get("path"), decision.get("old"), decision.get("new")
                    edit_path = PurePosixPath(path_value) if isinstance(path_value, str) and path_value else None
                    if edit_path is None or edit_path.is_absolute() or ".." in edit_path.parts or not isinstance(old_value, str) or not old_value or not isinstance(new_value, str):
                        observation_value = {"error": "invalid exact replacement"}
                    else:
                        script = (
                            "from pathlib import Path; import sys; p=Path(sys.argv[1]); old=sys.argv[2]; new=sys.argv[3]; "
                            "text=p.read_text(); count=text.count(old); assert count == 1, f'expected once, found {count}'; "
                            "p.write_text(text.replace(old,new,1))"
                        )
                        result = run(["docker", "exec", "-w", "/testbed", container, "python", "-c", script, str(edit_path), old_value, new_value], timeout=args.command_timeout)
                        observation_value = {"exit_code": result.returncode, "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:]}
                        if result.returncode == 0:
                            test_verified = False
                            diff_inspected = False
                elif action == "shell" and isinstance(decision.get("command"), str) and decision["command"].strip():
                    command = decision["command"]
                    before = sha256_bytes(tracked_diff(container, base_commit).encode("utf-8"))
                    try:
                        result = run(["docker", "exec", "-w", "/testbed", container, "/bin/bash", "-lc", command], timeout=args.command_timeout)
                        after = sha256_bytes(tracked_diff(container, base_commit).encode("utf-8"))
                        if after != before:
                            test_verified = False
                            diff_inspected = False
                        if result.returncode == 0 and TEST_COMMAND.search(command):
                            test_verified = True
                        if result.returncode == 0 and DIFF_COMMAND.search(command):
                            diff_inspected = True
                        observation_value = {
                            "exit_code": result.returncode,
                            "stdout": result.stdout[-12000:],
                            "stderr": result.stderr[-12000:],
                            "test_verified_after_latest_edit": test_verified,
                            "diff_inspected_after_latest_edit": diff_inspected,
                            "output_truncated": len(result.stdout) > 12000 or len(result.stderr) > 12000,
                        }
                    except subprocess.TimeoutExpired:
                        observation_value = {"timed_out": True, "timeout_seconds": args.command_timeout}
                else:
                    observation_value = {"error": "return exactly one shell, replace, or final JSON object"}
                record["observation"] = observation_value
                transcript.append(record)
                transcript_file.write(canonical_json(record) + "\n")
                transcript_file.flush()
                observation = canonical_json(observation_value)
                messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": "Tool result:\n" + observation}])
            if final_summary is None:
                error_message = "turn cap reached without an admissible verified final action"
            patch_result = subprocess.CompletedProcess([], 0, tracked_diff(container, base_commit), "")
            if not patch_result.stdout.strip():
                error_message = error_message or "refinement produced an empty tracked patch"
    except (ValueError, subprocess.TimeoutExpired, OSError) as error:
        error_message = str(error)
    finally:
        run(["docker", "rm", "--force", container], timeout=120)

    patch_path = args.out / "model.patch"
    patch_path.write_text(patch_result.stdout, encoding="utf-8")
    prediction_path = args.out / "prediction.jsonl"
    prediction = {"instance_id": args.instance, "model_name_or_path": profile.model + "+verified-refinement", "model_patch": patch_result.stdout}
    prediction_path.write_text(canonical_json(prediction) + "\n", encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_verified_refinement_artifact",
        "instance_id": args.instance,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "profile": args.profile,
        "model": profile.model,
        "image": image,
        "image_id": image_id,
        "base_commit": base_commit,
        "container_network": "none",
        "gold_patch_withheld": True,
        "test_patch_withheld": True,
        "benchmark_evaluation_feedback_used": False,
        "initial_patch": {"path": str(args.initial_patch), "sha256": initial_patch_sha256},
        "turns_used": len(transcript),
        "test_verified_after_latest_edit": test_verified,
        "diff_inspected_after_latest_edit": diff_inspected,
        "final_summary": final_summary,
        "generation_error": error_message,
        "patch_empty": not bool(patch_result.stdout.strip()),
        "protocol": {
            "generator_source_sha256": sha256_file(Path(__file__).resolve()),
            "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "user_prompt_template_sha256": sha256_bytes(USER_PROMPT_TEMPLATE.encode("utf-8")),
            "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens,
            "temperature": 0.0,
            "profile_reasoning_effort": profile.reasoning_effort,
        },
        "files": {
            "model_patch": {"path": patch_path.name, "sha256": sha256_file(patch_path)},
            "prediction": {"path": prediction_path.name, "sha256": sha256_file(prediction_path)},
            "transcript": {"path": transcript_path.name, "sha256": sha256_file(transcript_path)},
        },
    }
    (args.out / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "turns": len(transcript), "patch_empty": artifact["patch_empty"], "error": error_message}, indent=2))
    if error_message or artifact["patch_empty"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
