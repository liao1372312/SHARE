from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import LLMProfile, load_llm_profiles


SYSTEM_PROMPT = (
    "You are a coding agent working in /testbed on a SWE-bench task. Diagnose and implement a minimal fix. "
    "You may inspect files and run tests through one shell command per turn, or make one exact text replacement. "
    "Do not use network access, do not commit, "
    "and do not search git history for the answer. Edit existing files surgically: never replace an entire source file "
    "with echo, cat, placeholders, or ellipses. Inspect the resulting git diff and test after the final edit. Do not "
    "use sed insertion with escaped newlines for multi-line edits; prefer a short Python pathlib script that replaces "
    "an exact existing text block and fails if that block is not found. Do not "
    "return final while merely describing work that remains to be done. Respond only as JSON: "
    '{"action":"shell","command":"...","reason":"..."} or '
    '{"action":"replace","path":"relative/path.py","old":"exact existing text","new":"replacement text","reason":"..."} or '
    '{"action":"final","summary":"..."}. Keep commands non-interactive.'
)
USER_PROMPT_TEMPLATE = "Repository: {repository}\nBase commit: {base_commit}\n\nIssue:\n{task}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_instance(path: Path, instance_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"input line {line_number} is not JSON") from error
            if isinstance(value, dict) and value.get("workflow_id") == instance_id:
                if value.get("synthetic") is not False or value.get("source") != "swe-bench":
                    raise ValueError("workflow is not a real SWE-bench record")
                metadata = value.get("metadata")
                if not isinstance(metadata, Mapping) or metadata.get("gold_patch_withheld") is not True:
                    raise ValueError("workflow does not certify that the gold patch is withheld")
                return value
    raise ValueError(f"instance is not in the model-input workflows: {instance_id}")


def read_spec(path: Path, instance_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("instance_id") == instance_id:
                    return value
    raise ValueError(f"instance is not in pilot execution specs: {instance_id}")


def extract_json_object(content: str) -> dict[str, Any]:
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("model response is not one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def run(command: list[str], *, timeout: int = 120, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode != 0:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or f"command failed: {command[0]}")
    return result


def validate_tracked_patch(container_name: str, base_commit: str) -> dict[str, Any] | None:
    diff_check = run(
        ["docker", "exec", "-w", "/testbed", container_name, "git", "diff", "--check", base_commit, "--"]
    )
    if diff_check.returncode != 0:
        return {
            "kind": "git_diff_check_failed",
            "stdout": diff_check.stdout[-4000:],
            "stderr": diff_check.stderr[-4000:],
        }
    changed = run(
        [
            "docker",
            "exec",
            "-w",
            "/testbed",
            container_name,
            "git",
            "diff",
            "--diff-filter=d",
            "--name-only",
            "-z",
            base_commit,
            "--",
            "*.py",
        ],
        check=True,
    )
    python_files = [name for name in changed.stdout.split("\0") if name]
    for name in python_files:
        compile_result = run(
            ["docker", "exec", "-w", "/testbed", container_name, "python", "-m", "py_compile", name]
        )
        if compile_result.returncode != 0:
            return {
                "kind": "python_compile_failed",
                "file": name,
                "stdout": compile_result.stdout[-4000:],
                "stderr": compile_result.stderr[-4000:],
            }
    return None


def call_model(profile: LLMProfile, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": profile.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if profile.reasoning_effort is not None:
        body["reasoning_effort"] = profile.reasoning_effort
    request = urllib.request.Request(
        profile.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {profile.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ValueError(f"LLM request failed with HTTP {error.code}: {detail[:500]}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ValueError(f"LLM request failed: {error}") from error
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("LLM response lacks choices[0]")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response has no text content")
    audit = {
        "response_id": value.get("id"),
        "created": value.get("created"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": value.get("usage"),
    }
    return content, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one auditable SWE-bench AI patch in an isolated container.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--profile", choices=("primary", "backbone_2", "backbone_3"), default="primary")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--command-timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.max_turns <= 30:
        parser.error("--max-turns must be between 1 and 30")
    if not 256 <= args.max_output_tokens <= 8192:
        parser.error("--max-output-tokens must be between 256 and 8192")
    if not 10 <= args.api_timeout <= 600 or not 10 <= args.command_timeout <= 900:
        parser.error("timeouts are outside their supported ranges")
    return args


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite AI generation directory: {args.out}")
    try:
        workflow = read_instance(args.workflows, args.instance)
        spec = read_spec(args.pilot / "execution_specs.jsonl", args.instance)
        profiles = {profile.name: profile for profile in load_llm_profiles(args.env)}
        profile = profiles[args.profile]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    image = spec.get("image")
    base_commit = workflow.get("context", {}).get("base_commit")
    repository = workflow.get("context", {}).get("repository")
    if not all(isinstance(value, str) and value for value in (image, base_commit, repository, workflow.get("task"))):
        raise SystemExit("workflow or execution spec lacks required public model input")

    args.out.mkdir(parents=True)
    transcript_path = args.out / "transcript.jsonl"
    transcript_path.touch()
    container_name = "share-ai-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", args.instance).lower()
    try:
        image_info = run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], check=True)
    except (ValueError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"Docker image preflight failed: {error}") from error
    image_id = image_info.stdout.strip()
    started_at = datetime.now(timezone.utc).isoformat()
    create = run(
        ["docker", "run", "--detach", "--name", container_name, "--network", "none", "--entrypoint", "/bin/bash", image, "-lc", "sleep infinity"],
        timeout=120,
    )
    if create.returncode != 0:
        raise SystemExit(create.stderr.strip() or "could not start isolated SWE-bench container")

    transcript: list[dict[str, Any]] = []
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(repository=repository, base_commit=base_commit, task=workflow["task"]),
        },
    ]
    final_summary = None
    error_message = None
    baseline_diff_sha256 = None
    baseline_reset = False
    clean_baseline_verified = False
    try:
        head = run(["docker", "exec", "-w", "/testbed", container_name, "git", "rev-parse", "HEAD"], check=True)
        if head.stdout.strip() != base_commit:
            raise ValueError("container HEAD does not match the bound base commit")
        baseline_diff = run(
            ["docker", "exec", "-w", "/testbed", container_name, "git", "diff", "--binary", base_commit, "--"],
            check=True,
        ).stdout
        baseline_diff_sha256 = sha256_bytes(baseline_diff.encode("utf-8"))
        if baseline_diff:
            # Evaluation images may contain tracked setup edits. They are not
            # model output and must not contaminate the generated patch.
            run(
                ["docker", "exec", "-w", "/testbed", container_name, "git", "reset", "--hard", base_commit],
                check=True,
            )
            baseline_reset = True
        clean_check = run(
            ["docker", "exec", "-w", "/testbed", container_name, "git", "diff", "--quiet", base_commit, "--"]
        )
        if clean_check.returncode != 0:
            raise ValueError("could not establish a clean tracked baseline before model execution")
        clean_baseline_verified = True
        with transcript_path.open("w", encoding="utf-8") as transcript_file:
            for turn in range(1, args.max_turns + 1):
                content, response_audit = call_model(profile, messages, args.max_output_tokens, args.api_timeout)
                try:
                    decision = extract_json_object(content)
                except ValueError as error:
                    decision = {"action": "invalid", "error": str(error), "raw_content": content[:4000]}
                record: dict[str, Any] = {"turn": turn, "response": decision, "response_audit": response_audit}
                action = decision.get("action")
                if action == "final":
                    final_summary = str(decision.get("summary", ""))
                    diff_check = run(
                        ["docker", "exec", "-w", "/testbed", container_name, "git", "diff", "--quiet", base_commit, "--"]
                    )
                    if diff_check.returncode not in (0, 1):
                        raise ValueError(diff_check.stderr.strip() or "could not inspect the model patch before finalizing")
                    validation_error = validate_tracked_patch(container_name, base_commit) if diff_check.returncode == 1 else None
                    if (diff_check.returncode == 0 or validation_error is not None) and turn < args.max_turns:
                        observation = canonical_json(
                            {
                                "patch_empty": diff_check.returncode == 0,
                                "patch_validation": validation_error,
                                "instruction": (
                                    "No tracked patch exists yet. Implement the fix and verify it before returning final."
                                    if diff_check.returncode == 0
                                    else "The tracked patch is invalid. Repair it, inspect git diff, and verify it before returning final."
                                ),
                            }
                        )
                        record["observation"] = json.loads(observation)
                        transcript.append(record)
                        transcript_file.write(canonical_json(record) + "\n")
                        transcript_file.flush()
                        messages.extend(
                            [
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": "Tool result:\n" + observation},
                            ]
                        )
                        continue
                    transcript.append(record)
                    transcript_file.write(canonical_json(record) + "\n")
                    transcript_file.flush()
                    break
                tool_command: list[str] | None = None
                if action == "shell" and isinstance(decision.get("command"), str) and decision["command"].strip():
                    tool_command = [
                        "docker",
                        "exec",
                        "-w",
                        "/testbed",
                        container_name,
                        "/bin/bash",
                        "-lc",
                        decision["command"],
                    ]
                elif action == "replace":
                    path_value = decision.get("path")
                    old_value = decision.get("old")
                    new_value = decision.get("new")
                    edit_path = PurePosixPath(path_value) if isinstance(path_value, str) and path_value else None
                    if (
                        edit_path is None
                        or edit_path.is_absolute()
                        or ".." in edit_path.parts
                        or not isinstance(old_value, str)
                        or not old_value
                        or not isinstance(new_value, str)
                    ):
                        observation = "Invalid replace action. Use a relative path, non-empty exact old text, and string new text."
                    else:
                        replace_script = (
                            "from pathlib import Path; import sys; "
                            "p=Path(sys.argv[1]); old=sys.argv[2]; new=sys.argv[3]; text=p.read_text(); "
                            "count=text.count(old); "
                            "assert count == 1, f'expected exact old text once, found {count}'; "
                            "p.write_text(text.replace(old, new, 1))"
                        )
                        tool_command = [
                            "docker",
                            "exec",
                            "-w",
                            "/testbed",
                            container_name,
                            "python",
                            "-c",
                            replace_script,
                            str(edit_path),
                            old_value,
                            new_value,
                        ]
                else:
                    observation = "Invalid response. Return exactly one shell, replace, or final JSON object."
                if tool_command is not None:
                    try:
                        result = run(tool_command, timeout=args.command_timeout)
                        observation_value: dict[str, Any] = {
                            "exit_code": result.returncode,
                            "stdout": result.stdout[-12000:],
                            "stderr": result.stderr[-12000:],
                            "output_truncated": len(result.stdout) > 12000 or len(result.stderr) > 12000,
                        }
                        validation_error = validate_tracked_patch(container_name, base_commit)
                        if validation_error is not None:
                            observation_value["patch_validation"] = validation_error
                            observation_value["instruction"] = (
                                "The tracked patch is invalid. Repair it before continuing, then inspect git diff and rerun a relevant test. "
                                "For a multi-line edit, use Python pathlib exact text replacement instead of sed escaped-newline insertion."
                            )
                        observation = canonical_json(observation_value)
                    except subprocess.TimeoutExpired:
                        observation = canonical_json({"timed_out": True, "timeout_seconds": args.command_timeout})
                record["observation"] = json.loads(observation) if observation.startswith("{") else observation
                transcript.append(record)
                transcript_file.write(canonical_json(record) + "\n")
                transcript_file.flush()
                messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": "Tool result:\n" + observation}])
            patch_result = run(["docker", "exec", "-w", "/testbed", container_name, "git", "diff", "--binary", base_commit, "--"], timeout=120, check=True)
            final_validation_error = validate_tracked_patch(container_name, base_commit)
            if final_validation_error is not None:
                error_message = "final tracked patch validation failed: " + canonical_json(final_validation_error)
    except (ValueError, subprocess.TimeoutExpired) as error:
        error_message = str(error)
        patch_result = subprocess.CompletedProcess([], 1, "", error_message)
    finally:
        run(["docker", "rm", "--force", container_name], timeout=120)

    patch = patch_result.stdout
    patch_path = args.out / "model.patch"
    patch_path.write_text(patch, encoding="utf-8")
    prediction = {
        "instance_id": args.instance,
        "model_name_or_path": profile.model,
        "model_patch": patch,
    }
    prediction_path = args.out / "prediction.jsonl"
    prediction_path.write_text(canonical_json(prediction) + "\n", encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_ai_patch_generation_artifact",
        "instance_id": args.instance,
        "branch": "ai",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "profile": args.profile,
        "model": profile.model,
        "api_root_sha256": sha256_bytes(profile.base_url.rstrip("/").encode("utf-8")),
        "image": image,
        "image_id": image_id,
        "base_commit": base_commit,
        "initial_tracked_diff_sha256": baseline_diff_sha256,
        "initial_tracked_changes_reset": baseline_reset,
        "clean_tracked_baseline_verified": clean_baseline_verified,
        "container_network": "none",
        "gold_patch_withheld": True,
        "test_patch_withheld": True,
        "turns_used": len(transcript),
        "protocol": {
            "schema_version": 1,
            "generator_source_sha256": sha256_file(Path(__file__).resolve()),
            "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "user_prompt_template_sha256": sha256_bytes(USER_PROMPT_TEMPLATE.encode("utf-8")),
            "workflows_sha256": sha256_file(args.workflows),
            "execution_specs_sha256": sha256_file(args.pilot / "execution_specs.jsonl"),
            "pilot_manifest_sha256": sha256_file(args.pilot / "pilot_manifest.json"),
            "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens,
            "api_timeout_seconds": args.api_timeout,
            "command_timeout_seconds": args.command_timeout,
            "temperature": 0.0,
            "profile_reasoning_effort": profile.reasoning_effort,
        },
        "final_summary": final_summary,
        "generation_error": error_message,
        "patch_empty": not bool(patch.strip()),
        "files": {
            "model_patch": {"path": patch_path.name, "sha256": sha256_bytes(patch_path.read_bytes())},
            "prediction": {"path": prediction_path.name, "sha256": sha256_bytes(prediction_path.read_bytes())},
            "transcript": {"path": transcript_path.name, "sha256": sha256_bytes(transcript_path.read_bytes())},
        },
        "evidence_status": "generated_ai_patch_awaiting_swebench_execution",
    }
    artifact_path = args.out / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "model": profile.model, "turns": len(transcript), "patch_empty": artifact["patch_empty"], "error": error_message}, indent=2))
    if error_message or artifact["patch_empty"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
