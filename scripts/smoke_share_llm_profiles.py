from __future__ import annotations

import argparse
import hashlib
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import LLMProfile, load_llm_profiles


MAX_REQUESTS_PER_PROFILE = 2
MAX_COMPLETION_TOKENS = 128
PROBES = (
    ("connectivity", "Return only the JSON object {\"status\":\"ok\",\"probe\":\"connectivity\"}."),
    ("structured_response", "Return only the JSON object {\"status\":\"ok\",\"probe\":\"structured_response\"}."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a capped, non-paper-result smoke test for the three configured SHARE LLM profiles."
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, default=Path("results/share_llm_profile_smoke"))
    parser.add_argument("--requests-per-profile", type=int, default=MAX_REQUESTS_PER_PROFILE)
    parser.add_argument("--max-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--profile",
        action="append",
        choices=("primary", "backbone_2", "backbone_3"),
        help="Run only a named profile. Repeat the option to select multiple profiles.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a prior smoke artifact at --out.")
    args = parser.parse_args()
    if not 1 <= args.requests_per_profile <= MAX_REQUESTS_PER_PROFILE:
        parser.error(f"--requests-per-profile must be between 1 and {MAX_REQUESTS_PER_PROFILE}")
    if not 1 <= args.max_tokens <= MAX_COMPLETION_TOKENS:
        parser.error(f"--max-tokens must be between 1 and {MAX_COMPLETION_TOKENS}")
    if not 1 <= args.timeout_seconds <= 90:
        parser.error("--timeout-seconds must be between 1 and 90")
    return args


def select_profiles(profiles: tuple[LLMProfile, ...], selected_names: list[str] | None) -> tuple[LLMProfile, ...]:
    if not selected_names:
        return profiles
    selected = set(selected_names)
    return tuple(profile for profile in profiles if profile.name in selected)


def request_payload(profile: LLMProfile, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": profile.model,
        "messages": [
            {"role": "system", "content": "You are a concise evaluator for a non-production SHARE smoke test."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if profile.reasoning_effort is not None:
        payload["reasoning_effort"] = profile.reasoning_effort
    return payload


def response_content(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def json_response(content: str | None) -> bool:
    if content is None:
        return False
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return False
    return True


def transport_error_kind(error: urllib.error.URLError | TimeoutError | OSError) -> str:
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "tls_certificate_error"
    if isinstance(reason, socket.gaierror):
        return "dns_error"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(reason, ConnectionRefusedError):
        return "connection_refused"
    return "transport_error"


def run_probe(profile: LLMProfile, probe_name: str, prompt: str, max_tokens: int, timeout_seconds: float) -> dict[str, Any]:
    body = request_payload(profile, prompt, max_tokens)
    request = urllib.request.Request(
        profile.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {profile.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as raw_response:
            response = json.loads(raw_response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return {
            "probe": probe_name,
            "status": "error",
            "error_kind": "http_error",
            "http_status": error.code,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }
    except urllib.error.URLError as error:
        return {
            "probe": probe_name,
            "status": "error",
            "error_kind": transport_error_kind(error),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }
    except json.JSONDecodeError:
        return {
            "probe": probe_name,
            "status": "error",
            "error_kind": "invalid_json_body",
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }
    except (TimeoutError, OSError) as error:
        return {
            "probe": probe_name,
            "status": "error",
            "error_kind": transport_error_kind(error),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }

    content = response_content(response)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "probe": probe_name,
        "status": "success" if content is not None else "error",
        "error_kind": None if content is not None else "invalid_chat_completion_response",
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "response_id_configured": bool(response.get("id")),
        "content_is_json": json_response(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content is not None else None,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


def profile_result(profile: LLMProfile, requests_per_profile: int, max_tokens: int, timeout_seconds: float) -> dict[str, Any]:
    probes = []
    for index, (probe_name, prompt) in enumerate(PROBES[:requests_per_profile]):
        if probes and probes[-1]["status"] != "success":
            probes.append({"probe": probe_name, "status": "skipped", "reason": "previous_probe_failed"})
            continue
        probes.append(run_probe(profile, probe_name, prompt, max_tokens, timeout_seconds))
    return {
        "name": profile.name,
        "model": profile.model,
        "reasoning_effort": profile.reasoning_effort,
        "probes": probes,
        "successful_probes": sum(probe["status"] == "success" for probe in probes),
        "requested_probes": requests_per_profile,
    }


def main() -> None:
    args = parse_args()
    profiles = select_profiles(load_llm_profiles(args.env), args.profile)
    if not profiles:
        raise SystemExit("No configured profiles matched --profile")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.out}. Use --overwrite to replace this smoke artifact.")
    args.out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "kind": "share_llm_profile_smoke",
        "purpose": "capped connectivity and response-shape check; not paper evidence or a real paired-outcome dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "caps": {
            "requests_per_profile": args.requests_per_profile,
            "max_tokens_per_request": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
        },
        "profiles": [
            profile_result(profile, args.requests_per_profile, args.max_tokens, args.timeout_seconds)
            for profile in profiles
        ],
    }
    report["successful_probes"] = sum(profile["successful_probes"] for profile in report["profiles"])
    report["requested_probes"] = len(profiles) * args.requests_per_profile
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "profiles": len(profiles),
                "successful_probes": report["successful_probes"],
                "requested_probes": report["requested_probes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
