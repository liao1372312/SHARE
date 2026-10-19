from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import load_annotation_profile
from share.proposal_annotation import serve_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local blinded SHARE proposal-assessment study.")
    parser.add_argument("--study", type=Path, required=True, help="SQLite study database created by create_share_proposal_annotation_study.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-network", action="store_true", help="Required for non-loopback binding.")
    parser.add_argument("--tls-cert", type=Path, default=None, help="PEM certificate chain for direct HTTPS.")
    parser.add_argument("--tls-key", type=Path, default=None, help="PEM private key for direct HTTPS.")
    parser.add_argument("--llm-reference-env", type=Path, default=None, help="Optional .env with LLM_ANNOTATION_* for transient in-page references.")
    parser.add_argument("--calibration-manifest", type=Path, default=None, help="Required with --llm-reference-env; reserves pure-human calibration cases.")
    parser.add_argument("--llm-reference-max-calls", type=int, default=2, help="Maximum transient LLM references per assignment.")
    parser.add_argument("--llm-reference-max-output-tokens", type=int, default=1024)
    parser.add_argument("--llm-reference-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--allow-insecure-network",
        action="store_true",
        help="Allow remote plain HTTP only on an explicitly trusted private network. HTTPS is the default requirement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.llm_reference_env is None) != (args.calibration_manifest is None):
        raise SystemExit("--llm-reference-env and --calibration-manifest must be supplied together")
    if args.llm_reference_max_calls < 1 or not 1 <= args.llm_reference_max_output_tokens <= 2048 or not 1 <= args.llm_reference_timeout_seconds <= 90:
        raise SystemExit("LLM reference limits are invalid")
    profile = load_annotation_profile(args.llm_reference_env) if args.llm_reference_env is not None else None
    server = serve_study(
        args.study,
        args.host,
        args.port,
        allow_network=args.allow_network,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        allow_insecure_network=args.allow_insecure_network,
        llm_reference_profile=profile,
        calibration_manifest_path=args.calibration_manifest,
        llm_reference_max_calls=args.llm_reference_max_calls,
        llm_reference_max_output_tokens=args.llm_reference_max_output_tokens,
        llm_reference_timeout_seconds=args.llm_reference_timeout_seconds,
    )
    scheme = "https" if getattr(server, "tls_enabled", False) else "http"
    print(f"SHARE expert-assessment server listening on {scheme}://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
