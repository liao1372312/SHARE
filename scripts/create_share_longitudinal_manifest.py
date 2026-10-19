from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.deployment import canonical_digest, deployment_manifest_payload, is_sha256, load_deployment_events, sha256_file
from share.types import RewardConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind observed SHARE deployment events to an externally attested longitudinal manifest.")
    parser.add_argument("--deployment-events", type=Path, required=True, help="Immutable single-branch observed deployment JSONL.")
    parser.add_argument("--encoder", type=Path, required=True, help="Immutable feature encoder used by every declared policy state.")
    parser.add_argument("--frozen-at", required=True, help="UTC timestamp at which policy states and protocol were frozen.")
    parser.add_argument("--prior-unseen-attestation", required=True, help="External preregistration or release reference for prior-unseen data status.")
    parser.add_argument("--external-provenance-attestation", required=True, help="External audit, signature ledger, or data-governance reference for deployed execution provenance.")
    parser.add_argument("--assignment-protocol", required=True, help="Predeclared allocation rule for Frozen and Adaptive deployment arms.")
    parser.add_argument("--frozen-policy-sha256", required=True, help="SHA-256 of the frozen checkpoint used by Frozen-arm events.")
    parser.add_argument("--adaptive-policy-sha256", action="append", required=True, help="SHA-256 policy-state version allowed for Adaptive-arm events; repeat for multiple versions.")
    parser.add_argument("--human-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--code-revision", required=True, help="Immutable source revision or archived release identifier.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not all((0.0 <= args.human_budget <= 1.0, 0.0 <= args.unsafe_budget <= 1.0)):
        raise SystemExit("human and unsafe budgets must be in [0, 1]")
    hashes = [args.frozen_policy_sha256, *args.adaptive_policy_sha256]
    if any(not is_sha256(value) for value in hashes):
        raise SystemExit("every policy SHA-256 must be 64 hexadecimal characters")
    if not all((args.prior_unseen_attestation.strip(), args.external_provenance_attestation.strip(), args.assignment_protocol.strip(), args.code_revision.strip())):
        raise SystemExit("attestation, assignment protocol, and code revision fields must be non-empty")
    events = load_deployment_events(args.deployment_events)
    if not args.encoder.is_file():
        raise SystemExit("--encoder must be an existing feature-encoder file")
    payload = deployment_manifest_payload(
        events,
        args.frozen_at,
        args.prior_unseen_attestation.strip(),
        args.external_provenance_attestation.strip(),
        args.assignment_protocol.strip(),
        {"frozen": [args.frozen_policy_sha256], "adaptive": args.adaptive_policy_sha256},
        RewardConfig(),
        args.human_budget,
        args.unsafe_budget,
        args.code_revision.strip(),
        sha256_file(args.deployment_events),
        sha256_file(args.encoder),
    )
    payload["manifest_digest"] = canonical_digest(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "events": payload["event_count"], "episodes": len({event.episode_id for event in events})}, indent=2))


if __name__ == "__main__":
    main()
