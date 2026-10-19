from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EPOCH_IMAGE_TEMPLATE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
EPOCH_NJU_TEMPLATE = "ghcr.nju.edu.cn/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
EPOCH_CHENBY_TEMPLATE = "ghcr.chenby.cn/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
EPOCH_DAOCLOUD_TEMPLATE = "m.daocloud.io/ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"


def read_specs(path: Path) -> dict[str, dict[str, Any]]:
    specs = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"execution spec line {line_number} is not JSON") from error
            instance_id = value.get("instance_id") if isinstance(value, dict) else None
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"execution spec line {line_number} lacks instance_id")
            specs[instance_id] = value
    return specs


def inspect_image(image: str) -> dict[str, Any]:
    result = subprocess.run(["docker", "image", "inspect", image], text=True, capture_output=True)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"could not inspect {image}")
    values = json.loads(result.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"unexpected docker inspect result for {image}")
    value = values[0]
    return {
        "id": value.get("Id"),
        "repo_digests": sorted(value.get("RepoDigests") or []),
        "created": value.get("Created"),
        "architecture": value.get("Architecture"),
        "os": value.get("Os"),
        "size_bytes": value.get("Size"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull and bind one SWE-bench image from an alternate public registry.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument(
        "--direct-attempts",
        type=int,
        default=3,
        help="Attempts for direct GHCR before trying the proxy; use 0 to skip a known-unreachable direct registry.",
    )
    args = parser.parse_args()
    if not 0 <= args.direct_attempts <= 5:
        parser.error("--direct-attempts must be between 0 and 5")
    return args


def pull_image(image: str, attempts: int) -> bool:
    for attempt in range(1, attempts + 1):
        print(f"Pulling {image} (attempt {attempt}/{attempts})", flush=True)
        pull = subprocess.run(["docker", "pull", image])
        if pull.returncode == 0:
            return True
        if attempt < attempts:
            delay = min(10, attempt * 3)
            print(f"Pull failed; retrying the same registry in {delay} seconds so cached layers can be reused.", flush=True)
            time.sleep(delay)
    return False


def main() -> None:
    args = parse_args()
    specs_path = args.pilot / "execution_specs.jsonl"
    if not specs_path.is_file():
        raise SystemExit("pilot execution_specs.jsonl is missing")
    try:
        specs = read_specs(specs_path)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.instance not in specs:
        raise SystemExit(f"instance is not in the pilot: {args.instance}")
    target_image = specs[args.instance].get("image")
    if not isinstance(target_image, str) or not target_image:
        raise SystemExit("pilot execution spec lacks an image")
    direct_image = EPOCH_IMAGE_TEMPLATE.format(instance_id=args.instance)
    proxy_images = [
        EPOCH_CHENBY_TEMPLATE.format(instance_id=args.instance),
        EPOCH_NJU_TEMPLATE.format(instance_id=args.instance),
        EPOCH_DAOCLOUD_TEMPLATE.format(instance_id=args.instance),
    ]
    provenance_dir = args.pilot / "image_provenance"
    provenance_path = provenance_dir / f"{args.instance}.json"
    if provenance_path.exists():
        raise SystemExit(f"refusing to overwrite image provenance: {provenance_path}")

    source_image = direct_image if args.direct_attempts and pull_image(direct_image, args.direct_attempts) else None
    if source_image is None:
        for proxy_image in proxy_images:
            if pull_image(proxy_image, 1):
                source_image = proxy_image
                break
    if source_image is None:
        raise SystemExit("all alternate SWE-bench image sources failed")
    tag = subprocess.run(["docker", "tag", source_image, target_image])
    if tag.returncode != 0:
        raise SystemExit(f"docker tag failed with exit code {tag.returncode}")
    try:
        source_details = inspect_image(source_image)
        target_details = inspect_image(target_image)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if source_details["id"] != target_details["id"]:
        raise SystemExit("source and SWE-bench target tags do not resolve to the same image ID")

    provenance = {
        "schema_version": 1,
        "kind": "share_swebench_image_provenance",
        "instance_id": args.instance,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "image_provider": "epoch-research-swe-bench-ghcr",
        "source_image": source_image,
        "target_image": target_image,
        "image": target_details,
        "integrity_check": "source_and_target_tags_share_image_id",
    }
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready": True, "image_id": target_details["id"], "provenance": str(provenance_path)}, indent=2))


if __name__ == "__main__":
    main()
