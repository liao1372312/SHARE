from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the local runtime for a SWE-bench pilot.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--minimum-free-gb", type=float, default=50.0)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(".venv-swebench/bin/python"),
        help="Python interpreter that must import the official swebench package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.pilot / "pilot_manifest.json"
    specs_path = args.pilot / "execution_specs.jsonl"
    checks: dict[str, object] = {
        "pilot_manifest": manifest_path.is_file(),
        "execution_specs": specs_path.is_file(),
        "docker_cli": shutil.which("docker") is not None,
        "swebench_python": False,
    }
    swebench_error = None
    if args.python.is_file():
        result = subprocess.run(
            [str(args.python), "-c", "import swebench; print(getattr(swebench, '__version__', 'unknown'))"],
            text=True,
            capture_output=True,
            timeout=15,
        )
        checks["swebench_python"] = result.returncode == 0
        checks["swebench_python_path"] = str(args.python)
        checks["swebench_version"] = result.stdout.strip() if result.returncode == 0 else None
        swebench_error = result.stderr.strip() if result.returncode != 0 else None
    else:
        checks["swebench_python_path"] = str(args.python)
        checks["swebench_version"] = None
    free_gb = shutil.disk_usage(args.pilot.parent if args.pilot.parent.exists() else Path.cwd()).free / (1024**3)
    checks["free_disk_gb"] = round(free_gb, 2)
    checks["minimum_free_disk"] = free_gb >= args.minimum_free_gb
    docker_error = None
    if checks["docker_cli"]:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
            capture_output=True,
            timeout=15,
        )
        checks["docker_daemon"] = result.returncode == 0
        checks["docker_server_version"] = result.stdout.strip() if result.returncode == 0 else None
        docker_error = result.stderr.strip() if result.returncode != 0 else None
    else:
        checks["docker_daemon"] = False
        checks["docker_server_version"] = None
    required = ("pilot_manifest", "execution_specs", "docker_cli", "docker_daemon", "swebench_python", "minimum_free_disk")
    ready = all(checks.get(key) is True for key in required)
    report = {
        "schema_version": 1,
        "kind": "share_swebench_runtime_preflight",
        "ready": ready,
        "checks": checks,
        "docker_error": docker_error,
        "swebench_error": swebench_error,
        "next_action": None if ready else "Restore Docker daemon access and install the official swebench package before branch execution.",
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
