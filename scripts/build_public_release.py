"""Copy the explicit publication allowlist without altering the source checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "release/SHARE")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("use a new release output directory")
    manifest = json.loads((ROOT / "configs/release_files.json").read_text(encoding="utf-8"))
    paths = []
    for name in manifest["files"]:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f"missing or invalid release source: {name}")
        paths.append((name, path))
    out.mkdir(parents=True, exist_ok=False)
    for name, path in paths:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace(
        "Broad cleanup of this source directory is awaiting confirmation of its exact scope; use the curated `release/SHARE/` directory for the proposed public release, rather than uploading this entire source directory.",
        "This publication copy excludes the historical simulation outcomes, obsolete experiments, participant databases, credentials, and logs. It remains a migration draft pending the observed inputs listed in the checklist.",
    )
    readme.write_text(text, encoding="utf-8")
    checklist = out / "PAPER_COMPLETION_CHECKLIST.md"
    text = checklist.read_text(encoding="utf-8")
    text = text.replace("原工作目录的大范围删除尚待确认具体范围；精简发布目录位于 `release/SHARE/`。", "当前目录即精简发布副本，历史实验文件未包含在此副本内。")
    checklist.write_text(text, encoding="utf-8")
    cleanup_path = out / "docs/CLEANUP_MANIFEST.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup["status"] = "excluded_from_publication_copy"
    cleanup["archive"] = "private local backup; not distributed"
    cleanup.pop("source_workspace", None)
    cleanup_path.write_text(json.dumps(cleanup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release": str(out), "files": len(paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
