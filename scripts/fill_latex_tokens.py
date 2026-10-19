from __future__ import annotations

import argparse
import csv
from pathlib import Path

METHOD_TO_LATEX = {
    "No-Repair": "No-Repair",
    "Retry-only": "Retry-only",
    "Full Re-execution": "Full Re-execution",
    "Last-Step Repair": "Last-Step Repair",
    "Supervisor-only Repair": "Supervisor-only Repair",
    "LLM-Diagnose-and-Repair": "LLM-Diagnose-and-Repair",
    "EPF+Repair": "EPF+Repair",
    "FLARE": "FLARE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill the Tokens column in the ICSE LaTeX repair table from measured token usage.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tex", type=Path, default=Path("ICSE/IEEE-conference-template-062824.tex"))
    args = parser.parse_args()
    tokens = load_tokens(args.summary)
    text = args.tex.read_text(encoding="utf-8")
    for method, latex_method in METHOD_TO_LATEX.items():
        if method in tokens:
            text = replace_repair_token(text, latex_method, tokens[method])
    args.tex.write_text(text, encoding="utf-8")


def load_tokens(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["method"]: str(round(float(row["total_tokens"]))) for row in rows}


def replace_repair_token(text: str, method: str, token: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(method + " & ") and "& TBD" in line:
            prefix = line.rsplit("&", 1)[0]
            lines[index] = prefix + f"& {token} \\\\" 
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
