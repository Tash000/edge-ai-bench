#!/usr/bin/env python3
"""
Rebuilds the leaderboard tables in README.md from every results/**/aggregated_results.json
found in the repo. Never hand-edit the leaderboard section — it drifts from
reality the moment someone forgets to update it by hand. Run this after adding
new results and commit the resulting README.md diff.

Usage: python scripts/generate_leaderboard.py [--check]
  --check   exit non-zero if README.md would change (for CI), without writing.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
START_MARKER = "<!-- LEADERBOARD:START -->"
END_MARKER = "<!-- LEADERBOARD:END -->"


def _find_result_files():
    return sorted(REPO_ROOT.glob("results/**/aggregated_results.json"))


def _category_label(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT / "results").parts
    # laptop-desktop/tier-2-low-ram/<nickname>/<ts>/aggregated_results.json
    # edge-device/raspberry-pi/<nickname>/<ts>/aggregated_results.json
    if parts[0] == "laptop-desktop":
        return f"Laptop/Desktop — {parts[1].replace('-', ' ').title()}"
    if parts[0] == "edge-device":
        return f"Edge device — {parts[1].replace('-', ' ').title()}"
    return parts[0]


def _row(model_result: dict, hw: dict, submitted_by: str, ts: str) -> tuple:
    agg = model_result["aggregated"]
    tps = (agg["metrics"]["tokens_per_sec"] or {}).get("median")
    cold = (agg["metrics"]["cold_load_s"] or {}).get("median")
    rates = [v["pass_rate_median"] for v in agg["format_compliance"].values() if v["pass_rate_median"] is not None]
    json_avg = round(sum(rates) / len(rates) * 100) if rates else None
    intent_acc = agg["intent_detection"].get("accuracy_median")
    flags = ", ".join(agg["data_quality_flags"]) or "-"
    device = hw.get("device_type") or hw.get("hostname", "?")
    sort_key = tps if tps is not None else -1
    row_md = (f"| {model_result['model']} | {device} | "
              f"{tps:.1f} tok/s | {cold:.1f}s | {json_avg if json_avg is not None else 'N/A'}% | "
              f"{round(intent_acc*100) if intent_acc is not None else 'N/A'}% | "
              f"{submitted_by} | {ts} | {flags} |")
    return sort_key, row_md


def build_leaderboard_markdown() -> str:
    files = _find_result_files()
    if not files:
        return "_No results submitted yet — see RESULTS_SUBMISSION.md to add the first one._"

    by_category = {}
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        category = _category_label(f)
        hw = payload.get("hardware", {})
        submitted_by = payload.get("submitted_by", "anonymous")
        ts = payload.get("submission_date", "?")
        for r in payload.get("results", []):
            by_category.setdefault(category, []).append(_row(r, hw, submitted_by, ts))

    sections = []
    for category in sorted(by_category):
        rows = sorted(by_category[category], key=lambda t: t[0], reverse=True)
        header = ("| Model | Device | Tok/s (median) | Cold load | JSON pass% | Intent acc | Submitted by | Date | Flags |\n"
                   "|---|---|---|---|---|---|---|---|---|")
        table = "\n".join(r[1] for r in rows)
        sections.append(f"#### {category}\n\n{header}\n{table}")

    return "\n\n".join(sections)


def main():
    check_only = "--check" in sys.argv
    leaderboard_md = build_leaderboard_markdown()

    text = README.read_text()
    if START_MARKER not in text or END_MARKER not in text:
        sys.exit(f"README.md is missing {START_MARKER}/{END_MARKER} markers")

    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    new_text = f"{before}{START_MARKER}\n{leaderboard_md}\n{END_MARKER}{after}"

    if new_text == text:
        print("Leaderboard already up to date.")
        return

    if check_only:
        sys.exit("README.md leaderboard is out of date — run: python scripts/generate_leaderboard.py")

    README.write_text(new_text)
    print("README.md leaderboard updated.")


if __name__ == "__main__":
    main()
