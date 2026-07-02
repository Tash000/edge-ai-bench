#!/usr/bin/env python3
"""
Validates a results JSON (LLM or STT) against its schema plus community
submission rules. Used both locally by contributors and in CI on any PR
touching results/**/*.json — this is what keeps the leaderboard trustworthy
instead of relying on manual review of every PR.

Usage:
  python scripts/validate_results.py results/laptop-desktop/tier-2-low-ram/mybox/20260701_120000/aggregated_results.json
  python scripts/validate_results.py results/**/*.json   (shell-expanded)

Exit code 0 = all files pass. Non-zero = at least one file failed.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_SCHEMA = REPO_ROOT / "schemas" / "results_schema.json"
STT_SCHEMA = REPO_ROOT / "schemas" / "results_schema_stt.json"

MIN_SESSIONS = 3


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _pick_schema(payload: dict) -> Path:
    if "engines_tested" in payload.get("test_config", {}):
        return STT_SCHEMA
    return LLM_SCHEMA


def _jsonschema_check(payload: dict, schema_path: Path) -> list:
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema package not installed — skipping full schema validation "
                "(structural checks below still ran). Run: pip install jsonschema"]
    schema = json.loads(schema_path.read_text())
    validator_cls = jsonschema.Draft7Validator
    resolver = jsonschema.RefResolver(base_uri=schema_path.parent.as_uri() + "/", referrer=schema)
    validator = validator_cls(schema, resolver=resolver)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


def _rule_checks(payload: dict) -> list:
    errors = []

    hw = payload.get("hardware", {})
    required_hw_fields = ["hostname", "cpu_model", "cpu_cores_physical", "cpu_threads_logical",
                           "ram_total_gb", "os", "device_class", "ollama_version", "power_source"]
    for field in required_hw_fields:
        if hw.get(field) in (None, ""):
            errors.append(f"hardware.{field} is missing or null — every submission needs full hardware metadata")

    test_config = payload.get("test_config", {})
    sessions = test_config.get("sessions")
    if sessions is None:
        errors.append("test_config.sessions is missing")
    elif sessions < MIN_SESSIONS:
        errors.append(f"test_config.sessions={sessions} — community submissions require >={MIN_SESSIONS} "
                       f"independent sessions (run with --sessions {MIN_SESSIONS} or more)")

    results = payload.get("results", [])
    if not results:
        errors.append("results is empty — no models/engines were benchmarked")

    for r in results:
        model = r.get("model") or r.get("engine", "<unknown>")
        n_raw_sessions = len(r.get("sessions_raw", []))
        if sessions and n_raw_sessions != sessions:
            errors.append(f"{model}: sessions_raw has {n_raw_sessions} entries but "
                           f"test_config.sessions={sessions} — mismatch")

    return errors


def validate_file(path: Path) -> list:
    try:
        payload = _load_json(path)
    except Exception as e:
        return [f"could not parse JSON: {e}"]

    schema_path = _pick_schema(payload)
    errors = []
    errors += _jsonschema_check(payload, schema_path)
    errors += _rule_checks(payload)
    return errors


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/validate_results.py <results.json> [more.json ...]")

    any_failed = False
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"FAIL {path}: file not found")
            any_failed = True
            continue
        errors = validate_file(path)
        if errors:
            any_failed = True
            print(f"FAIL {path}")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK   {path}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
