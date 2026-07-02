# Changelog

## 1.0 — 2026-07-01

Initial public structure.

- `llm/llm_benchmark.py`: full rewrite of the original single-run script.
  - Hardware metadata (CPU/RAM/swap/OS/GPU/Ollama version/power source)
    recorded in every result.
  - `--sessions` (default 3, minimum for submissions): runs the whole suite
    N independent times and aggregates with median/mean/stdev, flagging
    high-variance and swap-active results.
  - Strict, versioned JSON schema (`schemas/results_schema.json`) so every
    contributor's output is shape-identical.
  - Config-driven personas and intent-detection tests (previously hardcoded
    to one robotics project) — generic defaults ship in the repo, with a
    "robot companion" example pack preserved for that use case.
  - `--report`: generates a self-contained, offline-safe HTML report
    (inline SVG charts, hardware summary, rule-based key findings) from any
    existing results JSON, no Ollama required.
  - Results auto-routed into `laptop-desktop/tier-N` vs `edge-device/<board>`
    based on detected hardware.
- Added `stt/whisper_benchmark.py` and `stt/vosk_benchmark.py` (model load
  time, RTF, WER — working skeletons, sharing hardware detection with the
  LLM script).
- Added `scripts/validate_results.py` and `scripts/generate_leaderboard.py`,
  wired into CI (`.github/workflows/validate-results.yml`) so malformed or
  under-sampled submissions can't merge silently.
