# Contributing to edge-ai-bench

Thanks for helping build a real picture of what local AI actually runs
usably on low-end and edge hardware. There are three ways to contribute:
submitting results, adding a config, or improving the tooling.

## 1. Submitting benchmark results

This is the most valuable contribution. See
[RESULTS_SUBMISSION.md](RESULTS_SUBMISSION.md) for the exact commands. In short:

```bash
cd llm
pip install -r requirements.txt
python llm_benchmark.py --models llama3.2:1b llama3.2:3b --sessions 3 --submitted-by "yourname"
```

This writes an `aggregated_results.json`, `summary.csv`, and `report.html`
into `results/<laptop-desktop|edge-device>/.../<device-nickname>/<timestamp>/`.
Open `report.html` yourself first — if a chart or finding looks wrong, your
results probably aren't ready to submit yet.

Before opening a PR:
```bash
python scripts/validate_results.py results/.../aggregated_results.json
python scripts/generate_leaderboard.py
```
Commit the results folder *and* the regenerated `README.md`. The PR template
has a checklist — fill it out honestly, especially the data-quality-flags
question.

**Why >=3 sessions is required:** a single run is noise — thermal state,
background load, and swap usage all skew latency by double digits of
percent on constrained hardware. Three independent full runs, aggregated
with median/stdev, is the minimum bar for a result someone else can trust.
See [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md).

## 2. Adding your own persona / intent config

The default persona and intent tests are intentionally generic. If you're
building something specific (a robot, a kiosk assistant, a game NPC), add
your own config instead of fighting the defaults:

- Persona: drop a `.yaml` file (see `llm/configs/personas/default_pack/generic_assistant.yaml`
  for the expected shape) into your own folder and pass `--persona-dir`.
- Intent: copy `llm/configs/intents/default_intents.yaml` and edit `labels`,
  `system_prompt`, and `test_cases`, then pass `--intent-config`.

If your config is broadly useful (not just for your one project), consider
PRing it in as a new named pack alongside `default_pack/` and
`robot_companion_pack/`.

## 3. Improving the tooling

Scripts live in `llm/`, `stt/`, and `scripts/`. Keep in mind:
- The JSON output schema (`schemas/results_schema.json`) is a contract —
  changing its shape means bumping `SCHEMA_VERSION` in the script and
  updating the schema file and `scripts/validate_results.py` together.
- `llm/llm_benchmark.py --report <json>` must keep working standalone (no
  Ollama, no network) — it's how someone sanity-checks a results file before
  a PR, or generates a report on a different machine than the one that ran
  the benchmark.
- No external JS/CSS in the HTML report — contributors may be offline or on
  very limited bandwidth. Charts are hand-rolled inline SVG on purpose.
- Run `python -m py_compile <file>` at minimum before submitting a script
  change; if you can, actually run it against a local Ollama instance.

## Code of conduct

Be honest about your hardware and results. Don't submit numbers you haven't
actually run. If something looks off in your own results (a data-quality
flag, an outlier), say so in the PR rather than hiding it — that's exactly
the kind of information this repo exists to surface.
