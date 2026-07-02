# Submitting results

## 1. Install dependencies

```bash
cd llm
pip install -r requirements.txt
```

`psutil` is optional but recommended — without it, RAM/swap detection falls
back to parsing `/proc/meminfo` (Linux only; other OSes get `null` RAM
fields, which will fail validation).

## 2. Make sure Ollama is running and has your models

```bash
ollama serve   # if not already running as a service
ollama pull llama3.2:1b   # etc — pull whatever you want to test
```

## 3. Run the benchmark

```bash
python llm_benchmark.py --models llama3.2:1b llama3.2:3b qwen2.5:1.5b \
  --sessions 3 --submitted-by "your-github-handle"
```

- `--sessions 3` is the minimum accepted for a PR (default is already 3).
  More is better if you have time.
- Add `--device-nickname mylaptop` if you don't want your hostname in the
  results path.
- On a Raspberry Pi / Jetson / other SBC, device class is auto-detected —
  you don't need to pass anything extra.
- This takes a while: 3 sessions x N models x the full test suite. Expect
  several minutes per model per session on constrained hardware.

This produces:
```
results/<laptop-desktop|edge-device>/.../<your-device>/<timestamp>/
  aggregated_results.json
  summary.csv
  report.html
```

## 4. Check your own results before submitting

```bash
open results/.../report.html   # or just double-click it — no server needed
```
Read the "Key findings" and check the hardware card matches your actual
machine. If you see a `data_quality_flags` banner, understand why (swap
active? high variance?) before treating the numbers as final — rerun if
something looks wrong (e.g. you had a browser doing a big download in the
background).

## 5. Validate against the schema

```bash
pip install jsonschema
python ../scripts/validate_results.py results/.../aggregated_results.json
```
Must print `OK`. If it fails, the PR's CI check will fail too — fix it
locally first.

## 6. Update the leaderboard

```bash
python ../scripts/generate_leaderboard.py
```
This rewrites the leaderboard tables in `README.md`. Commit that diff along
with your results folder.

## 7. Open a PR

Include the whole `results/.../<timestamp>/` folder (JSON + CSV + HTML) and
the updated `README.md`. Fill out the PR template checklist. If you want to
add your own use-case config (persona/intent), see
[CONTRIBUTING.md](CONTRIBUTING.md) section 2.

## Generating a report from a JSON someone else sent you

You don't need Ollama or even a matching OS to render a report — the
`--report` flag only reads the JSON:

```bash
python llm_benchmark.py --report path/to/aggregated_results.json
```

The JSON is also self-describing enough (schema version, all fields
labeled) to paste directly into any LLM chat and ask for a narrative
summary, if you want prose instead of/alongside the HTML report.
