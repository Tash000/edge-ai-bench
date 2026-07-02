# edge-ai-bench

A community benchmark for local AI models — LLM, STT, and (soon) TTS — on
low-end and edge hardware. Not "how good is this model," but "what actually
runs usably on the machine you already have."

Every result carries the hardware that produced it, is aggregated across at
least 3 independent sessions (a single run is noise — see
[docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md)), and is
schema-validated before it can be merged. Results split into two device
classes — laptops/desktops (tiered by RAM) and edge devices like Raspberry
Pi / Jetson Nano (grouped by board, since even a weak laptop has more of
everything than a fixed-spec SBC) — so you're only comparing like with like.

## Quickstart

```bash
git clone https://github.com/Tash000/edge-ai-bench.git
cd edge-ai-bench/llm
pip install -r requirements.txt
ollama serve   # if not already running

python llm_benchmark.py --models llama3.2:1b llama3.2:3b --sessions 3 --submitted-by "yourname"
```

This produces `aggregated_results.json`, `summary.csv`, and a self-contained
`report.html` (charts + hardware summary + key findings — open it directly,
no server needed, works offline) under `results/`.

Already have a results JSON and just want the report?
```bash
python llm_benchmark.py --report path/to/aggregated_results.json
```

## What's measured

Cold-load time, TTFT, TPOT, tokens/sec, end-to-end latency, strict-JSON and
hybrid-JSON+chat format compliance, persona/constraint adherence, intent
(command) detection accuracy, and context/memory recall. Full definitions in
[docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md).

STT (Whisper, Vosk) benchmarks — model load time, real-time factor, word
error rate — live in [`stt/`](stt/). TTS is planned; see
[`tts/README.md`](tts/README.md) if you want to help build it.

## Leaderboard

Generated from `results/**` via `python scripts/generate_leaderboard.py` —
never hand-edited. Sorted by median tokens/sec within each hardware
category.

<!-- LEADERBOARD:START -->
#### Laptop/Desktop — Tier 2 Low Ram

| Model | Device | Tok/s (median) | Cold load | JSON pass% | Intent acc | Submitted by | Date | Flags |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | praveen-Inspiron-15-3567 | 10.2 tok/s | 20.6s | 67% | 89% | Tash000 | 2026-07-02 | swap_active_during_test |
| llama3.2:1b | praveen-Inspiron-15-3567 | 9.3 tok/s | 26.8s | 67% | 89% | Tash000 | 2026-07-02 | swap_active_during_test |
| llama3.2:3b | praveen-Inspiron-15-3567 | 6.0 tok/s | 42.4s | 100% | 89% | Tash000 | 2026-07-02 | swap_active_during_test |
| gemma2:2b | praveen-Inspiron-15-3567 | 5.3 tok/s | 18.3s | 100% | 89% | Tash000 | 2026-07-02 | swap_active_during_test |
| qwen:latest | praveen-Inspiron-15-3567 | 4.0 tok/s | 45.6s | 33% | 67% | Tash000 | 2026-07-02 | swap_active_during_test |
<!-- LEADERBOARD:END -->

## Contributing results

See [RESULTS_SUBMISSION.md](RESULTS_SUBMISSION.md) for the full walkthrough,
and [CONTRIBUTING.md](CONTRIBUTING.md) for adding your own persona/intent
configs or improving the tooling. Every results PR is validated in CI
against [`schemas/results_schema.json`](schemas/results_schema.json) —
malformed submissions or runs with fewer than 3 sessions are rejected
automatically, so the leaderboard stays trustworthy without manual review of
every PR.

## Repo layout

```
llm/            LLM benchmark script + swappable persona/intent/context configs
stt/            STT benchmark scripts (Whisper, Vosk)
tts/            planned — help wanted
scripts/        validation, leaderboard generation, shared hardware detection
schemas/        JSON schemas every result file must satisfy
docs/           methodology
results/        submitted results, laptop-desktop/ and edge-device/
```

## License

MIT — see [LICENSE](LICENSE).
