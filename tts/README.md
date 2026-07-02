# TTS benchmarks

Not built yet — help wanted.

## Planned scope

Benchmarking local text-to-speech engines (Piper, Coqui TTS, pyttsx3,
espeak-ng, Kokoro, ...) on low-end and edge hardware, following the same
philosophy as `llm/llm_benchmark.py` and `stt/`:

| Metric | Meaning | Better is |
|---|---|---|
| `model_load_s` | time to load the model/voice | lower |
| `ttfa` (time-to-first-audio) | latency from text submitted to first audio chunk | lower |
| `rtf` | synthesis time / output audio duration | lower (< 1.0 = faster than real-time) |
| naturalness / MOS-proxy | can't be scored automatically without a trained model — likely a human-rated rubric or an optional LLM-judge listening test, not a hard number | — |

Same reliability rules as the rest of the repo would apply: hardware
metadata baked into every result, `--sessions >= 3` for submissions,
data-quality flags, a self-contained HTML report.

If you want to build this out, open an issue or PR — see
[CONTRIBUTING.md](../CONTRIBUTING.md). `scripts/hardware_info.py` and the
result-routing/aggregation pattern in `stt/_common.py` are ready to reuse.
