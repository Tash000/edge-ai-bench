# STT benchmarks

Benchmarks for local speech-to-text engines on low-end and edge hardware.

## What's measured

| Metric | Meaning | Better is |
|---|---|---|
| `model_load_s` | time to load the model into memory | lower |
| `rtf` | real-time factor = transcription time / audio duration | lower (< 1.0 = faster than real-time) |
| `wer` | word error rate vs. a reference transcript (word-level edit distance) | lower |

Like the LLM benchmark, every run aggregates across `--sessions` (default, and
community-submission minimum: 3) independent sessions, reporting
median/mean/stdev and flagging high-variance or single-session results.

## Test set format

A folder of matching pairs:
```
test_audio/
  sample1.wav
  sample1.txt   <- ground-truth transcript of sample1.wav
  sample2.wav
  sample2.txt
  ...
```
16kHz mono 16-bit PCM WAV is recommended (required for Vosk).

## Engines

- **`whisper_benchmark.py`** — [openai-whisper](https://github.com/openai/whisper). `--model` is a size name (`tiny`, `base`, `small`, ...).
  ```
  python whisper_benchmark.py --model tiny --audio-dir ./test_audio --sessions 3
  ```
- **`vosk_benchmark.py`** — [Vosk](https://alphacephei.com/vosk/models). `--model` is a path to an extracted model folder (models aren't named/downloaded automatically — grab one from the link above).
  ```
  python vosk_benchmark.py --model ./vosk-model-small-en-us-0.15 --audio-dir ./test_audio --sessions 3
  ```

Both scripts share hardware detection and result routing with the LLM
benchmark (`scripts/hardware_info.py`) and write to the same `results/`
tree, so STT and LLM results for the same device sit side by side.

Install engine dependencies: `pip install -r requirements.txt` (each script
also fails with a clear message if its engine isn't installed).

## Status

These are working skeletons — model loading, timing, RTF, and WER are real
and functional. Not yet built: an HTML report generator for STT results (the
LLM one in `llm/llm_benchmark.py --report` doesn't read this schema yet) and
a bundled example test-audio set. Contributions welcome — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
