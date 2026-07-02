# Benchmark methodology

This document explains what each metric means, why the benchmark is
structured the way it is, and what its numbers do *not* tell you.

## Why sessions, not just trials

There are two different axes of repetition in the LLM benchmark:

- **`--runs`** (default 3): trials of the *same test* within one session —
  smooths out per-request noise (e.g. one slow token burst).
- **`--sessions`** (default 3, minimum for community submissions): entirely
  separate invocations of the *whole test suite*, spaced apart in time —
  catches session-level variance that trial averaging can't: thermal
  throttling as the CPU heats up over minutes, background processes
  competing for the CPU, swap pressure building up, or an Ollama model
  getting a slightly different quantization loaded.

A single session, however many trials it averages internally, is still one
sample of "how this device happened to behave for the several minutes this
ran." Results are aggregated across sessions with **median** (robust to one
bad session), **stdev**, and **coefficient of variation (cv_pct)** — if
cv_pct exceeds 25% for a core latency metric, the result is flagged
`high_variance:<metric>` rather than silently reported as fact.

## Data-quality flags

Every aggregated result carries a `data_quality_flags` list. Treat any
result carrying these as provisional, not definitive:

| Flag | Meaning |
|---|---|
| `insufficient_sessions:N_of_3_minimum` | fewer than 3 sessions were run — high risk the numbers are noise, not signal |
| `high_variance:<metric>(cv%)` | that metric's coefficient of variation across sessions exceeded 25% |
| `swap_active_during_test` | the OS was actively swapping during at least one session — a strong latency confound on RAM-constrained hardware |

## Hardware metadata

Every result records CPU model, physical/logical core count, RAM, swap
usage at test time, OS, GPU (if any), Ollama version, and power source
(AC/battery). This exists because a bare number like "9.5 tok/s" is
meaningless without knowing what produced it — and because model tags
(e.g. `llama3.2:1b`) aren't pinned to a fixed quantization forever, the
Ollama version is recorded alongside the model name so future readers can
tell whether a discrepancy is the model, the runtime, or the hardware.

**GPU acceleration is verified, not assumed.** `hardware.gpu_model` is
descriptive only (what GPU exists on the box). Whether a given model
*actually* ran on it is decided per-session from Ollama's own
`size_vram_bytes` report via `ollama ps` — a laptop with an iGPU can list a
"GPU" while every model still runs pure-CPU.

## Device classes and tiers

Results are split into two top-level categories before any RAM tiering:

- **`laptop-desktop/`** — tiered by RAM: `tier-1-ultra-low-ram` (<=4GB),
  `tier-2-low-ram` (4-8GB), `tier-3-mid-ram` (8-16GB), or
  `tier-4-gpu-accelerated` (any RAM, if a session confirmed GPU offload).
- **`edge-device/`** — single-board computers (Raspberry Pi, Jetson Nano,
  etc.), grouped by board name instead of a RAM tier. Even the weakest x86
  laptop has more RAM, storage, cooling, and expandability than a
  fixed-spec SBC, so the two categories aren't comparable and shouldn't
  share a tier scale. Detected automatically via `/proc/device-tree/model`
  (present on ARM SBCs, absent on x86).

## Metrics glossary

| Metric | Definition | Better is |
|---|---|---|
| Cold load | time from "model not in memory" to first response after a forced unload | lower |
| TTFT | time to first streamed token | lower |
| TPOT / ITL | seconds per output token after the first (inter-token latency) | lower |
| Tokens/sec | decode throughput once generation has started | higher |
| End-to-end | full wall-clock time for one turn | lower |
| JSON / format compliance | pass rate against a validator across N trials, per format type (strict JSON, hybrid JSON+chat, plain text) | higher |
| Persona adherence | pass rate against *hard* constraints declared in a persona YAML (forbidden words, required address form, max sentences) — **not** a tone/character judgment, which still needs a human read of the `samples` field | higher |
| Intent accuracy | few-shot classification accuracy against a swappable YAML config of utterance -> label cases | higher |
| Context recall | whether a fact buried in the system/context text was correctly recalled when asked later | higher (boolean rate) |

## The "recommended pick" in report.html

The HTML report's key-findings section includes a single heuristic
recommendation: a weighted score (35% decode speed, 30% JSON compliance,
20% intent accuracy, 15% cold-load time), each metric min-max normalized
across the models actually tested. This is a deliberately simple, documented
heuristic — not a scientific ranking, and not applicable if your use case
weighs these differently (e.g. cold-load time barely matters if the model
stays resident). Read the full table, not just the recommendation.

## Known confounds this benchmark does not fully control for

- **Thermal state** — a laptop that's been running for an hour behaves
  differently than one that just woke up. Sessions are spaced by however
  long the previous session took, which helps some, but there's no explicit
  cooldown enforced.
- **Background load** — the script doesn't check what else is running on
  the machine. If you're benchmarking, close everything else.
- **Battery vs AC** — recorded (`power_source`), but not enforced. Many
  laptops throttle on battery; compare like-for-like when possible.
- **First-run disk cache effects** — the very first cold load of a
  just-pulled model may be slower than subsequent ones due to disk cache
  state. Not explicitly isolated.
