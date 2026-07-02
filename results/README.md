# Results layout

```
results/
  laptop-desktop/
    tier-1-ultra-low-ram/      <=4GB RAM
    tier-2-low-ram/            4-8GB RAM
    tier-3-mid-ram/            8-16GB RAM
    tier-4-gpu-accelerated/    any RAM, GPU offload confirmed via `ollama ps` VRAM usage
  edge-device/
    raspberry-pi/
    jetson-nano/
    other-sbc/
```

Each tier/board folder contains one subfolder per contributor device
(`<device-nickname>/`), and inside that, one subfolder per submission
(`<YYYYMMDD_HHMMSS>/`) holding `aggregated_results.json`, `summary.csv`, and
`report.html`.

Tier/category is chosen automatically by `llm_benchmark.py` based on
detected hardware (see `docs/BENCHMARK_METHODOLOGY.md` for the exact rules)
— you don't pick it manually.

Don't hand-edit the leaderboard in the top-level `README.md`; it's generated
from everything in this folder via `python scripts/generate_leaderboard.py`.
See [RESULTS_SUBMISSION.md](../RESULTS_SUBMISSION.md) to add your own.
