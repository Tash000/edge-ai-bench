## What kind of PR is this?

- [ ] New results submission
- [ ] Script / tooling change
- [ ] Docs
- [ ] Other

## If submitting results

- [ ] Ran with `--sessions 3` or more (`test_config.sessions >= 3`)
- [ ] Ran `python scripts/validate_results.py <your aggregated_results.json>` locally and it passed
- [ ] Included the generated `report.html` alongside the JSON
- [ ] Ran `python scripts/generate_leaderboard.py` and committed the resulting README.md diff
- [ ] Hardware fields (CPU, RAM, OS, GPU, Ollama version, power source) all look correct — spot-check `aggregated_results.json`'s `hardware` block
- [ ] No `data_quality_flags` I can't explain (or I've explained them below)

## Summary

<!-- What did you test, on what hardware, and what stood out? -->

## Data-quality notes (if any flags are present)

<!-- e.g. "swap_active_during_test — machine only has 4GB RAM, expected under heavy models" -->
