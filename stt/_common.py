#!/usr/bin/env python3
"""Shared helpers for the STT benchmark scripts: WER scoring, WAV duration,
and cross-session aggregation — mirrors the reliability approach in
llm/llm_benchmark.py (>=3 sessions, median/stdev, data-quality flags)."""

import statistics
import wave
from pathlib import Path


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard word-level Levenshtein distance: (substitutions + deletions
    + insertions) / len(reference words). Lower is better; 0.0 = perfect."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return round(d[n][m] / n, 4)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def find_audio_reference_pairs(audio_dir: Path) -> list:
    """A valid test set is a folder of matching <name>.wav + <name>.txt
    (reference transcript) pairs."""
    pairs = []
    for wav_path in sorted(Path(audio_dir).glob("*.wav")):
        ref_path = wav_path.with_suffix(".txt")
        if ref_path.exists():
            pairs.append((wav_path, ref_path.read_text().strip()))
    return pairs


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv_pct = round((stdev / mean) * 100, 1) if mean else None
    return {
        "median": round(statistics.median(vals), 4),
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        "cv_pct": cv_pct,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


def aggregate_stt_sessions(sessions: list) -> dict:
    n = len(sessions)
    flags = []
    if n < 3:
        flags.append(f"insufficient_sessions:{n}_of_3_minimum")

    metrics = {
        "model_load_s": stats([s.get("model_load_s") for s in sessions]),
        "rtf": stats([s.get("rtf_mean") for s in sessions]),
        "wer": stats([s.get("wer_mean") for s in sessions]),
    }
    for name, stat in metrics.items():
        if stat and stat["cv_pct"] is not None and stat["cv_pct"] > 25:
            flags.append(f"high_variance:{name}({stat['cv_pct']}%)")

    return {"metrics": metrics, "data_quality_flags": flags}
