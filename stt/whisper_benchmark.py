#!/usr/bin/env python3
"""
edge-ai-bench — Whisper (openai-whisper) STT benchmark.

Measures, per model, across --sessions independent sessions (default 3):
  - model_load_s : time to load the model into memory
  - rtf           : real-time factor = transcription_time / audio_duration
                    (rtf < 1.0 means faster than real-time; lower is better)
  - wer           : word error rate against reference transcripts
                    (word-level Levenshtein distance / reference word count)

TEST SET: a folder of matching pairs, e.g. sample1.wav + sample1.txt
(sample1.txt holds the ground-truth transcript). 16kHz mono WAV recommended.

USAGE:
  python whisper_benchmark.py --model tiny --audio-dir ./test_audio --sessions 3
  python whisper_benchmark.py --model base --audio-dir ./test_audio --submitted-by yourname

Requires: pip install -r requirements.txt   (openai-whisper)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))

from _common import word_error_rate, wav_duration_seconds, find_audio_reference_pairs, aggregate_stt_sessions  # noqa: E402
from hardware_info import get_hardware_info, classify_result_path  # noqa: E402

try:
    import whisper
except ImportError:
    whisper = None

SCHEMA_VERSION = "1.0"
BENCHMARK_VERSION = "1.0"


def run_session(model_name: str, pairs: list) -> dict:
    t0 = time.perf_counter()
    model = whisper.load_model(model_name)
    model_load_s = time.perf_counter() - t0

    per_file = []
    for wav_path, reference in pairs:
        duration = wav_duration_seconds(wav_path)
        t0 = time.perf_counter()
        result = model.transcribe(str(wav_path))
        elapsed = time.perf_counter() - t0
        hypothesis = result.get("text", "")
        wer = word_error_rate(reference, hypothesis)
        rtf = round(elapsed / duration, 4) if duration else None
        per_file.append({"file": wav_path.name, "duration_s": round(duration, 2),
                          "elapsed_s": round(elapsed, 3), "rtf": rtf, "wer": wer,
                          "hypothesis": hypothesis[:300]})

    rtf_vals = [f["rtf"] for f in per_file if f["rtf"] is not None]
    wer_vals = [f["wer"] for f in per_file if f["wer"] is not None]

    return {
        "timestamp": datetime.now().isoformat(),
        "model_load_s": round(model_load_s, 3),
        "rtf_mean": round(sum(rtf_vals) / len(rtf_vals), 4) if rtf_vals else None,
        "wer_mean": round(sum(wer_vals) / len(wer_vals), 4) if wer_vals else None,
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description="edge-ai-bench: benchmark Whisper STT on low-end hardware.")
    parser.add_argument("--model", required=True, help="Whisper model size, e.g. tiny, base, small")
    parser.add_argument("--audio-dir", required=True, help="folder of matching <name>.wav + <name>.txt pairs")
    parser.add_argument("--sessions", type=int, default=3, help="independent sessions to run and aggregate (default 3)")
    parser.add_argument("--device-nickname", type=str, default=None)
    parser.add_argument("--power-source", choices=["auto", "ac", "battery", "unknown"], default="auto")
    parser.add_argument("--device-class", choices=["auto", "laptop-desktop", "edge-device"], default="auto")
    parser.add_argument("--submitted-by", type=str, default="anonymous")
    parser.add_argument("--outdir", type=str, default=None,
                         help="root results directory (default: <repo-root>/results, regardless of cwd)")
    args = parser.parse_args()

    if whisper is None:
        sys.exit("Missing dependency. Run: pip install -r stt/requirements.txt")

    pairs = find_audio_reference_pairs(Path(args.audio_dir))
    if not pairs:
        sys.exit(f"No <name>.wav + <name>.txt pairs found in {args.audio_dir}")

    if args.sessions < 3:
        print(f"WARNING: --sessions {args.sessions} is below the recommended minimum of 3.")

    hw = get_hardware_info(power_source_override=args.power_source, device_class_override=args.device_class)
    nickname = args.device_nickname or hw["hostname"]

    print(f"Benchmarking Whisper[{args.model}] on {len(pairs)} audio file(s), {args.sessions} session(s)...")
    sessions = []
    for i in range(1, args.sessions + 1):
        print(f"  session {i}/{args.sessions}")
        sessions.append(run_session(args.model, pairs))

    agg = aggregate_stt_sessions(sessions)

    out_root = Path(args.outdir) if args.outdir else (SCRIPT_DIR.parent / "results")
    out_dir = classify_result_path(hw, nickname, out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "submitted_by": args.submitted_by,
        "submission_date": datetime.now().date().isoformat(),
        "hardware": hw,
        "test_config": {
            "engines_tested": [f"whisper:{args.model}"],
            "sessions": args.sessions,
            "audio_set": str(Path(args.audio_dir).resolve()),
        },
        "results": [{
            "engine": "whisper",
            "model": args.model,
            "sessions_raw": sessions,
            "aggregated": agg,
        }],
    }

    out_path = out_dir / "aggregated_results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    m = agg["metrics"]
    print(f"\nmodel_load_s: median={m['model_load_s'] and m['model_load_s']['median']}")
    print(f"rtf:          median={m['rtf'] and m['rtf']['median']}  (lower is better, <1.0 = faster than real-time)")
    print(f"wer:          median={m['wer'] and m['wer']['median']}  (lower is better)")
    if agg["data_quality_flags"]:
        print(f"flags: {', '.join(agg['data_quality_flags'])}")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
