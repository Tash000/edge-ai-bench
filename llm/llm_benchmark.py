#!/usr/bin/env python3
"""
edge-ai-bench — Local LLM Benchmark Suite

Targets Ollama's local REST API (http://localhost:11434). Designed for
comparing small/quantized local models on low-end and mid-range hardware,
including single-board edge devices (Raspberry Pi, Jetson, etc).

WHAT IT MEASURES (metric names follow inference-serving literature: NVIDIA's
LLM benchmarking series, AWS Neuron docs, DigitalOcean's LLM inference guide):

  1. Cold-load time    — time from "model not in memory" to first response.
                          Ollama evicts idle models, so this is your real
                          "boot to ready" cost after a power cycle or idle
                          timeout.
  2. TTFT               — Time To First Token. What the model "feels like"
                          to a human waiting for a reaction. Measured with
                          streaming responses, ignoring empty leading chunks.
  3. TPOT / ITL          — Time Per Output Token (inter-token latency).
                          (end_to_end - TTFT) / (num_output_tokens - 1).
  4. Tokens/sec (decode) — throughput once generation has started.
  5. End-to-end latency  — full wall-clock time for one turn.
  6. Persona adherence   — given a system persona + hard constraints
                          (forbidden words / address form / max sentences,
                          all declared in a YAML config), score whether the
                          model follows them.
  7. Output-format compliance — strict JSON, hybrid JSON+chat, and plain
                          conversational, checked against a validator per
                          test type across N trials.
  8. Intent / command detection — few-shot classification against a
                          swappable YAML config of utterance -> label cases.
  9. Memory/context recall — feeds a context blob with a fact buried in it,
                          asks a question later, checks recall.
  10. Resource footprint  — VRAM/RAM via `ollama ps`, used to confirm GPU
                          acceleration actually happened (not just present).

RELIABILITY: a single run is noise (thermal state, background load, swap).
This script runs the full suite across --sessions independent sessions
(default 3, minimum recommended for any community submission) and reports
median/mean/stdev per metric, flagging high-variance or single-session
results instead of silently trusting them.

OUTPUT: a single aggregated_results.json (schema in schemas/results_schema.json)
+ summary.csv + a self-contained report.html (charts, key findings, hardware
info — no external assets, works fully offline).

USAGE:
  python llm_benchmark.py --models llama3.2:1b llama3.2:3b qwen2.5:1.5b qwen:latest
  python llm_benchmark.py --models llama3.2:3b --sessions 5 --submitted-by "yourname"
  python llm_benchmark.py --report results/laptop-desktop/tier-2-low-ram/mybox/20260701_120000/aggregated_results.json

Requires: pip install -r requirements.txt   (requests, pyyaml)
Ollama must be running: `ollama serve` (usually auto-started).
"""

import argparse
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run: pip install -r requirements.txt")

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency (pyyaml). Run: pip install -r requirements.txt")

OLLAMA_URL = "http://localhost:11434"
GEN_ENDPOINT = f"{OLLAMA_URL}/api/generate"
PS_ENDPOINT = f"{OLLAMA_URL}/api/ps"

SCHEMA_VERSION = "1.0"
BENCHMARK_VERSION = "1.0"

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Low-level Ollama call wrapper with token-level timing
# --------------------------------------------------------------------------

@dataclass
class GenResult:
    ok: bool
    text: str = ""
    ttft: float = None          # seconds to first token
    total_time: float = None    # seconds, full request
    tpot: float = None          # seconds/token after first token
    tokens_out: int = None
    tokens_per_sec: float = None
    error: str = ""


def ollama_stream_generate(model: str, prompt: str, system: str = None,
                            keep_alive: str = "5m", options: dict = None,
                            timeout: int = 120) -> GenResult:
    """Stream a completion from Ollama and measure TTFT / TPOT precisely."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options

    t_start = time.perf_counter()
    first_token_time = None
    token_times = []
    chunks = []

    try:
        with requests.post(GEN_ENDPOINT, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("response", "")
                if piece:  # ignore empty leading chunks, per TTFT best practice
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_times.append(now)
                    chunks.append(piece)
                if obj.get("done"):
                    break
        t_end = time.perf_counter()
    except Exception as e:
        return GenResult(ok=False, error=str(e))

    if first_token_time is None:
        return GenResult(ok=False, error="No tokens received")

    ttft = first_token_time - t_start
    total_time = t_end - t_start
    n_tokens = len(token_times)
    tpot = ((t_end - first_token_time) / (n_tokens - 1)) if n_tokens > 1 else None
    tps = n_tokens / total_time if total_time > 0 else None

    return GenResult(
        ok=True,
        text="".join(chunks),
        ttft=ttft,
        total_time=total_time,
        tpot=tpot,
        tokens_out=n_tokens,
        tokens_per_sec=tps,
    )


# --------------------------------------------------------------------------
# Raw per-call log — every single generate call, untruncated, tagged with
# session/test/case/trial. aggregated_results.json only keeps medians and a
# couple of truncated samples, so if a mean ends up out of bounds there's no
# way to trace which specific call produced it. This log is the full record.
# --------------------------------------------------------------------------

RAW_CALL_LOG = []


def logged_generate(model: str, prompt: str, session_idx: int, test_name: str, case_id: str, trial_idx: int,
                     system: str = None, keep_alive: str = "10m", options: dict = None, timeout: int = 120) -> GenResult:
    res = ollama_stream_generate(model, prompt, system=system, keep_alive=keep_alive, options=options, timeout=timeout)
    RAW_CALL_LOG.append({
        "session": session_idx,
        "model": model,
        "test": test_name,
        "case": case_id,
        "trial": trial_idx,
        "timestamp": datetime.now().isoformat(),
        "swap_used_gb_at_call": _mem_info_gb().get("swap_used_gb"),
        "system_prompt": system,
        "prompt": prompt,
        "ok": res.ok,
        "error": res.error,
        "ttft_s": res.ttft,
        "total_time_s": res.total_time,
        "tpot_s_per_token": res.tpot,
        "tokens_out": res.tokens_out,
        "tokens_per_sec": res.tokens_per_sec,
        "response_text": res.text,
    })
    return res


def ollama_unload(model: str):
    """Force-unload a model from memory to simulate cold start."""
    try:
        requests.post(GEN_ENDPOINT, json={"model": model, "prompt": "", "keep_alive": 0}, timeout=30)
    except Exception:
        pass
    time.sleep(1)


def ollama_ps() -> list:
    try:
        r = requests.get(PS_ENDPOINT, timeout=10)
        return r.json().get("models", [])
    except Exception:
        return []


def check_ollama_alive():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=15)
    except Exception:
        sys.exit("Cannot reach Ollama at localhost:11434. Is `ollama serve` running?")


# --------------------------------------------------------------------------
# Hardware detection — every result file carries this so contributor
# submissions are self-describing and comparable.
# --------------------------------------------------------------------------

def _read_proc_cpuinfo_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _cpu_core_counts() -> tuple:
    logical = os.cpu_count() or 1
    physical = logical
    try:
        with open("/proc/cpuinfo") as f:
            ids = set()
            phys_id = core_id = None
            for line in f:
                line = line.strip()
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    if phys_id is not None and core_id is not None:
                        ids.add((phys_id, core_id))
            if ids:
                physical = len(ids)
    except Exception:
        pass
    return physical, logical


def _mem_info_gb() -> dict:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return {
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "swap_total_gb": round(sw.total / (1024 ** 3), 2),
            "swap_used_gb": round(sw.used / (1024 ** 3), 2),
        }
    except ImportError:
        pass
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k.strip()] = int(v.strip().split()[0])  # kB
        total = vals.get("MemTotal", 0)
        avail = vals.get("MemAvailable", 0)
        swap_total = vals.get("SwapTotal", 0)
        swap_free = vals.get("SwapFree", 0)
        return {
            "ram_total_gb": round(total / (1024 ** 2), 2),
            "ram_available_gb": round(avail / (1024 ** 2), 2),
            "swap_total_gb": round(swap_total / (1024 ** 2), 2),
            "swap_used_gb": round((swap_total - swap_free) / (1024 ** 2), 2),
        }
    except Exception:
        return {"ram_total_gb": None, "ram_available_gb": None,
                "swap_total_gb": None, "swap_used_gb": None}


def _detect_gpu() -> tuple:
    """Returns (gpu_model_or_None, gpu_vram_gb_or_None). Descriptive only —
    whether a given run actually used the GPU is decided per-session from
    `ollama ps` VRAM usage, not from hardware presence alone."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem = out.stdout.strip().split(",", 1)
            mem_mb = float(re.sub(r"[^\d.]", "", mem) or 0)
            return name.strip(), round(mem_mb / 1024, 2) if mem_mb else None
    except Exception:
        pass
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=15)
        for line in out.stdout.splitlines():
            if re.search(r"VGA|3D controller|Display controller", line):
                return line.split(":", 2)[-1].strip(), None
    except Exception:
        pass
    return None, None


def _detect_device_class() -> tuple:
    """Returns (device_class, device_type). Edge SBCs (Raspberry Pi, Jetson,
    etc) expose a model string at /proc/device-tree/model; x86 laptops and
    desktops do not — no matter how low-end the x86 box is, it still has
    more of everything (RAM, storage, cooling, expandable I/O) than a
    fixed-spec SBC, so the two are kept as separate result categories."""
    dt_model_path = Path("/proc/device-tree/model")
    if dt_model_path.exists():
        try:
            model = dt_model_path.read_bytes().split(b"\x00")[0].decode(errors="ignore").strip()
            if model:
                return "edge_device", model
        except Exception:
            pass
    if platform.machine().lower() in ("aarch64", "armv7l", "arm64"):
        return "edge_device", f"unknown ARM board ({platform.machine()})"
    return "laptop_desktop", None


def _detect_power_source() -> str:
    supply_dir = Path("/sys/class/power_supply")
    if not supply_dir.exists():
        return "unknown"
    try:
        has_battery = False
        ac_online = False
        for entry in supply_dir.iterdir():
            type_file = entry / "type"
            online_file = entry / "online"
            if type_file.exists() and "battery" in type_file.read_text().strip().lower():
                has_battery = True
            if online_file.exists() and online_file.read_text().strip() == "1":
                ac_online = True
        if not has_battery:
            return "ac"  # desktop / no battery present at all
        return "ac" if ac_online else "battery"
    except Exception:
        return "unknown"


def _ollama_version() -> Optional[str]:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=15)
        return (out.stdout.strip() or out.stderr.strip()) or None
    except Exception:
        return None


def get_hardware_info(power_source_override: str = "auto", device_class_override: str = "auto") -> dict:
    cpu_model = _read_proc_cpuinfo_model() or platform.processor() or platform.machine()
    physical, logical = _cpu_core_counts()
    mem = _mem_info_gb()
    gpu_model, gpu_vram_gb = _detect_gpu()
    device_class, device_type = _detect_device_class()
    if device_class_override != "auto":
        device_class = device_class_override.replace("-", "_")
    power_source = _detect_power_source() if power_source_override == "auto" else power_source_override

    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_cores_physical": physical,
        "cpu_threads_logical": logical,
        **mem,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine_arch": platform.machine(),
        "device_class": device_class,      # "laptop_desktop" | "edge_device"
        "device_type": device_type,        # e.g. "Raspberry Pi 4 Model B Rev 1.4" or None
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram_gb,
        "ollama_version": _ollama_version(),
        "power_source": power_source,      # "ac" | "battery" | "unknown"
    }


def classify_result_path(hw: dict, gpu_accelerated_any: bool, nickname: str, out_root: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", nickname.lower()).strip("-") or "device"
    if hw.get("device_class") == "edge_device":
        dt = (hw.get("device_type") or "").lower()
        if "raspberry" in dt:
            sub = "raspberry-pi"
        elif "jetson" in dt or "tegra" in dt:
            sub = "jetson-nano"
        else:
            sub = "other-sbc"
        return out_root / "edge-device" / sub / slug

    ram = hw.get("ram_total_gb") or 0
    if gpu_accelerated_any:
        tier = "tier-4-gpu-accelerated"
    elif ram <= 4:
        tier = "tier-1-ultra-low-ram"
    elif ram <= 8:
        tier = "tier-2-low-ram"
    else:
        tier = "tier-3-mid-ram"
    return out_root / "laptop-desktop" / tier / slug


# --------------------------------------------------------------------------
# Test 1: Cold-load / boot time
# --------------------------------------------------------------------------

def test_cold_load(model: str, num_ctx: int, session_idx: int) -> dict:
    ollama_unload(model)
    t0 = time.perf_counter()
    res = logged_generate(model, "Say OK.", session_idx, "cold_load", "cold_load", 1,
                           keep_alive="5m", options={"num_predict": 5, "num_ctx": num_ctx})
    t1 = time.perf_counter()
    return {
        "cold_load_to_first_response_s": round(t1 - t0, 3) if res.ok else None,
        "ok": res.ok,
        "error": res.error,
    }


# --------------------------------------------------------------------------
# Test 2: Warm latency (TTFT, TPOT, tokens/sec) over N runs
# --------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "Walk forward two steps and wave with your right hand.",
    "What is the weather like today?",
    "Pick up the red cup from the table and hand it to me.",
    "Introduce yourself in one sentence.",
]


def test_warm_latency(model: str, prompts: list, runs: int, num_ctx: int, session_idx: int, system: str = None) -> dict:
    per_prompt = []
    all_ttft, all_tpot, all_tps, all_total = [], [], [], []
    for p in prompts:
        trial_results = []
        for trial in range(1, runs + 1):
            res = logged_generate(model, p, session_idx, "warm_latency", p, trial, system=system, keep_alive="10m",
                                   options={"num_predict": 128, "temperature": 0.3, "num_ctx": num_ctx})
            if res.ok:
                trial_results.append(res)
                all_ttft.append(res.ttft)
                if res.tpot:
                    all_tpot.append(res.tpot)
                if res.tokens_per_sec:
                    all_tps.append(res.tokens_per_sec)
                all_total.append(res.total_time)
        per_prompt.append({
            "prompt": p,
            "n_ok": len(trial_results),
            "avg_ttft_s": round(statistics.mean([r.ttft for r in trial_results]), 4) if trial_results else None,
        })

    def stats(vals):
        if not vals:
            return None
        return {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "p95": round(sorted(vals)[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0], 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    return {
        "ttft_s": stats(all_ttft),
        "tpot_s_per_token": stats(all_tpot),
        "tokens_per_sec": stats(all_tps),
        "end_to_end_s": stats(all_total),
        "per_prompt": per_prompt,
    }


# --------------------------------------------------------------------------
# Test 3: Output-format compliance (strict JSON / hybrid / conversational)
# --------------------------------------------------------------------------

FORMAT_TESTS = [
    {
        "name": "strict_json_mission",
        "system": (
            "You are a task-planning assistant. Respond with ONLY valid JSON, "
            "no prose, no markdown fences. Schema: "
            '{"action": str, "target": str, "params": object}'
        ),
        "prompt": "Walk to the kitchen and pick up the blue bottle.",
        "validator": "strict_json",
    },
    {
        "name": "hybrid_json_plus_chat",
        "system": (
            "You are a helpful assistant. Reply naturally in conversation, "
            "but ALSO include a JSON block at the end wrapped in <action></action> "
            'tags with schema {"action": str, "target": str}.'
        ),
        "prompt": "Can you grab my phone from the couch?",
        "validator": "hybrid_json",
    },
    {
        "name": "plain_conversational",
        "system": "You are a friendly assistant. Respond naturally, no JSON, no lists.",
        "prompt": "How was your day?",
        "validator": "plain_text",
    },
]


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _validate_strict_json(text: str) -> dict:
    obj = _extract_json(text)
    ok = obj is not None and isinstance(obj, dict) and "action" in obj
    return {"pass": ok, "parsed": obj is not None}


def _validate_hybrid_json(text: str) -> dict:
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    has_chat = len(text.strip()) > (len(m.group(0)) if m else 0)
    obj = None
    if m:
        obj = _extract_json(m.group(1))
    return {"pass": bool(m and obj and has_chat), "parsed": obj is not None, "has_chat_text": has_chat}


def _validate_plain_text(text: str) -> dict:
    has_json_leak = bool(re.search(r"[{}\[\]]", text))
    return {"pass": not has_json_leak and len(text.strip()) > 0, "json_leak": has_json_leak}


VALIDATORS = {
    "strict_json": _validate_strict_json,
    "hybrid_json": _validate_hybrid_json,
    "plain_text": _validate_plain_text,
}


def test_format_compliance(model: str, runs: int, num_ctx: int, session_idx: int) -> dict:
    results = {}
    for spec in FORMAT_TESTS:
        passes = 0
        samples = []
        for trial in range(1, runs + 1):
            res = logged_generate(model, spec["prompt"], session_idx, "format_compliance", spec["name"], trial,
                                   system=spec["system"], keep_alive="10m",
                                   options={"num_predict": 200, "temperature": 0.4, "num_ctx": num_ctx})
            if not res.ok:
                continue
            verdict = VALIDATORS[spec["validator"]](res.text)
            passes += int(verdict["pass"])
            samples.append({"raw": res.text[:300], "verdict": verdict})
        results[spec["name"]] = {
            "pass_rate": round(passes / runs, 2) if runs else None,
            "samples": samples[:2],
        }
    return results


# --------------------------------------------------------------------------
# Persona loading + per-persona constraint scoring
# --------------------------------------------------------------------------

PERSONA_PROBES = [
    "Tell me about yourself.",
    "I dropped my coffee everywhere, what should I do?",
    "Can you do my homework for me?",
    "Turn off the lights and tell me a joke.",
]


def load_personas_from_dir(persona_dir: Path) -> dict:
    """
    Loads every .yaml/.yml in the folder. Schema is intentionally treated as
    loose — we pull whatever fields exist under common key names, since
    persona YAMLs vary project to project. Unrecognized keys are kept under
    'raw_keys_found' so nothing is silently dropped.
    """
    personas = {}
    for path in sorted(Path(persona_dir).glob("*.y*ml")):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        name = raw.get("name") or raw.get("persona_name") or path.stem
        system_prompt = (
            raw.get("system_prompt") or raw.get("prompt") or raw.get("system")
            or raw.get("description") or ""
        )
        tone = raw.get("tone") or raw.get("style") or ""
        forbidden_words = raw.get("forbidden_words") or raw.get("avoid") or raw.get("never_say") or []
        required_address = raw.get("address_user_as") or raw.get("user_title") or raw.get("call_user") or None
        max_sentences = raw.get("max_sentences") or raw.get("max_response_sentences") or None
        traits = raw.get("traits") or raw.get("personality_traits") or []

        if not system_prompt:
            parts = [f"You are {name}, a helpful assistant."]
            if tone:
                parts.append(f"Tone: {tone}.")
            if traits:
                parts.append(f"Traits: {', '.join(traits)}.")
            if required_address:
                parts.append(f"Always refer to the user as '{required_address}'.")
            if forbidden_words:
                parts.append(f"Never use these words: {', '.join(forbidden_words)}.")
            if max_sentences:
                parts.append(f"Keep responses to at most {max_sentences} sentences.")
            system_prompt = " ".join(parts)

        personas[path.stem] = {
            "file": str(path),
            "name": name,
            "system_prompt": system_prompt,
            "tone": tone,
            "forbidden_words": forbidden_words,
            "required_address": required_address,
            "max_sentences": max_sentences,
            "traits": traits,
            "raw_keys_found": list(raw.keys()),
        }
    return personas


def _score_persona_response(text: str, persona: dict) -> dict:
    """Cheap, deterministic checks — catches hard constraint violations
    reliably, but is not a substitute for human/LLM judgment on tone."""
    text_l = text.lower()
    violations = []

    for w in persona.get("forbidden_words") or []:
        if w.lower() in text_l:
            violations.append(f"used forbidden word '{w}'")

    if persona.get("required_address"):
        if persona["required_address"].lower() not in text_l:
            violations.append(f"did not address user as '{persona['required_address']}'")

    if persona.get("max_sentences"):
        n_sentences = len(re.findall(r"[.!?](?:\s|$)", text.strip()))
        if n_sentences > persona["max_sentences"]:
            violations.append(f"{n_sentences} sentences > max {persona['max_sentences']}")

    return {"violations": violations, "pass": len(violations) == 0}


def test_persona_adherence_multi(model: str, personas: dict, runs: int, num_ctx: int, session_idx: int) -> dict:
    results = {}
    for key, persona in personas.items():
        samples = []
        passes = 0
        total = 0
        for p in PERSONA_PROBES:
            for trial in range(1, runs + 1):
                res = logged_generate(model, p, session_idx, "persona_adherence", f"{key}::{p}", trial,
                                       system=persona["system_prompt"], keep_alive="10m",
                                       options={"num_predict": 150, "temperature": 0.5, "num_ctx": num_ctx})
                if not res.ok:
                    continue
                verdict = _score_persona_response(res.text, persona)
                total += 1
                passes += int(verdict["pass"])
                samples.append({"prompt": p, "response": res.text[:350], "verdict": verdict})
        results[key] = {
            "persona_name": persona["name"],
            "constraint_pass_rate": round(passes / total, 2) if total else None,
            "checked_constraints": {
                "forbidden_words": persona.get("forbidden_words"),
                "required_address": persona.get("required_address"),
                "max_sentences": persona.get("max_sentences"),
            },
            "note": "constraint_pass_rate only checks hard rules declared in the yaml "
                    "(forbidden words / address form / max sentences). Tone and "
                    "in-character quality still need a human read of 'samples'.",
            "samples": samples[:3],
        }
    return results


# --------------------------------------------------------------------------
# Test: Intent / command-word detection (few-shot, config-driven)
# --------------------------------------------------------------------------

def load_intent_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    test_cases = [(tc["utterance"], tc["expected"]) for tc in raw.get("test_cases", [])]
    if not test_cases:
        sys.exit(f"Intent config {path} has no test_cases")
    return {
        "name": raw.get("name", Path(path).stem),
        "system_prompt": raw["system_prompt"],
        "labels": raw.get("labels", []),
        "test_cases": test_cases,
    }


def test_intent_detection(model: str, intent_cfg: dict, num_ctx: int, session_idx: int) -> dict:
    correct = 0
    details = []
    cases = intent_cfg["test_cases"]
    for i, (utt, expected) in enumerate(cases, 1):
        res = logged_generate(model, utt, session_idx, "intent_detection", utt, i,
                               system=intent_cfg["system_prompt"], keep_alive="10m",
                               options={"num_predict": 10, "temperature": 0.0, "num_ctx": num_ctx})
        pred = res.text.strip().split()[0].upper().strip(".:,") if res.ok and res.text.strip() else "ERROR"
        is_correct = pred == expected
        correct += int(is_correct)
        details.append({"utterance": utt, "expected": expected, "predicted": pred, "correct": is_correct})
    return {
        "config_name": intent_cfg["name"],
        "accuracy": round(correct / len(cases), 2) if cases else None,
        "n_cases": len(cases),
        "details": details,
    }


# --------------------------------------------------------------------------
# Test: Memory / context recall
# --------------------------------------------------------------------------

def test_context_recall(model: str, context_text: str, num_ctx: int, session_idx: int) -> dict:
    fact_marker = "BENCH_FAVORITE_COLOR_IS_TEAL_42"
    injected_context = context_text + f"\n\n[fact: the owner's favorite color is teal-42, code {fact_marker}]"
    res = logged_generate(
        model,
        "Based on everything above, what is my favorite color? Answer in one short sentence.",
        session_idx, "context_recall", "context_recall", 1,
        system=injected_context,
        keep_alive="10m",
        options={"num_predict": 60, "temperature": 0.0, "num_ctx": num_ctx},
    )
    recalled = res.ok and "teal" in res.text.lower()
    return {"recalled_injected_fact": recalled, "response": res.text if res.ok else res.error}


# --------------------------------------------------------------------------
# Resource footprint
# --------------------------------------------------------------------------

def get_resource_snapshot(model: str) -> dict:
    procs = ollama_ps()
    for m in procs:
        if m.get("name") == model or m.get("model") == model:
            return {
                "size_vram_bytes": m.get("size_vram"),
                "size_bytes": m.get("size"),
                "expires_at": m.get("expires_at"),
            }
    return {"size_vram_bytes": None, "size_bytes": None, "expires_at": None, "note": "model not currently loaded"}


# --------------------------------------------------------------------------
# One full benchmark session (all tests, one model)
# --------------------------------------------------------------------------

def run_benchmark_session(model: str, runs: int, personas: dict, intent_cfg: dict,
                           context_text: str, num_ctx: int, session_idx: int, total_sessions: int) -> dict:
    print(f"  session {session_idx}/{total_sessions}")
    swap_before = _mem_info_gb().get("swap_used_gb")

    print("    [1/6] cold load...")
    cold = test_cold_load(model, num_ctx, session_idx)

    print("    [2/6] warm latency (TTFT/TPOT/tokens-per-sec)...")
    latency = test_warm_latency(model, DEFAULT_PROMPTS, runs, num_ctx, session_idx)

    print("    [3/6] output-format compliance...")
    fmt = test_format_compliance(model, runs, num_ctx, session_idx)

    print(f"    [4/6] persona adherence across {len(personas)} persona(s)...")
    persona_results = test_persona_adherence_multi(model, personas, max(1, runs // 2), num_ctx, session_idx)

    print("    [5/6] intent/command detection...")
    intent = test_intent_detection(model, intent_cfg, num_ctx, session_idx)

    print("    [6/6] context/memory recall...")
    recall = test_context_recall(model, context_text, num_ctx, session_idx)

    resources = get_resource_snapshot(model)
    gpu_accelerated = bool(resources.get("size_vram_bytes"))

    return {
        "session": session_idx,
        "timestamp": datetime.now().isoformat(),
        "swap_used_gb_at_start": swap_before,
        "cold_load": cold,
        "warm_latency": latency,
        "format_compliance": fmt,
        "persona_adherence": persona_results,
        "intent_detection": intent,
        "context_recall": recall,
        "resources": resources,
        "gpu_accelerated": gpu_accelerated,
    }


# --------------------------------------------------------------------------
# Cross-session aggregation
# --------------------------------------------------------------------------

def _stats(vals):
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


def aggregate_sessions(sessions: list) -> dict:
    n = len(sessions)
    flags = []
    if n < 3:
        flags.append(f"insufficient_sessions:{n}_of_3_minimum")

    cold_vals = [s["cold_load"].get("cold_load_to_first_response_s") for s in sessions]
    ttft_vals = [(s["warm_latency"]["ttft_s"] or {}).get("mean") for s in sessions]
    tpot_vals = [(s["warm_latency"]["tpot_s_per_token"] or {}).get("mean") for s in sessions]
    tps_vals = [(s["warm_latency"]["tokens_per_sec"] or {}).get("mean") for s in sessions]
    e2e_vals = [(s["warm_latency"]["end_to_end_s"] or {}).get("mean") for s in sessions]

    metrics = {
        "cold_load_s": _stats(cold_vals),
        "ttft_s": _stats(ttft_vals),
        "tpot_s_per_token": _stats(tpot_vals),
        "tokens_per_sec": _stats(tps_vals),
        "end_to_end_s": _stats(e2e_vals),
    }
    for name, stat in metrics.items():
        if stat and stat["cv_pct"] is not None and stat["cv_pct"] > 25:
            flags.append(f"high_variance:{name}({stat['cv_pct']}%)")

    fmt_names = sorted({k for s in sessions for k in s["format_compliance"].keys()})
    format_compliance = {}
    for name in fmt_names:
        rates = [s["format_compliance"].get(name, {}).get("pass_rate") for s in sessions]
        rates = [r for r in rates if r is not None]
        format_compliance[name] = {
            "pass_rate_median": round(statistics.median(rates), 2) if rates else None,
            "pass_rate_per_session": rates,
        }

    persona_keys = sorted({k for s in sessions for k in s["persona_adherence"].keys()})
    persona_adherence = {}
    for key in persona_keys:
        rates = [s["persona_adherence"].get(key, {}).get("constraint_pass_rate") for s in sessions]
        rates = [r for r in rates if r is not None]
        persona_adherence[key] = {
            "constraint_pass_rate_median": round(statistics.median(rates), 2) if rates else None,
            "constraint_pass_rate_per_session": rates,
        }

    intent_accs = [s["intent_detection"].get("accuracy") for s in sessions
                   if s["intent_detection"].get("accuracy") is not None]
    intent_detection = {
        "config_name": sessions[0]["intent_detection"].get("config_name") if sessions else None,
        "accuracy_median": round(statistics.median(intent_accs), 2) if intent_accs else None,
        "accuracy_per_session": intent_accs,
    }

    recall_flags = [bool(s["context_recall"].get("recalled_injected_fact")) for s in sessions]
    context_recall = {
        "recalled_rate": round(sum(recall_flags) / len(recall_flags), 2) if recall_flags else None,
        "per_session": recall_flags,
    }

    vram_vals = [s["resources"].get("size_vram_bytes") for s in sessions
                 if isinstance(s["resources"].get("size_vram_bytes"), (int, float))]
    ram_vals = [s["resources"].get("size_bytes") for s in sessions
                if isinstance(s["resources"].get("size_bytes"), (int, float))]
    gpu_accelerated_any = any(s.get("gpu_accelerated") for s in sessions)

    swap_vals = [s.get("swap_used_gb_at_start") for s in sessions if s.get("swap_used_gb_at_start") is not None]
    if any(v and v > 0.5 for v in swap_vals):
        flags.append("swap_active_during_test")

    return {
        "metrics": metrics,
        "format_compliance": format_compliance,
        "persona_adherence": persona_adherence,
        "intent_detection": intent_detection,
        "context_recall": context_recall,
        "resources": {
            "vram_bytes_median": round(statistics.median(vram_vals)) if vram_vals else None,
            "ram_bytes_median": round(statistics.median(ram_vals)) if ram_vals else None,
            "gpu_accelerated": gpu_accelerated_any,
        },
        "data_quality_flags": flags,
    }


# --------------------------------------------------------------------------
# Summary table (console)
# --------------------------------------------------------------------------

def print_summary_table(payload: dict):
    results = payload["results"]
    print(f"\n{'='*115}\nSUMMARY (median across {payload['test_config']['sessions']} session(s))\n{'='*115}")
    header = (f"{'Model':<16}{'Cold load(s)':<14}{'TTFT mean(s)':<14}{'TPOT(ms/tok)':<14}"
              f"{'Tok/s':<8}{'JSON pass%':<12}{'Persona%':<10}{'Intent acc':<11}{'Recall':<8}{'Flags'}")
    print(header)
    print("-" * len(header))
    for r in results:
        agg = r["aggregated"]
        cold = (agg["metrics"]["cold_load_s"] or {}).get("median")
        ttft = (agg["metrics"]["ttft_s"] or {}).get("median")
        tpot = (agg["metrics"]["tpot_s_per_token"] or {}).get("median")
        tps = (agg["metrics"]["tokens_per_sec"] or {}).get("median")
        json_rates = [v["pass_rate_median"] for v in agg["format_compliance"].values() if v["pass_rate_median"] is not None]
        json_avg = round(sum(json_rates) / len(json_rates) * 100) if json_rates else None
        persona_rates = [v["constraint_pass_rate_median"] for v in agg["persona_adherence"].values()
                          if v["constraint_pass_rate_median"] is not None]
        persona_avg = round(sum(persona_rates) / len(persona_rates) * 100) if persona_rates else None
        intent_acc = agg["intent_detection"].get("accuracy_median")
        recall = agg["context_recall"].get("recalled_rate")
        flags = ",".join(agg["data_quality_flags"]) or "-"
        print(f"{r['model']:<16}{str(cold):<14}{str(ttft):<14}"
              f"{(str(round(tpot*1000,1)) if tpot else 'N/A'):<14}"
              f"{str(round(tps,1) if tps else 'N/A'):<8}"
              f"{str(json_avg)+'%':<12}{str(persona_avg)+'%':<10}{str(intent_acc):<11}{str(recall):<8}{flags}")
    print("\nLower is better: Cold load, TTFT, TPOT. Higher is better: Tok/s, JSON pass%, Persona%, Intent acc, Recall.")
    if any(r["aggregated"]["data_quality_flags"] for r in results):
        print("Flags present on one or more models — see report.html for details before trusting these numbers.")


# --------------------------------------------------------------------------
# HTML report generator — self-contained, no external assets, works offline.
# --------------------------------------------------------------------------

def svg_bar_chart(title: str, items: list, unit: str = "", fmt: str = "{:.2f}",
                   width: int = 560, bar_height: int = 26, gap: int = 10) -> str:
    items = [(label, val) for label, val in items if val is not None]
    if not items:
        return f'<div class="chart"><h3>{title}</h3><p class="muted">no data</p></div>'
    max_val = max(val for _, val in items) or 1
    height = len(items) * (bar_height + gap) + 20
    bars = []
    for i, (label, val) in enumerate(items):
        y = 10 + i * (bar_height + gap)
        bar_w = max(2.0, (val / max_val) * (width - 190))
        bars.append(
            f'<text x="0" y="{y + bar_height/2 + 5}" font-size="13" fill="#333">{label}</text>'
            f'<rect x="150" y="{y}" width="{bar_w:.1f}" height="{bar_height}" rx="4" fill="#4f8ef7"/>'
            f'<text x="{150 + bar_w + 8:.1f}" y="{y + bar_height/2 + 5}" font-size="12" fill="#111">{fmt.format(val)}{unit}</text>'
        )
    svg = (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>')
    return f'<div class="chart"><h3>{title}</h3>{svg}</div>'


def _model_summary_rows(results: list) -> list:
    rows = []
    for r in results:
        agg = r["aggregated"]
        rates = [v["pass_rate_median"] for v in agg["format_compliance"].values() if v["pass_rate_median"] is not None]
        json_avg = round(sum(rates) / len(rates) * 100, 1) if rates else None
        rows.append({
            "model": r["model"],
            "cold_load": (agg["metrics"]["cold_load_s"] or {}).get("median"),
            "ttft": (agg["metrics"]["ttft_s"] or {}).get("median"),
            "tpot": (agg["metrics"]["tpot_s_per_token"] or {}).get("median"),
            "tps": (agg["metrics"]["tokens_per_sec"] or {}).get("median"),
            "json_avg": json_avg,
            "intent_acc": agg["intent_detection"].get("accuracy_median"),
            "recall": agg["context_recall"].get("recalled_rate"),
            "flags": agg["data_quality_flags"],
        })
    return rows


def compute_key_findings(results: list) -> list:
    rows = _model_summary_rows(results)
    findings = []

    valid_tps = [r for r in rows if r["tps"] is not None]
    if valid_tps:
        fastest = max(valid_tps, key=lambda r: r["tps"])
        findings.append(f"Fastest decode throughput: <b>{fastest['model']}</b> at {fastest['tps']:.1f} tok/s.")

    valid_cold = [r for r in rows if r["cold_load"] is not None]
    if valid_cold:
        quick = min(valid_cold, key=lambda r: r["cold_load"])
        findings.append(f"Fastest cold load: <b>{quick['model']}</b> at {quick['cold_load']:.1f}s.")

    valid_json = [r for r in rows if r["json_avg"] is not None]
    if valid_json:
        best_json = max(valid_json, key=lambda r: r["json_avg"])
        findings.append(f"Most reliable JSON/structured-output compliance: <b>{best_json['model']}</b> at {best_json['json_avg']:.0f}%.")
        for w in [r for r in valid_json if r["json_avg"] < 50]:
            findings.append(f"⚠ <b>{w['model']}</b> passed only {w['json_avg']:.0f}% of JSON-format tests "
                             f"— not recommended where strict parsing is required.")

    valid_intent = [r for r in rows if r["intent_acc"] is not None]
    if valid_intent:
        best_intent = max(valid_intent, key=lambda r: r["intent_acc"])
        findings.append(f"Best intent-detection accuracy: <b>{best_intent['model']}</b> at {best_intent['intent_acc']*100:.0f}%.")

    def norm(vals, invert=False):
        present = [v for v in vals if v is not None]
        if not present or max(present) == min(present):
            return [0.5 if v is not None else 0.0 for v in vals]
        lo, hi = min(present), max(present)
        out = []
        for v in vals:
            if v is None:
                out.append(0.0)
                continue
            n = (v - lo) / (hi - lo)
            out.append((1 - n) if invert else n)
        return out

    tps_n = norm([r["tps"] for r in rows])
    cold_n = norm([r["cold_load"] for r in rows], invert=True)
    json_n = norm([r["json_avg"] for r in rows])
    intent_n = norm([r["intent_acc"] for r in rows])
    scored = [(r["model"], 0.35 * tps_n[i] + 0.15 * cold_n[i] + 0.3 * json_n[i] + 0.2 * intent_n[i])
              for i, r in enumerate(rows)]
    if scored:
        best = max(scored, key=lambda t: t[1])
        findings.append("Overall recommended pick (heuristic score — speed 35%, JSON compliance 30%, "
                         f"intent accuracy 20%, cold load 15%): <b>{best[0]}</b>.")

    if any(r["flags"] for r in rows):
        findings.append("⚠ One or more models have data-quality flags (see banner above) — "
                         "treat those numbers as provisional.")

    return findings


def generate_html_report(json_path: Path, out_path: Path = None) -> Path:
    json_path = Path(json_path)
    payload = json.loads(json_path.read_text())
    hw = payload.get("hardware", {})
    results = payload.get("results", [])
    out_path = Path(out_path) if out_path else json_path.with_name("report.html")

    all_flags = sorted({f for r in results for f in r["aggregated"].get("data_quality_flags", [])})
    findings = compute_key_findings(results)
    rows = _model_summary_rows(results)

    charts_html = "".join([
        svg_bar_chart("Cold load time (s, lower is better)", [(r["model"], r["cold_load"]) for r in rows], unit="s"),
        svg_bar_chart("TTFT (s, lower is better)", [(r["model"], r["ttft"]) for r in rows], unit="s"),
        svg_bar_chart("Decode throughput (tok/s, higher is better)", [(r["model"], r["tps"]) for r in rows], unit=" tok/s", fmt="{:.1f}"),
        svg_bar_chart("JSON format compliance (%, higher is better)", [(r["model"], r["json_avg"]) for r in rows], unit="%", fmt="{:.0f}"),
        svg_bar_chart("Intent detection accuracy (%, higher is better)",
                      [(r["model"], (r["intent_acc"] * 100) if r["intent_acc"] is not None else None) for r in rows],
                      unit="%", fmt="{:.0f}"),
    ])

    device_label = hw.get("device_type") or hw.get("hostname", "unknown device")
    hw_card = f'''<div class="hw-card"><h2>Hardware</h2><table>
      <tr><td>Device</td><td>{device_label} ({hw.get('device_class','?')})</td></tr>
      <tr><td>CPU</td><td>{hw.get('cpu_model','?')} ({hw.get('cpu_cores_physical','?')}C / {hw.get('cpu_threads_logical','?')}T)</td></tr>
      <tr><td>RAM</td><td>{hw.get('ram_total_gb','?')} GB (swap used at test time: {hw.get('swap_used_gb','?')} GB)</td></tr>
      <tr><td>GPU</td><td>{hw.get('gpu_model') or 'none / integrated'}</td></tr>
      <tr><td>OS</td><td>{hw.get('os','?')} {hw.get('os_release','')}</td></tr>
      <tr><td>Ollama</td><td>{hw.get('ollama_version','?')}</td></tr>
      <tr><td>Power source</td><td>{hw.get('power_source','?')}</td></tr>
    </table></div>'''

    banner_html = ""
    if all_flags:
        items = "".join(f"<li>{f}</li>" for f in all_flags)
        banner_html = f'<div class="banner">⚠ Data-quality flags present<ul>{items}</ul></div>'

    findings_html = "<ul>" + "".join(f"<li>{f}</li>" for f in findings) + "</ul>" if findings else "<p class='muted'>Not enough data.</p>"

    rows_html = ""
    for r in rows:
        rows_html += (
            "<tr>"
            f"<td>{r['model']}</td>"
            f"<td>{r['cold_load'] if r['cold_load'] is not None else 'N/A'}</td>"
            f"<td>{r['ttft'] if r['ttft'] is not None else 'N/A'}</td>"
            f"<td>{round(r['tpot']*1000,1) if r['tpot'] else 'N/A'}</td>"
            f"<td>{round(r['tps'],1) if r['tps'] else 'N/A'}</td>"
            f"<td>{r['json_avg'] if r['json_avg'] is not None else 'N/A'}%</td>"
            f"<td>{round(r['intent_acc']*100) if r['intent_acc'] is not None else 'N/A'}%</td>"
            f"<td>{round(r['recall']*100) if r['recall'] is not None else 'N/A'}%</td>"
            f"<td>{', '.join(r['flags']) or '—'}</td>"
            "</tr>"
        )

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>edge-ai-bench report — {device_label}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; background:#fafafa; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2rem; border-bottom: 2px solid #eee; padding-bottom: .3rem; }}
  h3 {{ font-size: 1rem; margin-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; }}
  td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: .9rem; }}
  th {{ background: #f0f2f5; }}
  .hw-card table td:first-child {{ font-weight: 600; width: 140px; color:#555; }}
  .banner {{ background: #fff4e5; border: 1px solid #f0b429; padding: .8rem 1rem; border-radius: 6px; margin: 1rem 0; }}
  .chart {{ margin-bottom: 1.2rem; }}
  .muted {{ color: #888; }}
  .footer {{ color: #888; font-size: .8rem; margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem; }}
</style></head>
<body>
  <h1>edge-ai-bench report</h1>
  <p>Generated {datetime.now().isoformat(timespec='seconds')} &middot; schema v{payload.get('schema_version','?')}
     &middot; benchmark v{payload.get('benchmark_version','?')} &middot; submitted by {payload.get('submitted_by','anonymous')}</p>
  {banner_html}
  {hw_card}
  <h2>Key findings</h2>
  {findings_html}
  <h2>Charts</h2>
  {charts_html}
  <h2>Full results (median across {payload.get('test_config',{}).get('sessions','?')} sessions)</h2>
  <table>
    <tr><th>Model</th><th>Cold load (s)</th><th>TTFT (s)</th><th>TPOT (ms/tok)</th><th>Tok/s</th>
        <th>JSON pass%</th><th>Intent acc</th><th>Recall</th><th>Data-quality flags</th></tr>
    {rows_html}
  </table>
  <div class="footer">
    Lower is better: cold load, TTFT, TPOT. Higher is better: tok/s, JSON pass%, intent accuracy, recall.
    This is a heuristic community benchmark, not a certified evaluation — see docs/BENCHMARK_METHODOLOGY.md.
  </div>
</body></html>'''

    out_path.write_text(html)
    return out_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="edge-ai-bench: benchmark local Ollama models on low-end hardware.")
    parser.add_argument("--models", nargs="+", help="e.g. --models llama3.2:1b llama3.2:3b qwen2.5:1.5b")
    parser.add_argument("--runs", type=int, default=3, help="trials per test within a session (default 3)")
    parser.add_argument("--sessions", type=int, default=3,
                         help="independent full benchmark sessions to run and aggregate. "
                              ">=3 required for community result submissions (default 3)")
    parser.add_argument("--persona-dir", type=str, default=None,
                         help="folder of persona .yaml files (default: bundled generic persona)")
    parser.add_argument("--intent-config", type=str, default=None,
                         help="path to an intent-detection .yaml config (default: bundled generic intents)")
    parser.add_argument("--context", type=str, default=None,
                         help="path to a context/memory .txt file (default: bundled generic context)")
    parser.add_argument("--num-ctx", type=int, default=2048, help="context window passed to Ollama (default 2048)")
    parser.add_argument("--outdir", type=str, default=None,
                         help="root results directory (default: <repo-root>/results, regardless of cwd)")
    parser.add_argument("--device-nickname", type=str, default=None, help="folder name for this device (default: hostname)")
    parser.add_argument("--device-class", choices=["auto", "laptop-desktop", "edge-device"], default="auto")
    parser.add_argument("--power-source", choices=["auto", "ac", "battery", "unknown"], default="auto")
    parser.add_argument("--submitted-by", type=str, default="anonymous", help="your name/handle, recorded in the results file")
    parser.add_argument("--report", type=str, default=None,
                         help="path to an existing aggregated_results.json; generates report.html and exits "
                              "(no Ollama, no benchmarking, works on any machine)")
    parser.add_argument("--report-out", type=str, default=None, help="output path for --report (default: report.html next to the input)")
    args = parser.parse_args()

    if args.report:
        out = generate_html_report(Path(args.report), Path(args.report_out) if args.report_out else None)
        print(f"Report written to {out}")
        return

    if not args.models:
        parser.error("--models is required unless --report is given")

    check_ollama_alive()

    persona_dir = Path(args.persona_dir).expanduser() if args.persona_dir else SCRIPT_DIR / "configs" / "personas" / "default_pack"
    intent_path = Path(args.intent_config).expanduser() if args.intent_config else SCRIPT_DIR / "configs" / "intents" / "default_intents.yaml"
    context_path = Path(args.context).expanduser() if args.context else SCRIPT_DIR / "configs" / "context" / "default_context.txt"

    personas = load_personas_from_dir(persona_dir)
    if not personas:
        sys.exit(f"No .yaml/.yml persona files found in {persona_dir}")
    intent_cfg = load_intent_config(intent_path)
    context_text = context_path.read_text()

    if args.sessions < 3:
        print(f"WARNING: --sessions {args.sessions} is below the recommended minimum of 3. "
              f"This run will be flagged 'insufficient_sessions' and is not accepted for community PRs.")

    hw = get_hardware_info(power_source_override=args.power_source, device_class_override=args.device_class)
    nickname = args.device_nickname or hw["hostname"]

    # Interleaved by session (not grouped by model): looping all sessions for
    # one model before moving to the next would let the OS page cache stay
    # hot between that model's own back-to-back runs, making "cold load"
    # measure a warm disk-cache reload instead of a real cold start. Running
    # one session per model, then cycling back, forces every other model's
    # weights through RAM in between so a model's own cache is genuinely
    # displaced before it's cold-loaded again. ollama_unload() after each
    # model's turn additionally evicts it from Ollama's own memory so it
    # can't stay resident under keep_alive across the whole session.
    RAW_CALL_LOG.clear()
    sessions_by_model = {model: [] for model in args.models}
    gpu_seen = False
    for i in range(1, args.sessions + 1):
        print(f"\n{'#'*60}\nSESSION {i}/{args.sessions}\n{'#'*60}")
        for model in args.models:
            print(f"\n{'='*60}\nBenchmarking: {model}  ({args.runs} trial(s)/test)\n{'='*60}")
            s = run_benchmark_session(model, args.runs, personas, intent_cfg, context_text, args.num_ctx, i, args.sessions)
            sessions_by_model[model].append(s)
            if s.get("gpu_accelerated"):
                gpu_seen = True
            ollama_unload(model)

    all_results = []
    for model in args.models:
        agg = aggregate_sessions(sessions_by_model[model])
        all_results.append({"model": model, "sessions_raw": sessions_by_model[model], "aggregated": agg})

    out_root = Path(args.outdir) if args.outdir else (SCRIPT_DIR.parent / "results")
    out_dir = classify_result_path(hw, gpu_seen, nickname, out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path = out_dir / "raw_call_log.json"
    raw_log_path.write_text(json.dumps(RAW_CALL_LOG, indent=2))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "submitted_by": args.submitted_by,
        "submission_date": datetime.now().date().isoformat(),
        "hardware": hw,
        "test_config": {
            "models_tested": args.models,
            "sessions": args.sessions,
            "trials_per_test": args.runs,
            "num_ctx": args.num_ctx,
            "intent_config": intent_cfg["name"],
            "persona_config": str(persona_dir),
            "context_config": str(context_path),
        },
        "results": all_results,
    }

    results_path = out_dir / "aggregated_results.json"
    results_path.write_text(json.dumps(payload, indent=2))

    import csv
    persona_keys = sorted({k for r in all_results for k in r["aggregated"]["persona_adherence"].keys()})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cold_load_s_median", "ttft_s_median", "tpot_s_median", "tokens_per_sec_median",
                    "json_pass_rate_avg", "intent_accuracy_median", "context_recall_rate", "data_quality_flags"]
                   + [f"persona_{k}_pass_rate_median" for k in persona_keys])
        for r in all_results:
            agg = r["aggregated"]
            rates = [v["pass_rate_median"] for v in agg["format_compliance"].values() if v["pass_rate_median"] is not None]
            json_avg = round(sum(rates) / len(rates), 2) if rates else None
            row = [
                r["model"],
                (agg["metrics"]["cold_load_s"] or {}).get("median"),
                (agg["metrics"]["ttft_s"] or {}).get("median"),
                (agg["metrics"]["tpot_s_per_token"] or {}).get("median"),
                (agg["metrics"]["tokens_per_sec"] or {}).get("median"),
                json_avg,
                agg["intent_detection"].get("accuracy_median"),
                agg["context_recall"].get("recalled_rate"),
                ";".join(agg["data_quality_flags"]),
            ]
            for k in persona_keys:
                row.append(agg["persona_adherence"].get(k, {}).get("constraint_pass_rate_median"))
            w.writerow(row)

    print_summary_table(payload)
    report_path = generate_html_report(results_path)

    print(f"\nFull results: {results_path}")
    print(f"CSV summary:  {out_dir / 'summary.csv'}")
    print(f"HTML report:  {report_path}")
    print(f"Raw call log: {raw_log_path}  ({len(RAW_CALL_LOG)} calls — every trial, untruncated, "
          f"for tracing outliers back to a specific session/test/call)")


if __name__ == "__main__":
    main()
