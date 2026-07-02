#!/usr/bin/env python3
"""
edge-ai-bench — Local LLM Benchmark Suite

Targets Ollama's local REST API (http://localhost:11434). Designed for
comparing small/quantized local models on low-end and mid-range hardware,
including single-board edge devices (Raspberry Pi, Jetson, etc).

WHAT IT MEASURES (metric names follow inference-serving literature: NVIDIA's
LLM benchmarking series, AWS Neuron docs, DigitalOcean's LLM inference guide):

  1. Cold-load time    — time from "model not in memory" to first response.
  2. TTFT               — Time To First Token.
  3. TPOT / ITL         — Time Per Output Token (inter-token latency).
  4. Tokens/sec (decode) — throughput once generation has started.
  5. End-to-end latency — full wall-clock time for one turn.
  6. Persona adherence  — constraint checking (forbidden words, address, sentences).
  7. Output-format compliance — strict JSON, hybrid JSON+chat, conversational.
  8. Intent detection   — few-shot classification accuracy.
  9. Context recall     — memory/RAG capability testing.
  10. Adversarial robustness — handling edge cases, jailbreak attempts, nonsense.
  11. Domain Q&A        — factual accuracy on coding, general knowledge tasks.
  12. Semantic quality  — Gemini Flash evaluation (optional, --judge-with-gemini).
  13. Resource footprint — VRAM/RAM usage.

RELIABILITY: This script runs the full suite across --sessions independent
sessions (default 3, minimum recommended for any community submission) and
reports median/mean/stdev per metric, flagging high-variance or single-session
results instead of silently trusting them.

OUTPUT: aggregated_results.json + summary.csv + report.html + raw_call_log.json.

USAGE:
  python llm_benchmark.py --models llama3.2:1b llama3.2:3b qwen2.5:1.5b
  python llm_benchmark.py --models llama3.2:3b --sessions 5 --judge-with-gemini
  python llm_benchmark.py --report results/.../aggregated_results.json

Requires: pip install -r requirements.txt (requests, pyyaml, google-generativeai)
Ollama must be running: `ollama serve`.
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

# Try to import Gemini judge (optional)
try:
    from gemini_judge import (
        judge_all_responses,
        add_judge_scores_to_payload,
        validate_api_key,
    )
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

OLLAMA_URL = "http://localhost:11434"
GEN_ENDPOINT = f"{OLLAMA_URL}/api/generate"
PS_ENDPOINT = f"{OLLAMA_URL}/api/ps"

SCHEMA_VERSION = "1.1"
BENCHMARK_VERSION = "1.1"

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
                if piece:  # ignore empty leading chunks
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
# Raw per-call log
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
# Hardware detection
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
    """Returns (gpu_model_or_None, gpu_vram_gb_or_None)."""
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
    """Returns (device_class, device_type)."""
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
            return "ac"
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
        "device_class": device_class,
        "device_type": device_type,
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram_gb,
        "ollama_version": _ollama_version(),
        "power_source": power_source,
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
# Test 1: Cold-load time
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
# Test 2: Warm latency (TTFT, TPOT, tokens/sec)
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
# Test 3: Output-format compliance
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
# Test 4: Adversarial Robustness (edge case handling)
# --------------------------------------------------------------------------

ADVERSARIAL_TESTS = [
    {
        "name": "nonsense_input",
        "prompt": "xyzzy qwerty glibbertigibbet foobar baz",
        "description": "Random nonsense words",
    },
    {
        "name": "contradictory_prompt",
        "prompt": "The sky is both red and not red. Explain.",
        "description": "Logical contradiction",
    },
    {
        "name": "token_limit_stress",
        "prompt": "Repeat the following 100 times: hello ",
        "description": "Stress test for token generation",
    },
    {
        "name": "empty_like_input",
        "prompt": "   ",
        "description": "Nearly empty input",
    },
]


def test_adversarial_robustness(model: str, num_ctx: int, session_idx: int) -> dict:
    """Test how model handles edge cases without crashing or hanging."""
    results = {}
    for test in ADVERSARIAL_TESTS:
        res = logged_generate(
            model,
            test["prompt"],
            session_idx,
            "adversarial_robustness",
            test["name"],
            1,
            keep_alive="10m",
            options={"num_predict": 50, "temperature": 0.5, "num_ctx": num_ctx},
            timeout=30,
        )
        results[test["name"]] = {
            "description": test["description"],
            "ok": res.ok,
            "handled_gracefully": res.ok and len(res.text.strip()) > 0,
            "response_length": len(res.text) if res.ok else 0,
            "error": res.error if not res.ok else None,
        }
    return results


# --------------------------------------------------------------------------
# Test 5: Domain-specific Q&A
# --------------------------------------------------------------------------

DOMAIN_QA_TESTS = [
    {
        "category": "coding",
        "questions": [
            "What does the 'async' keyword do in Python?",
            "Explain what a REST API is in simple terms.",
        ],
    },
    {
        "category": "general_knowledge",
        "questions": [
            "What is the capital of France?",
            "How many continents are there?",
        ],
    },
    {
        "category": "reasoning",
        "questions": [
            "If a train travels 100 miles in 2 hours, what is its average speed?",
            "What comes next in this sequence: 2, 4, 8, 16, ?",
        ],
    },
]


def test_domain_qa(model: str, num_ctx: int, session_idx: int) -> dict:
    """Test factual accuracy and reasoning across different domains."""
    results = {}
    for category_test in DOMAIN_QA_TESTS:
        cat = category_test["category"]
        cat_results = []
        for i, q in enumerate(category_test["questions"], 1):
            res = logged_generate(
                model,
                q,
                session_idx,
                "domain_qa",
                f"{cat}_{i}",
                1,
                keep_alive="10m",
                options={"num_predict": 100, "temperature": 0.1, "num_ctx": num_ctx},
            )
            cat_results.append({
                "question": q,
                "ok": res.ok,
                "response": res.text[:200] if res.ok else res.error,
                "response_length": len(res.text) if res.ok else 0,
            })
        results[cat] = {
            "n_ok": sum(1 for r in cat_results if r["ok"]),
            "samples": cat_results,
        }
    return results


# --------------------------------------------------------------------------
# Test 6: Persona adherence (existing code, unchanged)
# --------------------------------------------------------------------------

PERSONA_PROBES = [
    "Tell me about yourself.",
    "I dropped my coffee everywhere, what should I do?",
    "Can you do my homework for me?",
    "Turn off the lights and tell me a joke.",
]


def load_personas_from_dir(persona_dir: Path) -> dict:
    """Loads every .yaml/.yml in the folder."""
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
    """Cheap, deterministic checks — catches hard constraint violations."""
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
            "samples": samples[:3],
        }
    return results


# --------------------------------------------------------------------------
# Test 7: Intent detection (existing code, unchanged)
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
# Test 8: Context recall (existing code, unchanged)
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
# Full benchmark session
# --------------------------------------------------------------------------

def run_benchmark_session(model: str, runs: int, personas: dict, intent_cfg: dict,
                           context_text: str, num_ctx: int, session_idx: int, total_sessions: int) -> dict:
    print(f"  session {session_idx}/{total_sessions}")
    swap_before = _mem_info_gb().get("swap_used_gb")

    print("    [1/8] cold load...")
    cold = test_cold_load(model, num_ctx, session_idx)

    print("    [2/8] warm latency (TTFT/TPOT/tokens-per-sec)...")
    latency = test_warm_latency(model, DEFAULT_PROMPTS, runs, num_ctx, session_idx)

    print("    [3/8] output-format compliance...")
    fmt = test_format_compliance(model, runs, num_ctx, session_idx)

    print("    [4/8] adversarial robustness...")
    adversarial = test_adversarial_robustness(model, num_ctx, session_idx)

    print("    [5/8] domain-specific Q&A...")
    domain_qa = test_domain_qa(model, num_ctx, session_idx)

    print(f"    [6/8] persona adherence across {len(personas)} persona(s)...")
    persona_results = test_persona_adherence_multi(model, personas, max(1, runs // 2), num_ctx, session_idx)

    print("    [7/8] intent/command detection...")
    intent = test_intent_detection(model, intent_cfg, num_ctx, session_idx)

    print("    [8/8] context/memory recall...")
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
        "adversarial_robustness": adversarial,
        "domain_qa": domain_qa,
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

    # Adversarial robustness aggregation
    adv_metrics = {}
    for s in sessions:
        for test_name, result in s["adversarial_robustness"].items():
            if test_name not in adv_metrics:
                adv_metrics[test_name] = []
            adv_metrics[test_name].append(result["handled_gracefully"])
    adversarial_robustness = {
        k: {"graceful_handling_rate": round(sum(v) / len(v), 2) if v else None}
        for k, v in adv_metrics.items()
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
        "adversarial_robustness": adversarial_robustness,
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
# Summary table
# --------------------------------------------------------------------------

def print_summary_table(payload: dict):
    results = payload["results"]
    print(f"\n{'='*130}\nSUMMARY (median across {payload['test_config']['sessions']} session(s))\n{'='*130}")
    header = (f"{'Model':<18}{'Cold(s)':<10}{'TTFT(s)':<10}{'TPOT(ms)':<10}{'Tok/s':<8}{'JSON%':<8}{'Adv%':<8}{'Intent%':<10}{'Recall%':<10}")
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
        adv_rates = [v["graceful_handling_rate"] for v in agg["adversarial_robustness"].values() if v["graceful_handling_rate"] is not None]
        adv_avg = round(sum(adv_rates) / len(adv_rates) * 100) if adv_rates else None
        intent_acc = agg["intent_detection"].get("accuracy_median")
        recall = agg["context_recall"].get("recalled_rate")
        print(f"{r['model']:<18}{str(cold):<10}{str(ttft):<10}"
              f"{(str(round(tpot*1000,1)) if tpot else 'N/A'):<10}"
              f"{str(round(tps,1) if tps else 'N/A'):<8}"
              f"{str(json_avg)+'%' if json_avg else 'N/A':<8}{str(adv_avg)+'%' if adv_avg else 'N/A':<8}"
              f"{str(round(intent_acc*100) if intent_acc else 'N/A'):<10}"
              f"{str(round(recall*100) if recall else 'N/A'):<10}")


# --------------------------------------------------------------------------
# HTML report generator
# --------------------------------------------------------------------------

def generate_html_report(json_path: Path, out_path: Path = None) -> Path:
    json_path = Path(json_path)
    payload = json.loads(json_path.read_text())
    hw = payload.get("hardware", {})
    results = payload.get("results", [])
    out_path = Path(out_path) if out_path else json_path.with_name("report.html")

    device_label = hw.get("device_type") or hw.get("hostname", "unknown device")
    hw_card = f'''<div class="hw-card"><h2>Hardware</h2><table>
      <tr><td>Device</td><td>{device_label} ({hw.get('device_class','?')})</td></tr>
      <tr><td>CPU</td><td>{hw.get('cpu_model','?')} ({hw.get('cpu_cores_physical','?')}C / {hw.get('cpu_threads_logical','?')}T)</td></tr>
      <tr><td>RAM</td><td>{hw.get('ram_total_gb','?')} GB</td></tr>
      <tr><td>GPU</td><td>{hw.get('gpu_model') or 'none'}</td></tr>
      <tr><td>Ollama</td><td>{hw.get('ollama_version','?')}</td></tr>
    </table></div>'''

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>edge-ai-bench report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2rem auto;color:#1a1a1a}}h1{{font-size:1.8rem}}h2{{border-bottom:2px solid #eee;margin-top:2rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}.hw-card table{{border:1px solid #ddd}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f0f2f5}}</style>
</head><body>
<h1>edge-ai-bench report</h1>
<p>Generated {datetime.now().isoformat(timespec='seconds')} • Schema v{payload.get('schema_version','?')} • Benchmark v{payload.get('benchmark_version','?')}</p>
{hw_card}
<h2>Results Summary</h2>
<table><tr><th>Model</th><th>Cold Load (s)</th><th>TTFT (s)</th><th>Tok/s</th><th>JSON Compliance</th><th>Intent Accuracy</th></tr>
'''

    for r in results:
        agg = r["aggregated"]
        cold = (agg["metrics"]["cold_load_s"] or {}).get("median", "N/A")
        ttft = (agg["metrics"]["ttft_s"] or {}).get("median", "N/A")
        tps = (agg["metrics"]["tokens_per_sec"] or {}).get("median", "N/A")
        json_rates = [v["pass_rate_median"] for v in agg["format_compliance"].values() if v["pass_rate_median"] is not None]
        json_avg = f"{round(sum(json_rates) / len(json_rates) * 100)}%" if json_rates else "N/A"
        intent = f"{round(agg['intent_detection'].get('accuracy_median', 0)*100)}%" if agg['intent_detection'].get('accuracy_median') else "N/A"
        html += f"<tr><td>{r['model']}</td><td>{cold}</td><td>{ttft}</td><td>{tps}</td><td>{json_avg}</td><td>{intent}</td></tr>"

    html += """</table>
<div style="margin-top:2rem;color:#888;font-size:0.9rem">
<p>Lower is better: Cold Load, TTFT. Higher is better: Tok/s, JSON Compliance, Intent Accuracy.</p>
</div>
</body></html>"""

    out_path.write_text(html)
    return out_path


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="edge-ai-bench: Local LLM benchmarking for edge hardware.")
    parser.add_argument("--models", nargs="+", help="e.g. --models llama3.2:1b llama3.2:3b")
    parser.add_argument("--runs", type=int, default=3, help="trials per test (default 3)")
    parser.add_argument("--sessions", type=int, default=3, help="independent benchmark sessions (default 3)")
    parser.add_argument("--persona-dir", type=str, default=None, help="folder of persona YAML files")
    parser.add_argument("--intent-config", type=str, default=None, help="path to intent-detection config")
    parser.add_argument("--context", type=str, default=None, help="path to context/memory text file")
    parser.add_argument("--num-ctx", type=int, default=2048, help="context window (default 2048)")
    parser.add_argument("--outdir", type=str, default=None, help="results directory")
    parser.add_argument("--device-nickname", type=str, default=None, help="device name for results folder")
    parser.add_argument("--submitted-by", type=str, default="anonymous", help="your name/handle")
    parser.add_argument("--judge-with-gemini", action="store_true", help="Enable Gemini Flash semantic judgment (requires GOOGLE_API_KEY)")
    parser.add_argument("--report", type=str, default=None, help="path to aggregated_results.json to generate report")
    parser.add_argument("--report-out", type=str, default=None, help="output path for --report")
    args = parser.parse_args()

    if args.report:
        out = generate_html_report(Path(args.report), Path(args.report_out) if args.report_out else None)
        print(f"Report written to {out}")
        return

    if not args.models:
        parser.error("--models is required unless --report is given")

    if args.judge_with_gemini and not GEMINI_AVAILABLE:
        print("⚠ --judge-with-gemini requested but gemini_judge module not found.")
        print("  Install: pip install google-generativeai")
        args.judge_with_gemini = False

    check_ollama_alive()

    persona_dir = Path(args.persona_dir).expanduser() if args.persona_dir else SCRIPT_DIR / "configs" / "personas" / "default_pack"
    intent_path = Path(args.intent_config).expanduser() if args.intent_config else SCRIPT_DIR / "configs" / "intents" / "default_intents.yaml"
    context_path = Path(args.context).expanduser() if args.context else SCRIPT_DIR / "configs" / "context" / "default_context.txt"

    personas = load_personas_from_dir(persona_dir)
    if not personas:
        sys.exit(f"No persona files found in {persona_dir}")
    intent_cfg = load_intent_config(intent_path)
    context_text = context_path.read_text() if context_path.exists() else "Generic context."

    if args.sessions < 3:
        print(f"WARNING: --sessions {args.sessions} < 3 (community minimum). Results will be flagged.")

    hw = get_hardware_info()
    nickname = args.device_nickname or hw["hostname"]

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
        },
        "results": all_results,
    }

    # Optional: Judge with Gemini
    if args.judge_with_gemini:
        print("\nSending responses to Gemini Flash for semantic evaluation...")
        try:
            judge_results = judge_all_responses(RAW_CALL_LOG, verbose=True)
            payload = add_judge_scores_to_payload(payload, judge_results)
            print("✓ Semantic judgments added to results.")
        except Exception as e:
            print(f"⚠ Gemini judging failed: {e}")

    results_path = out_dir / "aggregated_results.json"
    results_path.write_text(json.dumps(payload, indent=2))

    print_summary_table(payload)
    report_path = generate_html_report(results_path)

    print(f"\n✓ Full results: {results_path}")
    print(f"✓ HTML report:  {report_path}")
    print(f"✓ Raw call log: {raw_log_path}")


if __name__ == "__main__":
    main()
