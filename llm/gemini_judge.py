#!/usr/bin/env python3
"""
Gemini Flash semantic quality judge for edge-ai-bench.

After all local benchmarking completes, sends sampled responses to Gemini 2.0 Flash
for semantic scoring (relevance, coherence, usefulness, tone). This bridges the gap
between rule-based constraint checking and real semantic quality.

USAGE:
  python llm_benchmark.py --models llama3.2:1b --judge-with-gemini

Requires: GOOGLE_API_KEY environment variable + pip install google-generativeai

Cost estimate: ~$0.01-0.02 per full benchmark run (Gemini Flash is ~0.075/1M tokens).
"""

import json
import os
import sys
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None


DEFAULT_JUDGE_SYSTEM_PROMPT = """You are a semantic quality evaluator for local LLM responses in real-world use cases.
Score each response 1-5 on:

1. **Relevance** (1-5): Does it answer the prompt correctly and directly?
2. **Coherence** (1-5): Is it clear, grammatically correct, and well-structured?
3. **Usefulness** (1-5): Would a user find this helpful for their task?
4. **Tone consistency** (1-5): Does it match the expected persona/tone (if provided)?

IMPORTANT: You're judging edge-device local models, not GPT-4 or Claude.
Be fair: a 1B quantized model running on a Raspberry Pi should be judged on edge expectations, not desktop standards.
A coherent, relevant 3-sentence response from a 1B model on a Pi is good—don't penalize brevity.

For each response, return ONLY valid JSON (no prose):
{
  "scores": {
    "relevance": <1-5>,
    "coherence": <1-5>,
    "usefulness": <1-5>,
    "tone_consistency": <1-5>
  },
  "overall_score": <1-5 average>,
  "summary": "<1-2 sentence assessment>"
}
"""


def validate_api_key() -> str:
    """Check for GOOGLE_API_KEY environment variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get your key from https://aistudio.google.com/app/apikeys"
        )
    return api_key


def build_judge_prompt(test_name: str, samples: list, system_prompt: str = None) -> str:
    """Build a batch judgment prompt with multiple response samples."""
    prompt = f"""Evaluate these responses from a local LLM benchmark test: {test_name}

"""
    if system_prompt:
        prompt += f"System prompt used: {system_prompt}\n\n"
    
    prompt += "Responses to evaluate:\n\n"
    for i, sample in enumerate(samples, 1):
        prompt += f"[Sample {i}]\n"
        prompt += f"Prompt: {sample.get('prompt', 'N/A')[:200]}\n"
        prompt += f"Response: {sample.get('response_text', 'ERROR')[:400]}\n"
        prompt += f"---\n\n"
    
    prompt += "Score each response as JSON. Return a JSON array of scores (one per sample)."
    return prompt


def judge_batch_with_gemini(
    test_name: str,
    samples: list,
    system_prompt: str = None,
    judge_system: str = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.3,
) -> dict:
    """
    Send a batch of responses to Gemini Flash for semantic scoring.
    
    Args:
        test_name: Name of the test (e.g. "warm_latency", "persona_adherence")
        samples: List of dicts with keys: prompt, response_text
        system_prompt: Optional system prompt used for the test
        judge_system: Optional custom judge system prompt (default: DEFAULT_JUDGE_SYSTEM_PROMPT)
        model: Gemini model ID (default: gemini-2.0-flash, change to gemini-1.5-flash if needed)
        temperature: Temperature for judge reasoning (default 0.3 for consistency)
    
    Returns:
        dict with keys:
        - test_name: Input test name
        - n_samples: Number of samples judged
        - scores: List of score dicts (one per sample)
        - error: str or None
        - tokens_used: Approximate tokens consumed
    """
    if genai is None:
        return {
            "test_name": test_name,
            "error": "google.generativeai not installed. Run: pip install google-generativeai",
            "scores": [],
        }
    
    if not samples:
        return {
            "test_name": test_name,
            "n_samples": 0,
            "scores": [],
            "error": "No samples provided",
        }
    
    judge_system = judge_system or DEFAULT_JUDGE_SYSTEM_PROMPT
    judge_prompt = build_judge_prompt(test_name, samples, system_prompt)
    
    try:
        client = genai.GenerativeModel(
            model,
            system_instruction=judge_system,
        )
        response = client.generate_content(
            judge_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        
        # Parse response — try to extract JSON
        try:
            # Attempt to parse entire response as JSON array
            scores = json.loads(response.text)
            if not isinstance(scores, list):
                scores = [scores]  # Wrap single object
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response text
            import re
            json_matches = re.findall(r'\[.*?\]|\{.*?\}', response.text, re.DOTALL)
            if json_matches:
                try:
                    scores = json.loads(json_matches[0])
                    if not isinstance(scores, list):
                        scores = [scores]
                except json.JSONDecodeError:
                    scores = []
            else:
                scores = []
        
        # Pad or trim scores to match sample count
        while len(scores) < len(samples):
            scores.append({"overall_score": None, "error": "parsing failed"})
        scores = scores[:len(samples)]
        
        return {
            "test_name": test_name,
            "n_samples": len(samples),
            "scores": scores,
            "error": None,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else None,
            },
        }
    
    except Exception as e:
        return {
            "test_name": test_name,
            "n_samples": len(samples),
            "scores": [],
            "error": str(e),
        }


def judge_all_responses(
    raw_call_log: list,
    api_key: str = None,
    sample_per_test: int = 10,
    judge_system: str = None,
    verbose: bool = True,
) -> dict:
    """
    Judge all responses in the raw call log, grouped by test type.
    
    Args:
        raw_call_log: List of call dicts from RAW_CALL_LOG
        api_key: Optional API key (default: read from GOOGLE_API_KEY env var)
        sample_per_test: Max samples to judge per test (default 10, saves tokens)
        judge_system: Optional custom judge system prompt
        verbose: Print progress (default True)
    
    Returns:
        dict mapping test_name -> judge result dict
    """
    if not raw_call_log:
        if verbose:
            print("⚠ No call log data to judge.")
        return {}
    
    api_key = api_key or validate_api_key()
    genai.configure(api_key=api_key)
    
    # Group responses by test type
    by_test = {}
    for call in raw_call_log:
        test_name = call.get("test", "unknown")
        if test_name not in by_test:
            by_test[test_name] = []
        by_test[test_name].append(call)
    
    judge_results = {}
    total_tokens = 0
    
    for test_name, calls in sorted(by_test.items()):
        # Sample responses from this test (don't judge all, save tokens)
        step = max(1, len(calls) // sample_per_test)
        sampled = calls[::step][:sample_per_test]
        
        # Filter successful calls only
        sampled = [c for c in sampled if c.get("ok")]
        
        if not sampled:
            if verbose:
                print(f"  {test_name}: no successful responses to judge")
            judge_results[test_name] = {
                "n_samples": 0,
                "error": "no successful responses",
            }
            continue
        
        if verbose:
            print(f"  {test_name}: judging {len(sampled)} samples...")
        
        result = judge_batch_with_gemini(
            test_name,
            sampled,
            system_prompt=sampled[0].get("system_prompt"),
            judge_system=judge_system,
        )
        
        judge_results[test_name] = result
        
        if result.get("usage"):
            total_tokens += (
                result["usage"].get("prompt_tokens", 0) +
                result["usage"].get("output_tokens", 0)
            )
        
        if result.get("error") and verbose:
            print(f"    ⚠ Error: {result['error']}")
    
    if verbose:
        print(f"\nJudgment complete. Approximate tokens used: {total_tokens}")
    
    return judge_results


def add_judge_scores_to_payload(payload: dict, judge_results: dict) -> dict:
    """
    Merge judge results into the aggregated_results.json payload.
    
    Adds a "semantic_quality" section to each result's aggregated data.
    """
    for result in payload.get("results", []):
        result["aggregated"]["semantic_quality"] = {
            "judge_results": judge_results,
            "note": "Sampled responses scored by Gemini 2.0 Flash on relevance, coherence, usefulness, tone.",
        }
    
    return payload


if __name__ == "__main__":
    # Test mode: if run directly
    api_key = validate_api_key()
    genai.configure(api_key=api_key)
    
    test_samples = [
        {
            "prompt": "What is the weather like today?",
            "response_text": "I don't have access to real-time weather data, but you can check a weather app or website for current conditions."
        }
    ]
    
    print("Testing Gemini judge with sample response...")
    result = judge_batch_with_gemini("test", test_samples)
    print(json.dumps(result, indent=2))
