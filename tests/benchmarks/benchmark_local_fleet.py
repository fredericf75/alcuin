#!/usr/bin/env python3
"""
Empirical Benchmark: Local Multi-Model Fleet Comparison
Target: Cary Town Ordinances (Governed Traversal Evidence @ 8k Context)
Prompt: Multi-Departmental Enforcement Scenario
Models Tested:
  1. llama3.2:3b
  2. qwen2.5:7b-instruct
"""

import os
import re
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tests.benchmarks.benchmark_pruning_value import (
    run_governed_retrieval,
    score_detailed,
    DETAILED_CHECKLIST,
    SCENARIO_PROMPT,
)

def call_ollama(model: str, system_prompt: str, user_prompt: str, num_ctx: int = 16384) -> Tuple[str, float]:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": 0.1,
            "top_p": 0.95,
        },
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=600)
    dt = time.time() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    data = resp.json()
    msg = data.get("message", {})
    content = msg.get("content", "").strip()
    if not content and "thinking" in msg:
        content = msg.get("thinking", "").strip()
    return content, dt

def main():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK: LOCAL MULTI-MODEL FLEET (LLAMA 3.2 3B & QWEN 2.5 7B)")
    print("Target: Cary Town Ordinances (Governed Traversal Evidence)")
    print("=" * 80)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"docs_internal/test_logs/{timestamp}_benchmark_local_fleet.log")
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"# Benchmark: Local Fleet Comparison (llama3.2:3b vs qwen2.5:7b-instruct)")
    log(f"Timestamp: {timestamp}\n")

    # 1. Retrieval under 8k Context Governor (budget = 5,000 tokens)
    log("Executing Context-Governed Traversal (Budget = 5,000 tokens)...")
    t_ret0 = time.time()
    admitted, pruned, alert = run_governed_retrieval(5000)
    t_ret = time.time() - t_ret0

    evidence_text = "\n\n".join([f"[{c['id']}] {c['section_title']}\n{c['content']}" for c in admitted])
    admitted_tokens = int(len(evidence_text) / 4.0)
    log(f"Retrieval complete in {t_ret:.2f}s: Admitted {len(admitted)} chunks (~{admitted_tokens:,} tokens), Pruned {len(pruned)} chunks")
    if alert:
        log(f"\nTelemetry Alert Emitted:\n{alert}\n")

    # 2. Assemble Prompt
    system_prompt = (
        "You are an expert municipal law research assistant. Analyze the provided statutory evidence carefully "
        "and answer the user's multi-departmental enforcement scenario thoroughly. "
        "Cite exact section numbers, exact dollar penalties, and exact procedural days/boards from the evidence. "
        "If a specific detail is absent from the evidence, explicitly state that it was not provided."
    )
    user_prompt = f"### STATUTORY EVIDENCE:\n\n{evidence_text}\n\n### SCENARIO TASK:\n{SCENARIO_PROMPT}"

    models_to_test = [
        "llama3.2:3b",
        "qwen2.5:7b-instruct",
    ]

    fleet_results = {}

    for model in models_to_test:
        log("-" * 80)
        log(f"RUNNING MODEL: {model}")
        log("-" * 80)

        response_text, gen_time = call_ollama(model, system_prompt, user_prompt, num_ctx=16384)
        log(f"Response generated in {gen_time:.2f}s ({len(response_text)} chars)")

        resp_file = Path(f"docs_internal/test_logs/{timestamp}_{model.replace(':', '_').replace('.', '_')}_response.txt")
        with open(resp_file, "w") as f:
            f.write(response_text)
        log(f"Saved response to {resp_file}")

        score_res = score_detailed(response_text)
        log(f"Detailed Checklist Score: {score_res['score']}/{score_res['total']} ({score_res['percentage']}%)")
        for k, v in score_res["details"].items():
            status = "PASS" if v else "FAIL"
            log(f"  - {k}: {status}")

        log("\n--- EXCERPT OF GENERATED ANSWER ---")
        lines = response_text.splitlines()
        log("\n".join(lines[:35]))
        log("...\n")

        fleet_results[model] = {
            "gen_time": gen_time,
            "chars": len(response_text),
            "score": score_res,
            "response": response_text,
        }

    log("=" * 80)
    log("FLEET SUMMARY COMPARISON")
    log("=" * 80)
    for model, data in fleet_results.items():
        s = data["score"]
        log(f"{model:<22} | Score: {s['score']}/{s['total']} ({s['percentage']}%) | Latency: {data['gen_time']:.1f}s | Length: {data['chars']} chars")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nBenchmark log saved to {log_path}")

if __name__ == "__main__":
    main()
