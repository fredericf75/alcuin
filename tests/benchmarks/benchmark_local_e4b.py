#!/usr/bin/env python3
"""
Empirical Benchmark: Actual Local gemma4:e4b on Ollama
Target: Cary Town Ordinances (3,046 Chunks)
Prompt: Multi-Departmental Enforcement Scenario
Evaluates the architectural impact of governed retrieval on the actual local e4b model.
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

def call_local_e4b(system_prompt: str, user_prompt: str, num_ctx: int = 16384) -> Tuple[str, float]:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "gemma4:e4b",
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
        # Fallback if content was empty
        content = msg.get("thinking", "").strip()
    return content, dt

def main():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK: ACTUAL LOCAL gemma4:e4b ON OLLAMA")
    print("Target: Cary Town Ordinances (Governed Traversal @ 8k Context)")
    print("=" * 80)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"docs_internal/test_logs/{timestamp}_benchmark_local_e4b.log")
    resp_path = Path(f"docs_internal/test_logs/{timestamp}_local_e4b_response.txt")
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"# Benchmark: Local gemma4:e4b Architectural Evaluation")
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

    # 3. Call actual local gemma4:e4b
    log("Calling actual local 'gemma4:e4b' on Ollama (num_ctx=16384)...")
    response_text, gen_time = call_local_e4b(system_prompt, user_prompt, num_ctx=16384)
    log(f"Response generated in {gen_time:.2f}s ({len(response_text)} chars)")

    with open(resp_path, "w") as f:
        f.write(response_text)
    log(f"Saved response to {resp_path}")

    # 4. Score against detailed checklist
    score_res = score_detailed(response_text)
    log("\n" + "=" * 80)
    log(f"DETAILED CHECKLIST SCORE: {score_res['score']}/{score_res['total']} ({score_res['percentage']}%)")
    log("=" * 80)
    for k, v in score_res["details"].items():
        status = "PASS" if v else "FAIL"
        log(f"  - {k}: {status}")

    log("\n--- EXCERPT OF GENERATED ANSWER ---")
    lines = response_text.splitlines()
    log("\n".join(lines[:45]))
    log("...\n")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Benchmark log saved to {log_path}")

if __name__ == "__main__":
    main()
