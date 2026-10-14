#!/usr/bin/env python3
"""
Empirical Benchmark: Value of Feedback on Pruning Across Sizing Tiers
Target: Cary Town Ordinances (3,046 Chunks)
Prompt: Multi-Departmental Enforcement Scenario (Animal Control & Housing Inspection)
Compares:
  Tier 1: e4b (gemma-4-26b-a4b-it) @ 8k Context Window (5,000 token budget)
  Tier 2: Intermediate Upsize @ 12k Context Window (9,000 token budget)
  Tier 3: 31b (gemma-4-31b-it) @ 32k Context Window (16,000 token budget)
"""

import os
import re
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from alcuin.storage.vault import Vault
from alcuin.storage.engine import VaultStorage
from tests.benchmarks.benchmark_cary_section_expansion import (
    CHECKLIST,
    SCENARIO_PROMPT,
    score_facts,
)

load_dotenv()

# Extended Checklist for detailed analysis
DETAILED_CHECKLIST = [
    {
        "id": "animal_entry_warrant",
        "name": "Animal Control Entry & Warrant (Sec 6-28 / GS 15-27.2)",
        "pattern": r"(?:6-28|15-27\.2|administrative (?:search )?warrant|exigent|immediate action)",
    },
    {
        "id": "building_inspector_warrant",
        "name": "Building Inspector Warrant Required / No Absolute Entry (Sec 8-128 / GS 15-27.2)",
        "pattern": r"(?:8-128|15-27\.2|no absolute right|warrant to inspect|reasonable access)",
    },
    {
        "id": "leash_and_rabies_violations",
        "name": "Leash & Rabies Violations (Sec 6-65 & Sec 6-73)",
        "pattern": r"(?:6-65|6-62|running at large|restraint|6-73|rabies tag)",
    },
    {
        "id": "animal_3rd_offense_penalty",
        "name": "Exact 3rd Offense Animal Penalty ($75.00 under Sec 6-132)",
        "pattern": r"(?:\$75|75(?:\.00)?(?:\s*dollars)?|Schedule A)",
    },
    {
        "id": "chicken_slaughter_prohibited",
        "name": "On-Site Chicken Slaughter Prohibited (Sec 6-62(c))",
        "pattern": r"(?:6-62\(c\)|on-site slaughter|slaughter (?:of )?chickens (?:is )?prohibited|residential property)",
    },
    {
        "id": "fitness_standards",
        "name": "Deteriorated vs Dilapidated 50% Rule (Sec 8-117)",
        "pattern": r"(?:deteriorated|dilapidated|50%|fifty percent|value of the dwelling)",
    },
    {
        "id": "procedural_notice_waiting_period",
        "name": "Affordable Housing Notice & 45-Day Waiting Period (Sec 8-129 / GS 160D)",
        "pattern": r"(?:affordable housing|45 days|forty-five days|160D-1203|hearing|10.*30 days)",
    },
    {
        "id": "animal_appeals_board_and_bond",
        "name": "Animal Appeals Forum, 10 Days & $200 Bond (Sec 6-139)",
        "pattern": r"(?:town manager|animal control appeals|10 days|ten \(10\) days|\$200|appeal bond)",
    },
    {
        "id": "housing_appeals_board_and_court",
        "name": "Housing Appeals: Board of Adjustment (10 Days) & Superior Court (Sec 8-130)",
        "pattern": r"(?:board of adjustment|zoning board of adjustment|10 days|ten \(10\) days|superior court|certiorari|15 days|30 days)",
    },
]

def score_detailed(text: str) -> Dict[str, Any]:
    hits = {}
    total = 0
    for item in DETAILED_CHECKLIST:
        m = bool(re.search(item["pattern"], text, re.IGNORECASE))
        hits[item["id"]] = m
        if m:
            total += 1
    return {
        "score": total,
        "total": len(DETAILED_CHECKLIST),
        "percentage": round((total / len(DETAILED_CHECKLIST)) * 100.0, 1),
        "details": hits,
    }


def call_hosted_gemma(model: str, system_prompt: str, user_prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        # Extract non-thought text parts
        non_thought = [p.get("text", "") for p in parts if not p.get("thought", False) and p.get("text")]
        if non_thought:
            return "\n".join(non_thought).strip()
        return parts[-1].get("text", "").strip()


def run_governed_retrieval(budget_tokens: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    vault_path = Path("docs_internal/examples/cary_ordinances")
    vault = Vault.load(vault_path)
    storage = VaultStorage(vault)
    tbl = storage.table
    arrow_data = tbl.to_arrow()

    all_chunks = []
    for i in range(len(arrow_data)):
        all_chunks.append({
            "id": arrow_data["id"][i].as_py(),
            "file_path": arrow_data["file_path"][i].as_py(),
            "breadcrumbs": arrow_data["breadcrumbs"][i].as_py(),
            "section_title": arrow_data["section_title"][i].as_py(),
            "content": arrow_data["content"][i].as_py(),
            "vector": arrow_data["vector"][i].as_py(),
        })
    id_to_idx = {c["id"]: idx for idx, c in enumerate(all_chunks)}

    subgoals = [
        "Right of entry and search warrant authority for animal control versus building inspector under Town Code",
        "Specific violations and civil penalties for unleashed dogs rabies tag and chicken slaughter or processing facilities",
        "Housing fitness classification standards deteriorated versus dilapidated and abatement or demolition procedures",
        "Consolidated appeals procedure deadlines boards and fee bond requirements for animal and housing orders",
    ]

    q_vecs = [np.array(storage.embedder.get_embedding(q)) for q in subgoals]

    # Primary search
    primary_hits = []
    seen_ids = set()
    for q_idx, q in enumerate(subgoals):
        results = storage.search(q, top_k=4)
        for r in results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                primary_hits.append(all_chunks[id_to_idx[r.id]])

    # Collect sibling candidates
    candidate_dict = {}
    for hit in primary_hits:
        m = re.match(r"^(.*?)#(\d+)$", hit["id"])
        if not m:
            continue
        base_path, idx_str = m.group(1), int(m.group(2))
        for step in [-2, -1, 1, 2]:
            sib_idx = idx_str + step
            sib_id = f"{base_path}#{sib_idx}"
            if sib_id in id_to_idx and sib_id not in seen_ids:
                sib_chunk = all_chunks[id_to_idx[sib_id]]
                if hit["file_path"] == sib_chunk["file_path"] and hit["breadcrumbs"] == sib_chunk["breadcrumbs"]:
                    if sib_id not in candidate_dict:
                        candidate_dict[sib_id] = sib_chunk

    # Score sibling candidates
    scored_candidates = []
    for sib_id, sib_chunk in candidate_dict.items():
        v = np.array(sib_chunk["vector"])
        v_norm = np.linalg.norm(v) or 1.0
        scores = []
        for q_vec in q_vecs:
            q_norm = np.linalg.norm(q_vec) or 1.0
            scores.append(float(np.dot(q_vec, v) / (q_norm * v_norm)))
        best_score = max(scores) if scores else 0.0
        if best_score >= 0.45:
            scored_candidates.append((best_score, sib_chunk))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Context Governor Packing
    CHAR_PER_TOKEN = 4.0
    admitted = list(primary_hits)
    current_tokens = sum(len(c["content"]) for c in admitted) / CHAR_PER_TOKEN
    pruned = []

    for score, sib_chunk in scored_candidates:
        chunk_tokens = len(sib_chunk["content"]) / CHAR_PER_TOKEN
        if current_tokens + chunk_tokens <= budget_tokens:
            admitted.append(sib_chunk)
            current_tokens += chunk_tokens
        else:
            pruned.append((score, sib_chunk))

    alert_msg = ""
    if pruned:
        pruned_tokens = int(sum(len(c["content"]) for _, c in pruned) / CHAR_PER_TOKEN)
        top_pruned = pruned[:4]
        alert_msg = (
            f"[CONTEXT_GOVERNOR_ALERT]\n"
            f"Context budget reached: {int(current_tokens):,} tokens (budget: {budget_tokens:,} tokens).\n"
            f"Pruned {len(pruned)} high-relevance clauses (~{pruned_tokens:,} tokens) to prevent context degradation:\n"
        )
        for s, c in top_pruned:
            snippet = c["content"].split("\n")[0].replace("> **Context**:", "").strip()
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            alert_msg += f"  - {c['id']} (Score: {s:.2f}): {c['section_title']}\n"
        alert_msg += "Actionable Guidance: Sizing up context window is empirically justified."

    return admitted, [p[1] for p in pruned], alert_msg


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    print("=" * 80)
    print("EMPIRICAL BENCHMARK: VALUE OF FEEDBACK ON PRUNING ACROSS SIZING TIERS")
    print("Target: Cary Town Ordinances")
    print("=" * 80)

    tiers = [
        {
            "name": "Tier 1: e4b (gemma-4-26b-a4b-it) @ 8k Context (5,000 token budget)",
            "model": "gemma-4-26b-a4b-it",
            "budget": 5000,
        },
        {
            "name": "Tier 2: Intermediate Upsize (gemma-4-26b-a4b-it) @ 12k Context (9,000 token budget)",
            "model": "gemma-4-26b-a4b-it",
            "budget": 9000,
        },
        {
            "name": "Tier 3: 31b (gemma-4-31b-it) @ 32k Context (16,000 token budget)",
            "model": "gemma-4-31b-it",
            "budget": 16000,
        },
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"docs_internal/test_logs/{timestamp}_benchmark_pruning_value.log")
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"# Benchmark: Value of Pruning Feedback Across Sizing Tiers")
    log(f"Timestamp: {timestamp}\n")

    system_prompt = (
        "You are an expert municipal law research assistant. Analyze the provided statutory evidence carefully "
        "and answer the user's multi-departmental enforcement scenario thoroughly. "
        "Cite exact section numbers, exact dollar penalties, and exact procedural days/boards from the evidence. "
        "If a specific detail is absent from the evidence, explicitly state that it was not provided."
    )

    results = []

    for t_idx, tier in enumerate(tiers, 1):
        log("-" * 80)
        log(f"RUNNING {tier['name']}")
        log("-" * 80)

        t0 = time.time()
        admitted, pruned, alert = run_governed_retrieval(tier["budget"])
        retrieval_time = time.time() - t0

        evidence_text = "\n\n".join([f"[{c['id']}] {c['section_title']}\n{c['content']}" for c in admitted])
        admitted_tokens = int(len(evidence_text) / 4.0)
        log(f"Retrieval: Admitted {len(admitted)} chunks ({admitted_tokens:,} tokens) in {retrieval_time:.2f}s")
        log(f"Pruned: {len(pruned)} chunks")
        if alert:
            log(f"Telemetry Alert:\n{alert}")

        # Assemble prompt with evidence
        user_prompt = f"### STATUTORY EVIDENCE:\n\n{evidence_text}\n\n### SCENARIO TASK:\n{SCENARIO_PROMPT}"

        log(f"Calling hosted model '{tier['model']}'...")
        t_llm0 = time.time()
        response_text = call_hosted_gemma(tier["model"], system_prompt, user_prompt, api_key)
        llm_time = time.time() - t_llm0
        log(f"Response generated in {llm_time:.2f}s ({len(response_text)} chars)")

        score_res = score_detailed(response_text)
        log(f"Detailed Checklist Score: {score_res['score']}/{score_res['total']} ({score_res['percentage']}%)")
        for k, v in score_res["details"].items():
            status = "PASS" if v else "FAIL"
            log(f"  - {k}: {status}")

        log("\n--- EXCERPT OF GENERATED ANSWER ---")
        lines = response_text.splitlines()
        log("\n".join(lines[:35]))
        log("...\n")

        results.append({
            "tier": tier,
            "admitted_tokens": admitted_tokens,
            "pruned_count": len(pruned),
            "alert": alert,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "score": score_res,
            "response": response_text,
        })

    log("=" * 80)
    log("COMPARATIVE ANALYSIS: WHAT WAS GAINED BY SIZING UP?")
    log("=" * 80)

    r1, r2, r3 = results[0], results[1], results[2]

    log(f"Tier 1 (e4b 8k budget: 5k tokens):  Score = {r1['score']['score']}/{r1['score']['total']} ({r1['score']['percentage']}%), Pruned = {r1['pruned_count']} clauses")
    log(f"Tier 2 (Upsize 12k budget: 9k tokens): Score = {r2['score']['score']}/{r2['score']['total']} ({r2['score']['percentage']}%), Pruned = {r2['pruned_count']} clauses")
    log(f"Tier 3 (31b 32k budget: 16k tokens):  Score = {r3['score']['score']}/{r3['score']['total']} ({r3['score']['percentage']}%), Pruned = {r3['pruned_count']} clauses")

    log("\nDelta: Tier 2 vs Tier 1 (Gains from intermediate upsize):")
    gains_t2 = [k for k, v in r2['score']['details'].items() if v and not r1['score']['details'][k]]
    if gains_t2:
        for g in gains_t2:
            log(f"  + RECOVERED: {g}")
    else:
        log("  No checklist delta between Tier 1 and Tier 2")

    log("\nDelta: Tier 3 vs Tier 2 (Gains from full 31b 32k upsize):")
    gains_t3 = [k for k, v in r3['score']['details'].items() if v and not r2['score']['details'][k]]
    if gains_t3:
        for g in gains_t3:
            log(f"  + RECOVERED: {g}")
    else:
        log("  No checklist delta between Tier 2 and Tier 3")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    # Also save the full responses for qualitative review
    for idx, r in enumerate(results, 1):
        resp_path = Path(f"docs_internal/test_logs/{timestamp}_tier{idx}_response.txt")
        with open(resp_path, "w") as f:
            f.write(r["response"])
        log(f"Saved full Tier {idx} response to {resp_path}")

    log(f"\nBenchmark log saved to {log_path}")

if __name__ == "__main__":
    main()
