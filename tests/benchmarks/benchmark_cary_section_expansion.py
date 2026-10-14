#!/usr/bin/env python3
"""
Empirical Benchmark: Atomic Clause Retrieval vs Parent/Section-Expanded Retrieval
Target: Cary Ordinances Enforcement Scenario (3,046 Chunks)
Evaluator Model: gemma-4-31b-it on Google AI Studio
"""

import os
import re
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from alcuin.storage.vault import Vault
from alcuin.storage.engine import VaultStorage
from alcuin.agent.graph import LLMClient, ResearchAgent

load_dotenv()

SCENARIO_PROMPT = """Analyze the following multi-departmental enforcement scenario under the Town Code:

An animal control officer and a building inspector conduct a joint inspection of a residential property within the town limits. They uncover the following conditions:
1. The property owner keeps four adult dogs. Two dogs are roaming without leashes; one has an expired rabies tag. The owner also maintains an unpermitted chicken coop with ten chickens and on-site processing facilities.
2. The primary dwelling has major structural issues: the roof framing shows 40% damage, two exterior walls have plumb-line deviations of 4 inches, and the entire crawl space has standing water due to improper site grading.
3. The owner refuses to allow the building inspector entry into the backyard shed, claiming the inspector has no right to enter private property without a judicial warrant.

Provide a detailed legal analysis addressing:
A. Right of Entry & Procedures: Compare the statutory search and entry authority of the Animal Control Unit versus the Department of Inspections and Permits. Did the building inspector have an absolute right of entry, and under what conditions must administrative search warrants be sought?
B. Specific Violations & Penalties: Detail every specific violation under Chapters 6 and 8, citing exact sections. If the owner has previously been cited twice for unleashed animals, what is the exact civil penalty for the current animal offenses? Can the owner slaughter chickens on-site?
C. Housing Fitness & Timelines: Is the dwelling legally classified as "Deteriorated" or "Dilapidated"? What procedural steps, notices to third-party organizations, and waiting periods must the inspector follow before the Town Council can order demolition or abatement?
D. Consolidated Appeals Matrix: Construct a Markdown table detailing where, to whom, within how many days, and under what fee/bond conditions the owner must file appeals for (1) an animal citation/impoundment, (2) a dangerous animal determination, and (3) an order of the housing inspector."""

CHECKLIST = [
    {
        "id": "animal_entry",
        "name": "Animal Control Entry Authority (Chapter 6)",
        "pattern": r"(?:Sec\.\s*6-132|6-28|seizure and impoundment|enter upon premises)",
    },
    {
        "id": "building_entry",
        "name": "Building Inspector Warrant/Entry Authority (Chapter 8)",
        "pattern": r"(?:Sec\.\s*8-118|8-28|administrative (?:search )?warrant|warrantless entry|Fourth Amendment|15-27\.2)",
    },
    {
        "id": "animal_fine",
        "name": "3rd Offense Animal Civil Penalty Amount",
        "pattern": r"(?:\$100|\$200|\$250|\$500|Schedule A|third (?:offense|violation))",
    },
    {
        "id": "chicken_slaughter",
        "name": "Chicken Slaughter / Processing Restrictions",
        "pattern": r"(?:slaughter|processing facility|poultry|coop|Sec\.\s*6-7\d|chicken)",
    },
    {
        "id": "fitness_classification",
        "name": "Deteriorated vs Dilapidated Standard (Chapter 8)",
        "pattern": r"(?:deteriorated|dilapidated|fifty percent|50%|plumb[- ]line|structural damage)",
    },
    {
        "id": "third_party_notice",
        "name": "Third-Party Notice / Commission Requirements",
        "pattern": r"(?:historic (?:preservation )?commission|redevelopment|notice to (?:owner|mortgagee|parties)|certified mail)",
    },
    {
        "id": "waiting_period",
        "name": "Demolition / Abatement Waiting Period",
        "pattern": r"(?:90 days|ninety days|ten days|10 days|30 days|waiting period|order of demolition)",
    },
    {
        "id": "appeals_rules",
        "name": "Appeals Matrix (Deadlines, Boards, Fees/Bonds)",
        "pattern": r"(?:board of adjustment|board of appeals|superior court|ten \(10\) days|10 days|appeal bond)",
    },
]


def score_facts(evidence_text: str) -> Dict[str, Any]:
    hits = {}
    total_hits = 0
    for item in CHECKLIST:
        matched = bool(re.search(item["pattern"], evidence_text, re.IGNORECASE))
        hits[item["id"]] = matched
        if matched:
            total_hits += 1
    return {
        "score": total_hits,
        "total": len(CHECKLIST),
        "percentage": round((total_hits / len(CHECKLIST)) * 100.0, 1),
        "details": hits,
    }


def get_generic_heading_key(file_path: str, breadcrumbs: str) -> str:
    """Purely generic Markdown AST key: (file_path, breadcrumbs). Zero domain regexes."""
    return f"{file_path}::{breadcrumbs}"


def run_benchmark():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in environment or .env.")
        sys.exit(1)

    print("=" * 80)
    print("CARY ORDINANCES: ATOMIC RETRIEVAL VS PARENT/SECTION EXPANSION BENCHMARK")
    print("Model: gemma-4-31b-it on Google AI Studio")
    print("Vault: docs_internal/examples/cary_ordinances (3,046 Chunks)")
    print("=" * 80)

    vault_path = Path("docs_internal/examples/cary_ordinances")
    vault = Vault.load(vault_path)
    storage = VaultStorage(vault)

    llm = LLMClient(
        endpoint="https://generativelanguage.googleapis.com",
        model="gemma-4-31b-it",
        api_key=api_key,
        temperature=0.2,
        timeout=120.0,
    )
    agent = ResearchAgent(vault=vault, llm=llm)

    # Turn 0 Query Planning
    print("\nDecomposing scenario inquiry into initial subgoals...")
    subgoals = agent.decompose_query(SCENARIO_PROMPT)
    print(f"Subgoals generated ({len(subgoals)}):")
    for idx, sg in enumerate(subgoals):
        print(f"  [{idx + 1}] {sg}")

    search_queries = subgoals[:4] if len(subgoals) >= 4 else subgoals

    # -----------------------------------------------------------------------
    # MODE A: Status Quo (Atomic Clause Chunks)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("RUNNING MODE A: Status Quo (Raw Atomic Chunks)")
    print("-" * 70)
    t0 = time.perf_counter()
    mode_a_chunks = []
    seen_ids = set()

    for q in search_queries:
        results = storage.search(q, top_k=4)
        for r in results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                mode_a_chunks.append({
                    "id": r.id,
                    "breadcrumbs": r.breadcrumbs,
                    "section_title": r.section_title,
                    "content": r.content,
                })

    time_mode_a_retrieval = (time.perf_counter() - t0) * 1000.0
    text_mode_a = "\n\n---\n\n".join(
        f"Source: {c['breadcrumbs']} > {c['section_title']}\n{c['content']}" for c in mode_a_chunks
    )
    facts_a = score_facts(text_mode_a)

    print(f"Mode A Retrieved: {len(mode_a_chunks)} chunks in {time_mode_a_retrieval:.1f}ms")
    print(f"Mode A Fact Coverage: {facts_a['score']}/{facts_a['total']} ({facts_a['percentage']}%)")
    print(f"Mode A Context Length: {len(text_mode_a)} characters (~{len(text_mode_a)//4} tokens)")

    # -----------------------------------------------------------------------
    # MODE B: Purely Generic Markdown Heading AST Expansion
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("RUNNING MODE B: Purely Generic Markdown Heading AST Expansion")
    print("-" * 70)
    t0_b = time.perf_counter()

    # Pre-index section sibling map from the LanceDB table
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
        })

    # Group all vault chunks strictly by generic Markdown AST (file_path + breadcrumbs)
    heading_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_chunks:
        k = get_generic_heading_key(c["file_path"], c["breadcrumbs"])
        heading_map.setdefault(k, []).append(c)

    # In Mode B: for each matched search result, expand to all chunks in that exact Markdown heading AST
    mode_b_sections: Dict[str, List[Dict[str, Any]]] = {}
    for q in search_queries:
        results = storage.search(q, top_k=4)
        for r in results:
            k = get_generic_heading_key(r.file_path, r.breadcrumbs)
            if k not in mode_b_sections:
                siblings = heading_map.get(k, [{
                    "id": r.id,
                    "file_path": r.file_path,
                    "breadcrumbs": r.breadcrumbs,
                    "section_title": r.section_title,
                    "content": r.content,
                }])
                mode_b_sections[k] = siblings

    time_mode_b_retrieval = (time.perf_counter() - t0_b) * 1000.0

    # Format Mode B text: clean unified sections with full breadcrumbs
    formatted_sections = []
    total_expanded_chunks = 0
    for sec_key, chunks_in_sec in mode_b_sections.items():
        total_expanded_chunks += len(chunks_in_sec)
        lead = chunks_in_sec[0]
        cleaned_parts = []
        for ch in chunks_in_sec:
            body = re.sub(r"^> \*\*Context\*\*:.*?\n\n", "", ch["content"], flags=re.DOTALL)
            cleaned_parts.append(body.strip())
        merged_body = "\n\n".join(cleaned_parts)
        formatted_sections.append(f"## {lead['breadcrumbs']}\n\n{merged_body}")

    text_mode_b = "\n\n---\n\n".join(formatted_sections)
    facts_b = score_facts(text_mode_b)

    print(f"Mode B Retrieved: {len(mode_b_sections)} unified sections ({total_expanded_chunks} total clauses) in {time_mode_b_retrieval:.1f}ms")
    print(f"Mode B Fact Coverage: {facts_b['score']}/{facts_b['total']} ({facts_b['percentage']}%)")
    print(f"Mode B Context Length: {len(text_mode_b)} characters (~{len(text_mode_b)//4} tokens)")

    # -----------------------------------------------------------------------
    # Distribution Testing: Evaluator Sufficiency (n = 5 trials)
    # -----------------------------------------------------------------------
    N_TRIALS = 5
    print("\n" + "-" * 70)
    print(f"DISTRIBUTION TESTING (n = {N_TRIALS} trials per mode) WITH gemma-4-31b-it")
    print("-" * 70)

    sys_prompt = (
        "You are an evidence sufficiency evaluator. Respond ONLY in valid JSON.\n"
        "Analyze whether the collected evidence directly and completely answers the user's question.\n"
        "Format: {\"sufficient\": true/false, \"thought\": \"...\", \"confidence\": 0.0-1.0}"
    )

    def evaluate_trials(mode_name: str, evidence: str) -> List[Dict[str, Any]]:
        user_prompt = f"Question: {SCENARIO_PROMPT}\n\nCollected Evidence:\n{evidence}\n\nIs this evidence sufficient to answer all parts A, B, C, and D of the question?"
        results = []
        print(f"Evaluating {mode_name} ({N_TRIALS} trials)...")
        for t in range(N_TRIALS):
            t_start = time.perf_counter()
            raw = llm.chat(sys_prompt, user_prompt)
            latency = (time.perf_counter() - t_start) * 1000.0
            parsed = {"sufficient": False, "confidence": 0.0, "thought": "parse_failure", "latency_ms": latency}
            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    parsed_json = json.loads(m.group(0))
                    parsed["sufficient"] = bool(parsed_json.get("sufficient", False))
                    parsed["confidence"] = float(parsed_json.get("confidence", 0.0))
                    parsed["thought"] = str(parsed_json.get("thought", ""))
            except Exception as e:
                parsed["thought"] = f"Error: {e}"
            results.append(parsed)
            print(f"  [{mode_name} Trial {t + 1}/{N_TRIALS}] Sufficient: {parsed['sufficient']} | Conf: {parsed['confidence']:.2f} | Latency: {parsed['latency_ms']:.0f}ms")
        return results

    evals_a = evaluate_trials("Mode A (Atomic)", text_mode_a)
    evals_b = evaluate_trials("Mode B (Generic AST Expansion)", text_mode_b)

    # Compute aggregate stats
    def aggregate(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        suff_count = sum(1 for t in trials if t["sufficient"])
        confs = [t["confidence"] for t in trials]
        lats = [t["latency_ms"] for t in trials]
        return {
            "pass_count": suff_count,
            "pass_rate": suff_count / len(trials),
            "mean_conf": sum(confs) / len(confs),
            "median_lat": sorted(lats)[len(lats) // 2],
            "raw_trials": trials,
        }

    stats_a = aggregate(evals_a)
    stats_b = aggregate(evals_b)

    # -----------------------------------------------------------------------
    # Comparative Summary Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON RESULTS (n = 5 Distribution)")
    print("=" * 80)
    print("| Metric | Mode A: Status Quo (Atomic) | Mode B: Generic AST Expansion | Delta |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| Fact Coverage (Turn 1) | {facts_a['score']}/{facts_a['total']} ({facts_a['percentage']}%) | {facts_b['score']}/{facts_b['total']} ({facts_b['percentage']}%) | +{facts_b['score'] - facts_a['score']} facts (+{round(facts_b['percentage'] - facts_a['percentage'], 1)}%) |")
    print(f"| Sufficiency Pass Rate (n=5) | {stats_a['pass_count']}/{N_TRIALS} ({stats_a['pass_rate']*100:.0f}%) | {stats_b['pass_count']}/{N_TRIALS} ({stats_b['pass_rate']*100:.0f}%) | +{round((stats_b['pass_rate'] - stats_a['pass_rate'])*100, 1)}% |")
    print(f"| Mean Evaluator Confidence | {stats_a['mean_conf']:.2f} | {stats_b['mean_conf']:.2f} | +{round(stats_b['mean_conf'] - stats_a['mean_conf'], 2)} |")
    print(f"| Context Length (Chars) | {len(text_mode_a):,} chars (~{len(text_mode_a)//4:,} tok) | {len(text_mode_b):,} chars (~{len(text_mode_b)//4:,} tok) | +{len(text_mode_b) - len(text_mode_a):,} chars |")
    print(f"| Cohesive Sections | {len(mode_a_chunks)} disjointed chunks | {len(mode_b_sections)} unified sections | Consolidated |")
    print(f"| Median Eval Latency | {stats_a['median_lat']:.0f}ms | {stats_b['median_lat']:.0f}ms | +{stats_b['median_lat'] - stats_a['median_lat']:.0f}ms |")

    # Detailed Checklist Breakdown
    print("\n" + "-" * 70)
    print("CHECKLIST BREAKDOWN")
    print("-" * 70)
    for item in CHECKLIST:
        hit_a = "YES" if facts_a["details"].get(item["id"]) else "NO"
        hit_b = "YES" if facts_b["details"].get(item["id"]) else "NO"
        print(f"  * {item['name']:<48} | Mode A: {hit_a} | Mode B: {hit_b}")

    # Save to test logs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"docs_internal/test_logs/{timestamp}_benchmark_cary_generic_ast_expansion.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Cary Generic AST Section Expansion Benchmark\n\n")
        f.write(f"Model: gemma-4-31b-it on Google AI Studio\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Trials per mode: {N_TRIALS}\n\n")
        f.write(f"Mode A Facts: {facts_a['score']}/{facts_a['total']} ({facts_a['percentage']}%)\n")
        f.write(f"Mode B Facts: {facts_b['score']}/{facts_b['total']} ({facts_b['percentage']}%)\n\n")
        f.write(f"Mode A Stats: pass={stats_a['pass_count']}/{N_TRIALS}, mean_conf={stats_a['mean_conf']:.2f}, median_lat={stats_a['median_lat']:.0f}ms\n")
        f.write(f"Mode B Stats: pass={stats_b['pass_count']}/{N_TRIALS}, mean_conf={stats_b['mean_conf']:.2f}, median_lat={stats_b['median_lat']:.0f}ms\n\n")
        f.write("## Mode A Trials\n")
        f.write(json.dumps(stats_a["raw_trials"], indent=2) + "\n\n")
        f.write("## Mode B Trials\n")
        f.write(json.dumps(stats_b["raw_trials"], indent=2) + "\n")
    print(f"\nSaved empirical log to: {log_path}")


if __name__ == "__main__":
    run_benchmark()
