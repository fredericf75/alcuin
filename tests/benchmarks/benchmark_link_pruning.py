#!/usr/bin/env python3
"""
Benchmark harness for link pruning in Alcuin:
Compares:
1. Candidate 1: Laya Decision Model (ModernBERT-large non-autoregressive encoder)
2. Candidate 2: Embedding Cosine Similarity (nomic-embed-text via Ollama)
3. Candidate 3: Fast SLM Choice (llama3.2:3b via Ollama zero-shot prompt)
4. Baseline: Current heuristic (_is_in_inquiry_scope keyword/regex overlap)

Measures:
- Latency (mean, p50, p95 in ms)
- Accuracy, Precision, Recall, F1
- Precision@2 & Recall@2 (top-2 ranking quality)
- Token overhead & failure modes
Runs n=5 trials per candidate to satisfy empirical distribution rigor.
"""

import os
import sys
import time
import json
import math
import statistics
from typing import List, Dict, Any, Tuple
import httpx

# Ensure alcuin package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from alcuin.agent.graph import ResearchAgent

# ---------------------------------------------------------------------------
# 1. Curated Benchmark Dataset (12 Scenarios, 62 Link Decisions)
# ---------------------------------------------------------------------------
DATASET = [
    {
        "id": "med_pediatrician",
        "domain": "Personal/Medical",
        "query": "What is the phone number and address for Charlie's pediatrician?",
        "source": "Charlie Profile.md",
        "links": [
            {"target": "Dr. Sarah Miller Pediatrics", "context": "Primary pediatrician contact, clinic address and immunization records.", "label": 1},
            {"target": "Charlie Pediatric Clinic Log", "context": "Recent checkup dates, vital signs, and clinic location on Main St.", "label": 1},
            {"target": "Dr. Miller Orthodontics", "context": "Dental braces adjustment schedule for Charlie.", "label": 0},
            {"target": "Pediatric Asthma Overview", "context": "General medical brochure regarding childhood respiratory conditions.", "label": 0},
            {"target": "Toyota Sienna Maintenance", "context": "Oil change, tire rotation, and car seat installation notes.", "label": 0},
        ],
    },
    {
        "id": "insurance_deductible",
        "domain": "Financial/Insurance",
        "query": "What is our home insurance hurricane deductible and policy number?",
        "source": "Insurance Overview.md",
        "links": [
            {"target": "State Farm Homeowners Policy 2025", "context": "Declaration page detailing windstorm, hurricane deductible, and coverage limits.", "label": 1},
            {"target": "Roof Inspection & Insurance Claim", "context": "Adjuster report for hail damage citing policy number and hurricane rider.", "label": 1},
            {"target": "GEICO Auto Insurance Policy", "context": "Collision and comprehensive deductible for vehicle fleet.", "label": 0},
            {"target": "Hurricane Preparedness Checklist", "context": "Emergency kit, canned food supplies, and generator guidelines.", "label": 0},
            {"target": "Home Depot Tool Receipts", "context": "Invoices for plywood sheets and storm shutters.", "label": 0},
        ],
    },
    {
        "id": "tax_1099_deductions",
        "domain": "Financial/Tax",
        "query": "Where are our 1099-MISC contractor expense receipts for the home office?",
        "source": "Taxes 2024.md",
        "links": [
            {"target": "Contractor Invoices 2024", "context": "Detailed line items for freelance developers and 1099 forms.", "label": 1},
            {"target": "Home Office Renovation Deductions", "context": "Itemized depreciation schedules and square footage allocation.", "label": 1},
            {"target": "W-2 Tax Return Summary", "context": "Corporate salary withholdings and standard deductions.", "label": 0},
            {"target": "Office Furniture Assembly Instructions", "context": "Manual for adjustable standing desk from IKEA.", "label": 0},
            {"target": "Property Tax Assessment Notice", "context": "County real estate valuation and municipal tax bill.", "label": 0},
        ],
    },
    {
        "id": "zoning_fence_setback",
        "domain": "Municipal/Zoning",
        "query": "What is the maximum fence height and rear setback in residential district R-40?",
        "source": "Town Ordinances Chapter 6.md",
        "links": [
            {"target": "Sec. 6-118 Residential Fences", "context": "Height limitations for side and rear yard fencing in R-40 zones.", "label": 1},
            {"target": "Sec. 6-28 Yard Setbacks and Buffers", "context": "Rear, front, and corner lot setback requirements for residential structures.", "label": 1},
            {"target": "Sec. 8-10 Commercial Signage", "context": "Illuminated signs and billboard heights along arterial roads.", "label": 0},
            {"target": "Fence Installation Invoice", "context": "Estimate from Apex Fencing for cedar dog-ear perimeter fence.", "label": 0},
            {"target": "Dog Waste Ordinance", "context": "Pet leash laws and fines for running at large in residential areas.", "label": 0},
        ],
    },
    {
        "id": "legal_deed_easement",
        "domain": "Legal/Property",
        "query": "Is there a utility easement granting right of entry on our western boundary?",
        "source": "Property Title & Closing.md",
        "links": [
            {"target": "Warranty Deed and Plat Map", "context": "Metes and bounds legal description specifying Duke Energy 15-foot western easement.", "label": 1},
            {"target": "Title Insurance Policy Exceptions", "context": "Schedule B covenants, encumbrances, and utility right of way reservations.", "label": 1},
            {"target": "HOA Architectural Guidelines", "context": "Permitted paint colors, roofing materials, and mailbox standards.", "label": 0},
            {"target": "Utility Bills Summary", "context": "Monthly power and natural gas expenditure charts.", "label": 0},
            {"target": "Neighbor Boundary Dispute Email", "context": "Correspondence regarding overgrown crepe myrtles along boundary.", "label": 0},
        ],
    },
    {
        "id": "medical_immunization",
        "domain": "Personal/Medical",
        "query": "Which vaccines does Maya need before starting kindergarten in the fall?",
        "source": "Maya Records.md",
        "links": [
            {"target": "State Kindergarten Vaccine Requirements", "context": "Mandatory DTaP, MMR, Polio, and Varicella doses required for enrollment.", "label": 1},
            {"target": "Maya Official Immunization Record", "context": "Pediatrician clinic certified record of administered childhood vaccinations.", "label": 1},
            {"target": "Kindergarten School Calendar", "context": "First day of school, teacher workdays, and holiday breaks.", "label": 0},
            {"target": "Maya Allergy Action Plan", "context": "EpiPen instructions and tree nut avoidance protocol for cafeteria.", "label": 0},
            {"target": "School Supply List", "context": "Backpack, glue sticks, crayons, and safety scissors checklist.", "label": 0},
        ],
    },
    {
        "id": "auto_lease_mileage",
        "domain": "Financial/Auto",
        "query": "What is the annual mileage allowance and excess per-mile charge on the Honda lease?",
        "source": "Vehicle Leases.md",
        "links": [
            {"target": "Honda Financial Lease Agreement", "context": "Section 14: 12,000 annual miles allowed, $0.20 per mile excess charge.", "label": 1},
            {"target": "Honda Turn-In & End of Lease Inspection", "context": "Wear and tear standards, odometer verification, and disposition fee.", "label": 1},
            {"target": "Honda Scheduled Maintenance Intervals", "context": "B16 service guide: engine oil, differential fluid, and tire rotation.", "label": 0},
            {"target": "Gasoline Mileage Tracker", "context": "Fuel economy logs, road trip MPG records, and fuel receipts.", "label": 0},
            {"target": "Auto Insurance Proof of Coverage", "context": "State minimum liability limits and lienholder notification card.", "label": 0},
        ],
    },
    {
        "id": "server_backup_retention",
        "domain": "Technical/DevOps",
        "query": "How many daily and weekly snapshots are retained for the PostgreSQL database?",
        "source": "Infrastructure Specs.md",
        "links": [
            {"target": "Postgres Backup & Retention Policy", "context": "pgBackRest automated schedule: 7 daily incrementals, 4 weekly fulls, 12 monthly archives.", "label": 1},
            {"target": "Disaster Recovery Runbook", "context": "RTO and RPO targets with step-by-step point-in-time database restoration.", "label": 1},
            {"target": "Postgres Slow Query Optimization", "context": "Index tuning and vacuum parameters for large transaction tables.", "label": 0},
            {"target": "AWS S3 Monthly Billing Report", "context": "Storage tier costs for standard, infrequent access, and Glacier storage.", "label": 0},
            {"target": "Docker Compose Production Config", "context": "Container environment variables, port mappings, and volume mounts.", "label": 0},
        ],
    },
    {
        "id": "will_executor_power",
        "domain": "Legal/Estate",
        "query": "Who is designated as alternate executor and what powers do they have over estate accounts?",
        "source": "Estate Planning Portfolio.md",
        "links": [
            {"target": "Last Will and Testament of Frederic", "context": "Article III: Appointment of primary and alternate executors and administrative powers.", "label": 1},
            {"target": "Revocable Living Trust Agreement", "context": "Successor trustee duties, asset liquidation permissions, and probate bypass.", "label": 1},
            {"target": "Healthcare Power of Attorney", "context": "Advance directive for medical decisions and living will declarations.", "label": 0},
            {"target": "Funeral Wishes and Memorial Notes", "context": "Preference for cremation, service music selection, and burial plot details.", "label": 0},
            {"target": "Safety Deposit Box Access Log", "context": "Bank branch address and key location at Wells Fargo.", "label": 0},
        ],
    },
    {
        "id": "ordinance_tree_removal",
        "domain": "Municipal/Zoning",
        "query": "Can I remove an oak tree over 8 inches diameter without a town permit?",
        "source": "Tree Protection Ordinance.md",
        "links": [
            {"target": "Sec. 11.3.7 Removal and Pruning of Champion Trees", "context": "Permit required for removal or significant pruning of specimen trees 8 inches or greater.", "label": 1},
            {"target": "Urban Forest Mitigation & Tree Replacement Table", "context": "Tree replacement caliper ratio and civil penalty matrix for unpermitted clearings.", "label": 1},
            {"target": "Arborist Tree Pruning Estimates", "context": "Quote from Bartlett Tree Experts for deadwood pruning and canopy reduction.", "label": 0},
            {"target": "Firewood Storage Guidelines", "context": "Stacking distance from house to prevent termite infestation.", "label": 0},
            {"target": "Storm Debris Curbside Pickup Schedule", "context": "Yard waste collection guidelines for branches under four feet.", "label": 0},
        ],
    },
    {
        "id": "dental_orthodontics_coverage",
        "domain": "Financial/Medical",
        "query": "What is the lifetime orthodontic benefit limit under our Delta Dental plan?",
        "source": "Health Benefits.md",
        "links": [
            {"target": "Delta Dental Comprehensive Benefit Booklet", "context": "Class IV Orthodontia: 50% co-insurance up to $2,000 lifetime maximum per child.", "label": 1},
            {"target": "Charlie Orthodontist Pre-Authorization Claim", "context": "Treatment estimate showing carrier orthodontic allowance and family copay.", "label": 1},
            {"target": "Annual Dental Hygiene Schedule", "context": "Biannual cleaning appointments and fluoride treatment dates.", "label": 0},
            {"target": "HSA Contribution and Balance Sheet", "context": "FSA/HSA pre-tax payroll deductions and eligible medical supplies.", "label": 0},
            {"target": "Vision Care Plan Summary", "context": "Eyeglass frame discounts and annual optometry exam allowances.", "label": 0},
        ],
    },
    {
        "id": "mortgage_escrow_analysis",
        "domain": "Financial/RealEstate",
        "query": "Why did our monthly mortgage payment increase in the annual escrow statement?",
        "source": "Mortgage Statements.md",
        "links": [
            {"target": "Annual Escrow Analysis Statement 2025", "context": "Breakdown showing county property tax reassessment and insurance premium shortfall.", "label": 1},
            {"target": "County Real Estate Tax Revaluation Notice", "context": "Assessed market value increase and updated millage rate.", "label": 1},
            {"target": "Mortgage Amortization Schedule", "context": "Principal and interest breakdown across 30-year fixed loan term.", "label": 0},
            {"target": "HOA Dues Annual Statement", "context": "Quarterly community association fees and capital reserve assessment.", "label": 0},
            {"target": "Refinance Closing Disclosure", "context": "Historical loan origination points, title fees, and interest rate lock.", "label": 0},
        ],
    },
]

# ---------------------------------------------------------------------------
# 2. Pruning Evaluator Implementations
# ---------------------------------------------------------------------------

class BaseEvaluator:
    name: str = "Base"

    def score_links(self, query: str, source: str, links: List[Dict[str, Any]]) -> List[float]:
        """Return a relevance score in [0.0, 1.0] for each candidate link."""
        raise NotImplementedError


class LayaEvaluator(BaseEvaluator):
    name = "Laya (System 1 Decision Model)"

    def __init__(self):
        import laya
        self.router = laya.Router(default="english", device="cpu")
        self.question = {
            "relevant": {
                "type": "choice",
                "instructions": "Does this candidate wikilink help answer the research query?",
                "criteria": ["relevant", "irrelevant"],
            }
        }

    def score_links(self, query: str, source: str, links: List[Dict[str, Any]]) -> List[float]:
        scores = []
        for l in links:
            state = f"Query: {query}\nSource Note: {source}\nCandidate Link: [[{l['target']}]]\nContext: {l['context']}"
            res = self.router.predict(state, self.question)
            probs = res.get("answers", {}).get("relevant", {}).get("probabilities", {})
            score = float(probs.get("relevant", 0.0))
            scores.append(score)
        return scores


class EmbeddingCosineEvaluator(BaseEvaluator):
    name = "Embedding Cosine (nomic-embed-text)"

    def __init__(self, endpoint: str = "http://localhost:11434"):
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=10.0)

    def _get_embedding(self, text: str) -> List[float]:
        resp = self.client.post(
            f"{self.endpoint}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
        return []

    def score_links(self, query: str, source: str, links: List[Dict[str, Any]]) -> List[float]:
        q_vec = self._get_embedding(query)
        if not q_vec:
            return [0.0] * len(links)

        q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0

        scores = []
        for l in links:
            l_text = f"{l['target']}: {l['context']}"
            l_vec = self._get_embedding(l_text)
            if not l_vec:
                scores.append(0.0)
                continue
            l_norm = math.sqrt(sum(x * x for x in l_vec)) or 1.0
            dot = sum(a * b for a, b in zip(q_vec, l_vec))
            sim = max(0.0, dot / (q_norm * l_norm))
            scores.append(sim)
        return scores


class SLMEvaluator(BaseEvaluator):
    name = "Fast SLM (llama3.2:3b @ temp=0)"

    def __init__(self, endpoint: str = "http://localhost:11434"):
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=15.0)

    def score_links(self, query: str, source: str, links: List[Dict[str, Any]]) -> List[float]:
        scores = []
        sys_prompt = (
            "You are a link pruning specialist. Given a user query and a candidate wikilink with snippet, "
            "determine if traversing the link is relevant to answering the query.\n"
            "Reply strictly with either RELEVANT or IRRELEVANT."
        )
        for l in links:
            user_prompt = (
                f"Query: {query}\n"
                f"Candidate Link: [[{l['target']}]]\n"
                f"Context: {l['context']}\n"
                "Classification:"
            )
            try:
                resp = self.client.post(
                    f"{self.endpoint}/api/chat",
                    json={
                        "model": "llama3.2:3b",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                )
                if resp.status_code == 200:
                    text = resp.json().get("message", {}).get("content", "").upper()
                    scores.append(1.0 if "RELEVANT" in text and "IRRELEVANT" not in text else 0.0)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores


class BaselineHeuristicEvaluator(BaseEvaluator):
    name = "Baseline Heuristic (_is_in_inquiry_scope)"

    def score_links(self, query: str, source: str, links: List[Dict[str, Any]]) -> List[float]:
        scores = []
        for l in links:
            text = f"{l['target']} {l['context']}"
            in_scope = ResearchAgent._is_in_inquiry_scope(text, query)
            scores.append(1.0 if in_scope else 0.0)
        return scores


# ---------------------------------------------------------------------------
# 3. Benchmark Runner & Metric Aggregator
# ---------------------------------------------------------------------------

def compute_metrics(evaluator: BaseEvaluator, n_trials: int = 5) -> Dict[str, Any]:
    trial_latencies = []
    trial_precisions_at_2 = []
    trial_recalls_at_2 = []
    trial_accuracies = []
    trial_f1s = []

    print(f"\nEvaluating: {evaluator.name} across {n_trials} trials...")

    for trial_idx in range(n_trials):
        start_trial = time.perf_counter()
        correct_classifications = 0
        total_decisions = 0
        p2_hits = 0
        r2_captured = 0
        total_true_links = 0

        tp = fp = tn = fn = 0

        for scenario in DATASET:
            q = scenario["query"]
            src = scenario["source"]
            links = scenario["links"]
            labels = [l["label"] for l in links]
            num_true = sum(labels)
            total_true_links += num_true

            t0 = time.perf_counter()
            scores = evaluator.score_links(q, src, links)
            t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms for scenario

            per_link_lat = t_elapsed / len(links)
            trial_latencies.append(per_link_lat)

            # Binary classification metrics (threshold 0.5)
            for s, y in zip(scores, labels):
                pred = 1 if s >= 0.5 else 0
                if pred == y:
                    correct_classifications += 1
                if pred == 1 and y == 1:
                    tp += 1
                elif pred == 1 and y == 0:
                    fp += 1
                elif pred == 0 and y == 1:
                    fn += 1
                else:
                    tn += 1
                total_decisions += 1

            # Precision@2 and Recall@2 (ranking)
            paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
            top_2 = paired[:2]
            hits_in_top2 = sum(y for s, y in top_2)
            p2_hits += hits_in_top2
            r2_captured += hits_in_top2

        acc = (correct_classifications / total_decisions) * 100.0 if total_decisions else 0.0
        prec_2 = (p2_hits / (len(DATASET) * 2)) * 100.0
        rec_2 = (r2_captured / total_true_links) * 100.0 if total_true_links else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) * 100.0 if (precision + recall) > 0 else 0.0

        trial_accuracies.append(acc)
        trial_precisions_at_2.append(prec_2)
        trial_recalls_at_2.append(rec_2)
        trial_f1s.append(f1)

        print(f"  Trial {trial_idx + 1}/{n_trials}: Acc={acc:.1f}%, P@2={prec_2:.1f}%, F1={f1:.1f}%")

    sorted_lat = sorted(trial_latencies)
    p50_lat = statistics.median(sorted_lat)
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]
    mean_lat = statistics.mean(sorted_lat)

    return {
        "name": evaluator.name,
        "n_trials": n_trials,
        "n_decisions": len(DATASET) * len(DATASET[0]["links"]),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "accuracy_mean": round(statistics.mean(trial_accuracies), 1),
        "accuracy_std": round(statistics.stdev(trial_accuracies) if len(trial_accuracies) > 1 else 0.0, 1),
        "p2_mean": round(statistics.mean(trial_precisions_at_2), 1),
        "r2_mean": round(statistics.mean(trial_recalls_at_2), 1),
        "f1_mean": round(statistics.mean(trial_f1s), 1),
    }


def main():
    print("=" * 80)
    print("ALCUIN SYSTEM 1 LINK PRUNING BENCHMARK")
    print(f"Dataset: {len(DATASET)} Scenarios | Total Link Decisions per Trial: 60 | Trials: n=5")
    print("=" * 80)

    evaluators = [
        LayaEvaluator(),
        EmbeddingCosineEvaluator(),
        SLMEvaluator(),
        BaselineHeuristicEvaluator(),
    ]

    results = []
    for ev in evaluators:
        res = compute_metrics(ev, n_trials=5)
        results.append(res)

    # Print Summary Markdown Table
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK RESULTS")
    print("=" * 80)
    print("| Candidate | p50 Latency (ms) | p95 Latency (ms) | Accuracy (%) | Precision@2 (%) | Recall@2 (%) | F1 Score (%) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(
            f"| {r['name']} | {r['p50_latency_ms']} | {r['p95_latency_ms']} | "
            f"{r['accuracy_mean']}% ± {r['accuracy_std']} | {r['p2_mean']}% | {r['r2_mean']}% | {r['f1_mean']}% |"
        )

    # Save to test logs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"docs_internal/test_logs/{timestamp}_benchmark_link_pruning.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Link Pruning Benchmark Log\n\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Dataset Scenarios: {len(DATASET)}\n\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n")
    print(f"\nArtifact saved to: {log_path}")


if __name__ == "__main__":
    main()
