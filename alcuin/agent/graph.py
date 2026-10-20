"""
alcuin.agent.graph - LangGraph Cyclic ReAct Research Agent with Hard Turn Ceiling
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypedDict

from dotenv import load_dotenv
import httpx
from langgraph.graph import END, START, StateGraph

from alcuin.storage.engine import SearchResult, VaultStorage
from alcuin.storage.vault import Vault, VaultRegistry

load_dotenv()


class AgentState(TypedDict):
    question: str
    iteration: int
    max_iterations: int
    max_hops: int
    current_query: str
    visited_links: List[str]
    pending_links: List[str]
    sub_goals: List[str]
    sub_questions: List[str]
    visited_queries: List[str]
    recomposed_topics: List[str]
    evidence: List[Dict[str, Any]]
    thought_history: List[str]
    is_sufficient: bool
    final_answer: str
    confidence: float
    sources: List[str]
    governor_alert: Optional[str]
    truncation_alert: Optional[str]
    evidence_only: bool


def _extract_and_clean_thinking(raw_text: str) -> Tuple[str, str]:
    """Extract latent reasoning enclosed in <think>...</think> tags (or template-prefixed </think>) and return (clean_text, thinking_trace)."""
    if not raw_text:
        return "", ""
    thinking = ""
    if "<think>" in raw_text:
        m = re.search(r"<think>(.*?)(?:</think>|$)", raw_text, flags=re.DOTALL)
        if m:
            thinking = m.group(1).strip()
        cleaned_text = re.sub(r"<think>.*?(?:</think>|$)", "", raw_text, flags=re.DOTALL).strip()
        return cleaned_text, thinking
    if "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        thinking = parts[0].strip()
        cleaned_text = parts[1].strip()
        return cleaned_text, thinking
    return raw_text.strip(), ""


class LLMClient:
    """Interacts with LLM endpoint (Ollama / Google AI Studio / OpenAI-compatible / Fallback)."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        context_window: Optional[int] = None,
        think: Optional[bool] = None,
    ):
        global_cfg = VaultRegistry.get_config_dir()
        cfg_file = global_cfg / "config.toml"
        g_model = ""
        g_endpoint = ""
        if cfg_file.exists():
            try:
                import tomllib
                with open(cfg_file, "rb") as f:
                    data = tomllib.load(f)
                    g_model = data.get("agent", {}).get("model_name", "")
                    g_endpoint = data.get("agent", {}).get("model_endpoint", "")
            except Exception:
                pass

        resolved_endpoint = endpoint or os.getenv("ALCUIN_ENDPOINT") or g_endpoint or "http://localhost:11434"
        resolved_model = model or os.getenv("ALCUIN_MODEL") or g_model or ""

        self.endpoint = resolved_endpoint.rstrip("/")
        self.model = resolved_model
        self.temperature = temperature
        self.timeout = timeout
        self.context_window = context_window
        self.think = think
        self.api_key = (
            api_key
            or (os.getenv("OPENROUTER_API_KEY") if "openrouter" in resolved_endpoint else None)
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        return_metadata: bool = False,
    ) -> Any:
        is_local = "localhost" in self.endpoint or "127.0.0.1" in self.endpoint
        is_openrouter = "openrouter.ai" in self.endpoint
        is_google = (
            not is_openrouter
            and not is_local
            and (
                "generativelanguage.googleapis.com" in self.endpoint
                or "google" in self.endpoint
                or (self.api_key and self.model.startswith(("gemini", "gemma")))
            )
        )

        # 1. If Google AI Studio (native generateContent API)
        if is_google and self.api_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": {"temperature": self.temperature},
                }
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            finish_reason = candidates[0].get("finishReason", "")
                            is_truncated = finish_reason in ("MAX_TOKENS", "LENGTH")
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = ""
                            if parts:
                                non_thought = [p.get("text", "") for p in parts if not p.get("thought", False) and p.get("text")]
                                if non_thought:
                                    text = "\n".join(non_thought).strip()
                                else:
                                    text = parts[-1].get("text", "").strip()
                            clean_text, thinking = _extract_and_clean_thinking(text)
                            meta = {
                                "truncated": is_truncated,
                                "finish_reason": finish_reason,
                                "provider": "google",
                                "thinking": thinking,
                            }
                            if return_metadata:
                                return clean_text, meta
                            return clean_text
            except Exception:
                pass

        # 2. If local endpoint (Ollama /api/chat)
        if is_local:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    ollama_options: Dict[str, Any] = {"temperature": self.temperature}
                    if self.context_window:
                        ollama_options["num_ctx"] = int(self.context_window)
                    ollama_body: Dict[str, Any] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": ollama_options,
                    }
                    if self.think is not None:
                        ollama_body["think"] = self.think
                    resp = client.post(
                        f"{self.endpoint}/api/chat",
                        json=ollama_body,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        done_reason = data.get("done_reason", "")
                        is_truncated = done_reason in ("length", "max_tokens")
                        msg = data.get("message", {})
                        ollama_thinking = msg.get("thinking", "").strip() if isinstance(msg, dict) else ""
                        text = msg.get("content", "").strip()
                        if not text and ollama_thinking and is_truncated:
                            text = ollama_thinking
                        clean_text, extracted_thinking = _extract_and_clean_thinking(text)
                        final_thinking = ollama_thinking or extracted_thinking
                        meta = {
                            "truncated": is_truncated,
                            "finish_reason": done_reason,
                            "provider": "ollama",
                            "thinking": final_thinking,
                        }
                        if return_metadata:
                            return clean_text, meta
                        return clean_text
            except Exception:
                pass

        # 3. OpenAI-compatible /v1/chat/completions (hosted or local)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                base_ep = self.endpoint.rstrip("/")
                if base_ep.endswith("/chat/completions"):
                    target_url = base_ep
                elif base_ep.endswith("/v1"):
                    target_url = f"{base_ep}/chat/completions"
                else:
                    target_url = f"{base_ep}/v1/chat/completions"
                resp = client.post(
                    target_url,
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": 2048,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        finish_reason = choices[0].get("finish_reason", "")
                        is_truncated = finish_reason in ("length", "max_tokens")
                        raw_text = choices[0].get("message", {}).get("content", "").strip()
                        clean_text, thinking = _extract_and_clean_thinking(raw_text)
                        meta = {
                            "truncated": is_truncated,
                            "finish_reason": finish_reason,
                            "provider": "openai",
                            "thinking": thinking,
                        }
                        if return_metadata:
                            return clean_text, meta
                        return clean_text
        except Exception:
            pass

        # 4. Fallback to Ollama if not already tried
        if not is_local:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    ollama_options: Dict[str, Any] = {"temperature": self.temperature}
                    if self.context_window:
                        ollama_options["num_ctx"] = int(self.context_window)
                    fallback_body: Dict[str, Any] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": ollama_options,
                    }
                    if self.think is not None:
                        fallback_body["think"] = self.think
                    resp = client.post(
                        f"{self.endpoint}/api/chat",
                        json=fallback_body,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        done_reason = data.get("done_reason", "")
                        is_truncated = done_reason in ("length", "max_tokens")
                        msg = data.get("message", {})
                        ollama_thinking = msg.get("thinking", "").strip() if isinstance(msg, dict) else ""
                        text = msg.get("content", "").strip()
                        if not text and ollama_thinking and is_truncated:
                            text = ollama_thinking
                        clean_text, extracted_thinking = _extract_and_clean_thinking(text)
                        final_thinking = ollama_thinking or extracted_thinking
                        meta = {
                            "truncated": is_truncated,
                            "finish_reason": done_reason,
                            "provider": "ollama",
                            "thinking": final_thinking,
                        }
                        if return_metadata:
                            return clean_text, meta
                        return clean_text
            except Exception:
                pass

        if return_metadata:
            return "", {"truncated": False, "finish_reason": "none", "provider": "none", "thinking": ""}
        return ""


class ResearchAgent:
    """Autonomous multi-hop knowledge research agent."""

    def __init__(
        self,
        vault: Vault,
        llm: Optional[LLMClient] = None,
        planner_llm: Optional[LLMClient] = None,
    ):
        self.vault = vault
        self.storage = VaultStorage(vault)
        ctx_win = getattr(vault.config, "context_window", 16384)
        self.llm = llm or LLMClient(
            endpoint=vault.config.model_endpoint,
            model=vault.config.model_name,
            temperature=vault.config.temperature,
            context_window=ctx_win,
        )
        if hasattr(self.llm, "context_window") and self.llm.context_window is None:
            self.llm.context_window = ctx_win
        self.planner_llm = planner_llm or self.llm
        self.graph = self._build_graph()

    def get_model_context_window(self) -> int:
        """Return configured model context window from vault config or LLM client."""
        if hasattr(self.vault, "config") and hasattr(self.vault.config, "context_window"):
            return int(self.vault.config.context_window)
        if hasattr(self.llm, "context_window") and self.llm.context_window is not None:
            return int(self.llm.context_window)
        return 16384

    def resolve_model_headroom(self, window_tokens: int) -> int:
        """Resolve dynamic model generation headroom with a 50% safety cap on context window."""
        cfg_headroom = getattr(self.vault.config, "context_headroom_tokens", None)
        hr = int(cfg_headroom) if cfg_headroom is not None else 2000
        max_safe = max(int(0.50 * window_tokens), 1000)
        return min(hr, max_safe)

    def get_context_governor_budget(self) -> int:
        """Compute the calibrated evidence token ceiling: min(configured, window - headroom, 0.65 * window)."""
        cfg_max = getattr(self.vault.config, "context_governor_max_tokens", None)
        if cfg_max is not None:
            return int(cfg_max)
        window = self.get_model_context_window()
        headroom = self.resolve_model_headroom(window)
        budget = min(window - headroom, int(0.65 * window))
        return max(budget, 2000)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("synthesize", self._synthesize_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self._route_evaluation,
            {"retrieve": "retrieve", "synthesize": "synthesize", "end": END},
        )
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def decompose_query(self, question: str) -> Tuple[List[str], List[str]]:
        """Decompose a complex or multi-part query into:
        1. sub_goals: targeted keyword queries for document search.
        2. sub_questions: full grammatical questions for answer synthesis.
        """
        has_subparts = bool(re.search(r"(?:[A-Z]\.|\b\d+[\.\)])", question))
        is_long = len(question.split()) > 25

        if not (has_subparts or is_long):
            return [question], [question]

        # 1. Deterministic extraction: Full numbered or lettered items
        matches = re.findall(r"(?:^|\n)\s*(?:[A-Z]\.|\d+[\.\)])\s*([^\n]+)", question)
        cleaned_items = [
            re.sub(r"^\s*(?:[A-Z]\.|\d+[\.\)])\s*", "", m).strip()
            for m in matches
            if len(m.strip()) >= 8
        ]

        if len(cleaned_items) >= 2:
            sub_questions = cleaned_items
            sub_goals: List[str] = []

            # Harvest domain-agnostic technical codes, citations, numbers, and identifiers across question
            global_identifiers = re.findall(
                r"\b(?:§\s*\d+|[A-Z]{2,}(?:[\-\/][A-Z0-9]+)*|\b[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+\b|\d+[\-\w]+)\b",
                question,
            )

            filler = {
                "within", "how", "many", "days", "after", "must", "the", "a", "an", "for",
                "is", "under", "what", "which", "of", "are", "in", "to", "at", "by", "on",
                "shall", "may", "can", "could", "be", "do", "does", "specifically", "rule",
                "section", "both", "either", "or", "and", "if", "there", "has", "have", "with",
            }

            for q in cleaned_items:
                q_clean = q.rstrip("?.:;!")
                clause_identifiers = re.findall(
                    r"\b(?:§\s*\d+|[A-Z]{2,}(?:[\-\/][A-Z0-9]+)*|\b[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+\b|\d+[\-\w]+)\b",
                    q_clean,
                )
                body = re.sub(
                    r"^(?:in|under|pursuant to|according to|for|with respect to|regarding)\s+[^,;:\n]+[,\-:]\s*",
                    "",
                    q_clean,
                    flags=re.IGNORECASE,
                )
                words = re.findall(r"\b[\w\-\.]+\b", body)
                meaningful = [w for w in words if w.lower() not in filler]
                anchors = list(dict.fromkeys(clause_identifiers + [g for g in global_identifiers if g.lower() in q.lower()]))
                content_terms = [w for w in meaningful if w not in anchors]

                if len(content_terms) <= 8:
                    selected = content_terms
                else:
                    selected = content_terms[:4] + content_terms[-4:]

                search_query = " ".join(dict.fromkeys(anchors + selected))
                if not search_query.strip():
                    search_query = q_clean
                sub_goals.append(search_query)

            return sub_goals, sub_questions

        # 2. Dual-field LLM decomposition for unstructured inquiries
        sys_prompt = (
            "You are a research query planner.\n"
            "Identify every distinct question, condition, metric, exception, or consequence raised in the user's inquiry.\n"
            "Respond with a JSON object adhering strictly to this schema:\n"
            "{\n"
            '  "sub_goals": [\n'
            '    {\n'
            '      "question": "Full grammatical question addressing one specific issue",\n'
            '      "search_query": "3 to 6 keywords for document retrieval"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Output ONLY valid JSON."
        )
        llm_resp = self.planner_llm.chat(sys_prompt, question[:2000])
        llm_resp, _ = _extract_and_clean_thinking(llm_resp)
        sub_goals: List[str] = []
        sub_questions: List[str] = []

        if llm_resp:
            try:
                m = re.search(r"\{.*\}", llm_resp, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    items = data.get("sub_goals", [])
                    for it in items:
                        if isinstance(it, dict):
                            q = str(it.get("question", "")).strip()
                            s = str(it.get("search_query", "")).strip()
                            if q and s:
                                sub_questions.append(q)
                                sub_goals.append(s)
                        elif isinstance(it, str) and it.strip():
                            sub_questions.append(it.strip())
                            sub_goals.append(it.strip())
            except Exception:
                pass

            if not sub_goals:
                try:
                    m = re.search(r"\[.*\]", llm_resp, re.DOTALL)
                    if m:
                        items = json.loads(m.group(0))
                        if isinstance(items, list):
                            for it in items:
                                if isinstance(it, dict):
                                    q = str(it.get("question", "")).strip()
                                    s = str(it.get("search_query", "")).strip()
                                    if q and s:
                                        sub_questions.append(q)
                                        sub_goals.append(s)
                                elif isinstance(it, str) and it.strip():
                                    sub_questions.append(it.strip())
                                    sub_goals.append(it.strip())
                except Exception:
                    pass

            # Resilient fallback if non-JSON lines
            if not sub_goals:
                for line in llm_resp.strip().splitlines():
                    l_clean = line.strip()
                    l_lower = l_clean.lower()
                    if any(l_lower.startswith(p) for p in ("here are", "here is", "to locate", "based on", "the following")):
                        continue
                    if l_clean.endswith(":"):
                        continue
                    cleaned = re.sub(r"^[\d\.\-\*\•\s]+", "", l_clean).strip()
                    cleaned = re.sub(r'^(?:(?:search\s+)?query:\s*)+', '', cleaned, flags=re.IGNORECASE).strip(' "\'')
                    if cleaned and len(cleaned.split()) >= 2 and cleaned not in sub_goals:
                        sub_goals.append(cleaned)
                        sub_questions.append(cleaned)

        if sub_goals:
            return sub_goals, sub_questions
        return [question], [question]

    @staticmethod
    def _normalize_section_key(section_title: str) -> str:
        """Strip trailing '(Part N)' from a section title to group split section chunks."""
        return re.sub(
            r"\s*\(part\s+\d+\)\s*$", "", (section_title or "").strip(), flags=re.IGNORECASE
        ).lower()

    @staticmethod
    def _matches_structural_anchor(
        r: SearchResult,
        query: str,
        include_doc_title: bool = True,
    ) -> bool:
        """Check if a chunk's structural header (section_title, breadcrumbs, document_title) or
        TABLE_DATA keys match the query's identifiers or keywords, strictly excluding body prose."""
        if not query or not query.strip():
            return False

        q_lower = query.lower()

        sec_title = (r.section_title or "").lower()
        doc_title = (r.document_title or "").lower()
        breadcrumbs = (r.breadcrumbs or "").lower()
        if not include_doc_title and doc_title and breadcrumbs.startswith(doc_title):
            breadcrumbs = breadcrumbs[len(doc_title) :].lstrip(" >")

        table_match = re.search(r"<!--\s*TABLE_DATA:\s*(.*?)\s*-->", r.content or "", re.DOTALL)
        table_text = table_match.group(1).lower() if table_match else ""

        header_parts = [sec_title, breadcrumbs, table_text]
        if include_doc_title:
            file_stem = os.path.splitext(os.path.basename(r.file_path or ""))[0].lower()
            header_parts.extend([doc_title, file_stem])
        anchor_text = " ".join(p for p in header_parts if p)
        if not anchor_text.strip():
            return False

        # Rule 2a: Structural Identifier Match (e.g. 6-132, 15.1.3, 1099-INT, Sec. 5, RFC 9110)
        compound_ids = re.findall(r"\b[a-z0-9]*\d+(?:[\-\.][a-z0-9]+)+\b", q_lower)
        prefixed_ids = [
            m.group(0).strip()
            for m in re.finditer(
                r"\b(?:sec\.?|section|§|article|chapter|clause|rfc|schedule|exhibit|appendix)\s*#?[0-9]+[a-z]?\b",
                q_lower,
            )
        ]
        identifiers = list(dict.fromkeys(compound_ids + prefixed_ids))
        if identifiers:
            if all(ident in anchor_text for ident in identifiers):
                return True

        # Rule 2b: Heading / Note Title / Table-Key Match on substantive query keywords
        stop_words = {
            "section", "article", "chapter", "clause", "schedule", "exhibit", "appendix",
            "under", "what", "when", "where", "which", "within", "many", "days", "must",
            "does", "from", "that", "this", "with", "have", "about", "there", "their",
        }
        keywords = [w for w in re.findall(r"[a-z]{4,}", q_lower) if w not in stop_words]
        if keywords:
            matched = sum(1 for w in keywords if w in anchor_text)
            required = max(1, math.ceil(0.65 * len(keywords)))
            if matched >= required:
                return True

        return False

    def prune_by_score_cliff(
        self,
        results: List[SearchResult],
        min_drop: float = 0.12,
        min_relative_drop: float = 0.20,
        query: str = "",
    ) -> List[SearchResult]:
        """Apply adaptive score-cliff pruning (Track 1) with non-body structural header & table rescue (Track 2)."""
        if len(results) <= 1:
            return results

        # Ensure results are strictly ordered descending by semantic score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        # Track 1: Pure Vector Cliff
        if sorted_results[0].score < 0.40:
            admitted: List[SearchResult] = []
            pruned: List[SearchResult] = list(sorted_results)
        else:
            scores = [r.score for r in sorted_results]
            deltas = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

            cutoff_idx = None
            for i, delta in enumerate(deltas):
                # Condition 1: Minimum absolute drop
                if delta < min_drop:
                    continue
                # Condition 2: Minimum relative drop from predecessor
                if scores[i] <= 0 or (delta / scores[i]) < min_relative_drop:
                    continue
                # Condition 3: Gradient dominance over previous plateau (if i > 0)
                if i > 0:
                    mean_prior = sum(deltas[:i]) / i
                    if delta < 2.0 * mean_prior:
                        continue
                cutoff_idx = i + 1
                break

            if cutoff_idx is not None:
                admitted = sorted_results[:cutoff_idx]
                pruned = sorted_results[cutoff_idx:]
            else:
                cap = min(len(sorted_results), 6)
                admitted = sorted_results[:cap]
                pruned = sorted_results[cap:]

        # Track 2: Non-Body Structural Header & Companion Table Rescue
        if pruned:
            distinct_files = {r.file_path for r in sorted_results if r.file_path}
            include_doc_title = len(distinct_files) > 1
            admitted_sec_keys = {
                self._normalize_section_key(a.section_title)
                for a in admitted
                if a.section_title and a.section_title != "Preamble / General"
            }

            for r in pruned:
                # 1. Rescue if structural header or TABLE_DATA matches query identifiers/keywords
                if query and self._matches_structural_anchor(r, query, include_doc_title=include_doc_title):
                    admitted.append(r)
                    if r.section_title and r.section_title != "Preamble / General":
                        admitted_sec_keys.add(self._normalize_section_key(r.section_title))
                    continue

                # 2. Rescue companion TABLE_DATA chunk sharing the same section_title as an admitted chunk
                if "<!-- TABLE_DATA:" in (r.content or "") and r.section_title:
                    if self._normalize_section_key(r.section_title) in admitted_sec_keys:
                        admitted.append(r)

        return admitted

    def refine_stagnated_topic(self, topic: str, question: str) -> List[str]:
        """Drill down and re-decompose a topic that yielded no new evidence (Depth-1)."""
        topic_words = set(re.findall(r"\w{4,}", topic.lower()))
        relevant_context = []
        for line in question.splitlines():
            line_clean = line.strip()
            if not line_clean or len(line_clean) < 10:
                continue
            line_words = set(re.findall(r"\w{4,}", line_clean.lower()))
            if any(w in line_words for w in topic_words):
                relevant_context.append(line_clean)

        context_str = " ".join(relevant_context) if relevant_context else question[:600]

        # 1. Ask LLM to refine into 2-3 concrete search queries
        sys_prompt = (
            "You are a search query refinement specialist.\n"
            f"The high-level query '{topic}' yielded insufficient information.\n"
            "Generate 2 to 3 concise, specific search queries (synonyms, technical terms, or concrete elements) "
            "designed to locate the relevant sections in a document database.\n"
            "Output each query on a new line. Do not number or prefix them."
        )
        llm_resp = self.planner_llm.chat(sys_prompt, f"Context:\n{context_str}\n\nHigh-level topic: {topic}")
        llm_resp, _ = _extract_and_clean_thinking(llm_resp)
        refined: List[str] = []
        if llm_resp:
            for line in llm_resp.strip().splitlines():
                cleaned = re.sub(r"^[\d\.\-\*\•\s]+", "", line).strip()
                if cleaned and len(cleaned.split()) >= 2 and cleaned.lower() != topic.lower() and cleaned not in refined:
                    refined.append(cleaned)

        # 2. Deterministic fallback if LLM is offline or empty: extract concrete phrases from relevant context
        if not refined and relevant_context:
            for ctx in relevant_context:
                phrases = re.findall(r"(?:[a-zA-Z]{3,}\s+){1,3}[a-zA-Z]{3,}", ctx)
                for p in phrases:
                    p_clean = p.strip()
                    if len(p_clean.split()) in (2, 3) and p_clean.lower() not in topic.lower() and p_clean not in refined:
                        refined.append(p_clean)
                        if len(refined) >= 3:
                            break
                if len(refined) >= 3:
                    break

        return refined[:3]

    @staticmethod
    def _is_in_inquiry_scope(text: str, question: str) -> bool:
        """Dynamically evaluate whether a candidate chunk or link falls within the inquiry's scope."""
        t_lower = text.lower()
        q_lower = question.lower()

        # 1. Always allow universal procedural or definitional keywords
        if any(term in t_lower for term in ("definition", "right of entry", "warrant", "appeal", "general penalty")):
            return True

        # 2. Extract structural scope identifiers (e.g. Chapters 6 and 8, Article V, RFC 7540)
        matches = re.findall(
            r"\b(?:chapters?|articles?|subchapters?|titles?)\s+([a-zA-Z0-9]+)(?:\s*(?:and|,|&)\s*([a-zA-Z0-9]+))?",
            q_lower,
        )
        scoped_numbers = set()
        for m in matches:
            for val in m:
                if val:
                    scoped_numbers.add(val.lower())

        if scoped_numbers:
            # Check statutory section prefix (e.g. Sec. 6-28 -> 6, Sec. 8-118 -> 8)
            sec_match = re.search(r"(?:sec\.|section|§)\s*([a-zA-Z0-9]+)[\-\.]", t_lower)
            if sec_match:
                prefix = sec_match.group(1).lower()
                # Permit if matching chapter or foundational general provisions (Chapters 1 or 2)
                if prefix in scoped_numbers or prefix in ("1", "2"):
                    return True
                # If chunk explicitly belongs to an unrequested chapter, filter out
                return False

            # If orphan schedule / section with no chapter tag, require chapter mention in text
            if "schedule" in t_lower and not any(f"chapter {num}" in t_lower for num in scoped_numbers):
                return False

        # 3. Substantive domain term overlap
        q_words = set(re.findall(r"[a-z]{4,}", q_lower))
        stop_words = {
            "detail", "every", "specific", "exact", "provide", "analyze", "analysis",
            "scenario", "following", "under", "what", "which", "where", "whom", "within",
            "many", "days", "when", "would", "could", "should", "construct", "markdown",
            "table", "matrix", "steps", "owner", "conduct", "uncover", "conditions",
            "primary", "entire", "claiming", "statutory", "authority", "versus", "absolute"
        }
        substantive = q_words - stop_words
        t_words = set(re.findall(r"[a-z]{4,}", t_lower))
        if any(w in substantive for w in t_words):
            return True

        return not bool(scoped_numbers)

    def score_candidate_links(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
    ) -> List[Tuple[str, float]]:
        """Score candidate wikilinks against the research query using embedding cosine similarity.
        
        Returns a list of (link_name, similarity_score) sorted descending by score.
        """
        if not candidates:
            return []

        try:
            q_vec = self.storage.embedder.get_embedding(query)
            q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0

            scored: List[Tuple[str, float]] = []
            for link_clean, snippet in candidates:
                link_text = f"{link_clean}: {snippet}" if snippet else link_clean
                l_vec = self.storage.embedder.get_embedding(link_text)
                l_norm = math.sqrt(sum(x * x for x in l_vec)) or 1.0
                dot = sum(a * b for a, b in zip(q_vec, l_vec))
                sim = max(0.0, dot / (q_norm * l_norm))
                scored.append((link_clean, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except Exception:
            # Fallback gracefully to retaining all candidates
            return [(link_clean, 1.0) for link_clean, _ in candidates]

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        evidence_list = list(state.get("evidence", []))
        visited_ids = {e["id"] for e in evidence_list}
        new_sources: Set[str] = set(state.get("sources", []))
        pending_links = list(state.get("pending_links", []))
        visited_links = list(state.get("visited_links", []))
        visited_queries = list(state.get("visited_queries", []))
        sub_goals = list(state.get("sub_goals", []))
        sub_questions = list(state.get("sub_questions", []))
        recomposed_topics = list(state.get("recomposed_topics", []))
        thought_history = list(state.get("thought_history", []))

        # 1. On turn 0, decompose question into discrete sub-goals and sub-questions
        if state["iteration"] == 0:
            sub_goals, sub_questions = self.decompose_query(state["question"])
            # Batch discrete sub-goals up to 5 concurrent queries on initial turn
            search_targets = [g for g in sub_goals if g not in visited_queries][:5]
            if not search_targets:
                search_targets = [state["question"]]
        else:
            # Turn > 0: Anti-stagnation selection
            search_targets = []
            curr = state.get("current_query", "").strip()
            if curr and curr != state["question"] and curr not in visited_queries:
                search_targets.append(curr)
            elif pending_links and len(visited_links) < state["max_hops"]:
                next_l = pending_links.pop(0)
                visited_links.append(next_l)
                search_targets.append(next_l)
            else:
                unvisited = [g for g in sub_goals if g not in visited_queries]
                if unvisited:
                    search_targets.append(unvisited[0])

        current_turn_primary: List[SearchResult] = []
        new_candidates: List[Tuple[str, str]] = []

        for target_query in search_targets:
            if target_query in visited_queries:
                continue
            visited_queries.append(target_query)
            raw_results = self.storage.search(target_query, top_k=4)
            qualified = self.prune_by_score_cliff(raw_results, query=target_query)
            for r in qualified:
                if r.id not in visited_ids and all(p.id != r.id for p in current_turn_primary):
                    current_turn_primary.append(r)
                    # Extract candidate cross-references if in broad inquiry scope
                    for link in r.wikilinks:
                        link_clean = " ".join(link.split()).strip()
                        if re.match(r"^§\s*\d+(?:\.\d+)?$", link_clean):
                            continue
                        if self._is_in_inquiry_scope(link_clean, state["question"]):
                            if link_clean not in visited_links and link_clean not in pending_links:
                                snippet = r.breadcrumbs or ""
                                for sentence in r.content.split(". "):
                                    if link in sentence or link_clean in sentence:
                                        snippet = sentence[:200].strip()
                                        break
                                new_candidates.append((link_clean, snippet))

        # Intra-Document Sibling Expansion (SPEC-006)
        sibling_candidates: List[SearchResult] = []
        sibling_scores: Dict[str, float] = {}
        if getattr(self.vault.config, "sibling_traversal", True) and current_turn_primary:
            cand_sibling_ids = []
            seen_cand_ids = set(visited_ids) | {p.id for p in current_turn_primary}
            for p in current_turn_primary:
                # Only expand siblings for primary hits with strong semantic relevance
                if p.score >= 0.58 and "#" in p.id:
                    file_part, chunk_str = p.id.rsplit("#", 1)
                    if chunk_str.isdigit():
                        num = int(chunk_str)
                        for off in (-2, -1, 1, 2):
                            cand_id = f"{file_part}#{num + off}"
                            if num + off > 0 and cand_id not in seen_cand_ids:
                                seen_cand_ids.add(cand_id)
                                cand_sibling_ids.append(cand_id)

            if cand_sibling_ids:
                fetched_siblings = self.storage.get_chunks_by_ids(cand_sibling_ids)
                if fetched_siblings:
                    cand_pairs = [
                        (s.id, f"{s.breadcrumbs}: {s.content[:300]}")
                        for s in fetched_siblings
                    ]
                    scored_tuples = self.score_candidate_links(state["question"], cand_pairs)
                    threshold = float(getattr(self.vault.config, "sibling_pruning_threshold", 0.58))
                    max_per_hit = int(getattr(self.vault.config, "sibling_max_per_hit", 2))

                    fetched_map = {s.id: s for s in fetched_siblings}
                    scored_qualified = [
                        (cid, score) for cid, score in scored_tuples if score >= threshold and cid in fetched_map
                    ]
                    # Take top siblings
                    for cid, score in scored_qualified[: len(current_turn_primary) * max_per_hit]:
                        sibling_candidates.append(fetched_map[cid])
                        sibling_scores[cid] = score

        # Context Governor & Greedy Token Budget Packing
        governor_enabled = getattr(self.vault.config, "context_governor", True)
        budget_tokens = self.get_context_governor_budget() if governor_enabled else 999999
        current_tokens = sum(int(len(e["content"]) / 4) for e in evidence_list)
        pruned_high_rel: List[Dict[str, Any]] = []
        new_chunks_count = 0
        governor_alert = state.get("governor_alert")

        # 1. Pack primary chunks
        for r in current_turn_primary:
            tok = int(len(r.content) / 4)
            if current_tokens + tok <= budget_tokens:
                visited_ids.add(r.id)
                new_sources.add(r.file_path)
                new_chunks_count += 1
                evidence_list.append(
                    {
                        "id": r.id,
                        "file_path": r.file_path,
                        "breadcrumbs": r.breadcrumbs,
                        "content": r.content,
                        "wikilinks": r.wikilinks,
                        "score": r.score,
                    }
                )
                current_tokens += tok
            else:
                pruned_high_rel.append(
                    {
                        "id": r.id,
                        "title": r.section_title or r.breadcrumbs,
                        "score": 1.0,
                        "tokens": tok,
                    }
                )

        # 2. Pack sibling chunks (sorted descending by score)
        sibling_candidates.sort(key=lambda s: sibling_scores.get(s.id, 0.0), reverse=True)
        for s in sibling_candidates:
            if s.id in visited_ids:
                continue
            tok = int(len(s.content) / 4)
            score = sibling_scores.get(s.id, 0.0)
            if current_tokens + tok <= budget_tokens:
                visited_ids.add(s.id)
                new_sources.add(s.file_path)
                new_chunks_count += 1
                evidence_list.append(
                    {
                        "id": s.id,
                        "file_path": s.file_path,
                        "breadcrumbs": s.breadcrumbs,
                        "content": s.content,
                        "wikilinks": s.wikilinks,
                        "score": score,
                    }
                )
                current_tokens += tok
            else:
                if score >= 0.50:
                    pruned_high_rel.append(
                        {
                            "id": s.id,
                            "title": s.section_title or s.breadcrumbs,
                            "score": score,
                            "tokens": tok,
                        }
                    )

        if pruned_high_rel and not governor_alert:
            window_size = self.get_model_context_window()
            model_name = getattr(self.llm, "model", "active model")
            pruned_tok_sum = sum(p["tokens"] for p in pruned_high_rel)
            unaddressed_goals = [g for g in sub_goals if g not in visited_queries]
            alert_lines = [
                "[CONTEXT_GOVERNOR_ALERT]",
                f"Context budget reached: {current_tokens:,} tokens (allocated for {window_size:,}-token model '{model_name}').",
                f"Pruned {len(pruned_high_rel)} tail / lower-ranked clauses (~{pruned_tok_sum:,} tokens) to prevent context degradation:",
            ]
            for p in pruned_high_rel[:4]:
                alert_lines.append(f"  - {p['id']} (Score: {p['score']:.2f}): {p['title'][:55]}")
            if unaddressed_goals:
                alert_lines.append(f"Deferred {len(unaddressed_goals)} remaining sub-goal(s) due to context capacity:")
                for g in unaddressed_goals[:3]:
                    alert_lines.append(f"  - {g}")
                if len(unaddressed_goals) > 3:
                    alert_lines.append(f"  ... and {len(unaddressed_goals) - 3} more")
            alert_lines.append("Actionable Guidance: Sizing up context window is empirically justified.")
            governor_alert = "\n".join(alert_lines)
            thought_history.append(governor_alert)

        # System 1 Link Pruning & Priority Queuing
        if new_candidates:
            if getattr(self.vault.config, "link_pruning", True):
                threshold = float(getattr(self.vault.config, "link_pruning_threshold", 0.40))
                max_queue = int(getattr(self.vault.config, "link_pruning_max_queue", 10))
                scored_candidates = self.score_candidate_links(state["question"], new_candidates)

                kept = [l for l, sim in scored_candidates if sim >= threshold]
                pruned_count = len(scored_candidates) - len(kept)

                for l in kept:
                    if l not in pending_links:
                        pending_links.append(l)

                if len(pending_links) > max_queue:
                    pending_links = pending_links[:max_queue]

                if pruned_count > 0:
                    thought_history.append(
                        f"System 1 Link Pruner: Evaluated {len(scored_candidates)} candidate link(s); pruned {pruned_count} below threshold ({threshold:.2f})."
                    )
            else:
                for l, _ in new_candidates:
                    if l not in pending_links:
                        pending_links.append(l)

        # Targeted Topic Re-Decomposition Guard (Depth 1):
        # If query yielded zero new chunks and hasn't been re-decomposed yet, drill down!
        if new_chunks_count == 0 and search_targets and search_targets[0] not in recomposed_topics:
            target_query = search_targets[0]
            if len(target_query.split()) >= 2 and not target_query.lower().startswith("sec"):
                recomposed_topics.append(target_query)
                refined = self.refine_stagnated_topic(target_query, state["question"])
                if refined:
                    # Prepend refined queries to sub_goals for immediate follow-up
                    sub_goals = [q for q in refined if q not in visited_queries] + sub_goals
                    thought_history.append(
                        f"Topic '{target_query}' stagnated (0 new chunks). Re-decomposed into: {', '.join(refined)}."
                    )

        targets_summary = ", ".join(search_targets) if search_targets else "None (queue empty)"
        thought = f"Iteration {state['iteration'] + 1}: Retrieved {len(evidence_list)} evidence blocks across queries: {targets_summary}."
        thought_history.append(thought)

        return {
            "evidence": evidence_list,
            "sources": sorted(list(new_sources)),
            "pending_links": pending_links,
            "visited_links": visited_links,
            "visited_queries": visited_queries,
            "sub_goals": sub_goals,
            "sub_questions": sub_questions,
            "recomposed_topics": recomposed_topics,
            "thought_history": thought_history,
            "governor_alert": governor_alert,
        }

    def _evaluate_node(self, state: AgentState) -> Dict[str, Any]:
        iteration = state["iteration"] + 1
        max_iter = min(state["max_iterations"], 5)  # Hard guardrail ceiling

        # If reached iteration limit or no evidence found, force synthesis
        if iteration >= max_iter or not state["evidence"]:
            thought = f"Evaluation: Turn ceiling ({iteration}/{max_iter}) reached. Proceeding to synthesis."
            return {
                "iteration": iteration,
                "is_sufficient": True,
                "thought_history": state["thought_history"] + [thought],
            }

        visited_queries = set(state.get("visited_queries", []))
        sub_goals = state.get("sub_goals", [])
        pending_links = state.get("pending_links", [])

        # Check if there are still unvisited sub-goals or pending links
        unvisited_goals = [g for g in sub_goals if g not in visited_queries]

        # Ask LLM if evidence is sufficient (dynamic budget based on endpoint)
        is_local = "localhost" in self.llm.endpoint or "127.0.0.1" in self.llm.endpoint
        eval_char_limit = 12000 if is_local else 60000
        eval_snippets = []
        eval_chars = 0
        for e in state["evidence"]:
            snip = f"Source: {e['breadcrumbs']} ({e['file_path']})\n{e['content'].strip()}"
            if eval_chars + len(snip) > eval_char_limit and eval_snippets:
                break
            eval_snippets.append(snip)
            eval_chars += len(snip) + 2

        context_snippets = "\n---\n".join(eval_snippets)

        sys_prompt = (
            "You are an evidence sufficiency evaluator. Respond ONLY in valid JSON.\n"
            "Analyze whether the collected evidence directly and completely answers the user's question.\n"
            "Format: {\"sufficient\": true/false, \"thought\": \"...\", \"next_query\": \"...\", \"confidence\": 0.0-1.0}"
        )
        user_prompt = (
            f"Question: {state['question']}\n\n"
            f"Collected Evidence:\n{context_snippets}\n\n"
            "Is this evidence sufficient to answer the question?"
        )

        response = self.llm.chat(sys_prompt, user_prompt)
        sufficient = False
        next_query = ""
        thought = f"Evaluation: Evidence reviewed ({iteration}/{max_iter})."
        confidence = 0.5

        if response:
            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    sufficient = bool(parsed.get("sufficient", False))
                    thought = parsed.get("thought", thought)
                    if parsed.get("next_query"):
                        next_query = parsed["next_query"].strip()
                    confidence = float(parsed.get("confidence", 0.5))
            except Exception:
                pass

        # If LLM said sufficient but primary decomposed sub-goals remain unsearched, do not stop prematurely
        # UNLESS the Context Governor has already saturated the model's budget
        has_governor_saturation = bool(state.get("governor_alert"))
        if sufficient and unvisited_goals:
            if not has_governor_saturation:
                sufficient = False

        # Deterministic Anti-Stagnation Fallback: Never repeat the same query!
        if not sufficient:
            if has_governor_saturation:
                sufficient = True
                thought = "Evaluation: Context capacity reached. Halting further retrieval to synthesize from admitted evidence."
            elif not next_query or next_query in visited_queries or next_query == state["question"]:
                if unvisited_goals:
                    next_query = unvisited_goals[0]
                elif pending_links:
                    next_query = pending_links[0]
                else:
                    sufficient = True
                    thought = "Evaluation: All sub-goals and references explored. Proceeding to synthesis."

        return {
            "iteration": iteration,
            "is_sufficient": sufficient,
            "current_query": next_query,
            "confidence": confidence,
            "thought_history": state["thought_history"] + [f"Eval (turn {iteration}): {thought}"],
        }

    def _route_evaluation(self, state: AgentState) -> str:
        if state["is_sufficient"]:
            if state.get("evidence_only", False):
                return "end"
            return "synthesize"
        return "retrieve"

    def _record_gap_if_needed(self, question: str, confidence: float, evidence_count: int, answer: str) -> None:
        try:
            from alcuin.storage.gaps import KnowledgeGapTracker
            tracker = KnowledgeGapTracker(self.vault)
            tracker.record_query(query=question, confidence=confidence, evidence_count=evidence_count, answer=answer)
        except Exception:
            pass

    def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        if not state["evidence"]:
            answer = "No relevant documents or evidence found in vault."
            confidence = 0.0
            self._record_gap_if_needed(state["question"], confidence, 0, answer)
            return {
                "final_answer": answer,
                "confidence": confidence,
            }

        # Budget evidence to fit model context window (coupled to calibrated Context Governor)
        max_evidence_chars = self.get_context_governor_budget() * 4
        budgeted_chunks = []
        total_chars = 0

        for e in state["evidence"]:
            chunk_repr = f"[{e['breadcrumbs']} ({e['file_path']})]\n{e['content'].strip()}"
            if total_chars + len(chunk_repr) > max_evidence_chars and budgeted_chunks:
                break
            budgeted_chunks.append(chunk_repr)
            total_chars += len(chunk_repr) + 2

        evidence_text = "\n\n---\n\n".join(budgeted_chunks)

        sys_prompt = (
            "You are Alcuin, an autonomous knowledge assistant. Answer the user's question directly and concisely "
            "based strictly on the provided evidence.\n"
            "If multiple questions or core tasks are identified, address each in a separate numbered response.\n"
            "State the exact numerical deadline, threshold, or entity directly before providing explanatory context.\n"
            "When citing tables, schedules, deadlines, or fine structures, transcribe the row values and numbers verbatim.\n"
            "If the evidence does not contain the answer, state clearly and concisely that the information is not found."
        )

        sub_questions = state.get("sub_questions", [])
        if sub_questions and len(sub_questions) >= 2:
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(sub_questions)])
            user_prompt = (
                f"User Inquiry:\n{state['question']}\n\n"
                f"Questions to Address:\n{questions_text}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Provide a direct, numbered response explicitly answering each Question using the evidence:\n"
            )
        else:
            user_prompt = (
                f"Question: {state['question']}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Direct Answer:"
            )

        thought_history = list(state.get("thought_history", []))

        # Invoke synthesis with truncation metadata detection
        try:
            res = self.llm.chat(sys_prompt, user_prompt, return_metadata=True)
        except TypeError:
            res = self.llm.chat(sys_prompt, user_prompt)
        if isinstance(res, tuple):
            answer, meta = res
        else:
            answer, meta = res, {}

        truncation_alert = None
        if meta.get("thinking"):
            thought_history.append(f"[LATENT_THOUGHT_TRACE]\n{meta['thinking']}")

        if meta.get("truncated"):
            finish_reason = meta.get("finish_reason") or "length"
            truncation_alert = (
                "[GENERATION_TRUNCATION_ALERT]\n"
                f"Warning: Answer generation was halted because the model reached its context token ceiling (finish_reason: {finish_reason}).\n"
                "The synthesized answer is incomplete.\n"
                "Actionable Guidance: Sizing up the model context window or increasing context headroom is empirically justified."
            )
            thought_history.append(truncation_alert)
            if answer:
                answer = f"{answer}\n\n{truncation_alert}"
            else:
                answer = f"The model ran out of tokens before generating an answer.\n\n{truncation_alert}"

        if not answer:
            # Concise fallback if model endpoint is unreachable
            top_evidence = state["evidence"][0]["content"].strip()[:400] if state["evidence"] else "No evidence."
            answer = (
                "Could not reach LLM for synthesis. Relevant excerpt:\n\n"
                f"> {top_evidence}..."
            )

        ans_lower = answer.lower()
        if (
            "not found" in ans_lower
            or "no relevant documents" in ans_lower
            or "information is not found" in ans_lower
            or "not contained in the vault" in ans_lower
            or "not contained in the provided evidence" in ans_lower
        ):
            confidence = 0.0
        else:
            confidence = max(state.get("confidence", 0.5), 0.7 if state["evidence"] else 0.0)

        self._record_gap_if_needed(state["question"], confidence, len(state["evidence"]), answer)
        alert = state.get("governor_alert")
        if alert and alert not in answer:
            answer = f"{answer}\n\n{alert}"
        return {
            "final_answer": answer,
            "confidence": confidence,
            "governor_alert": alert,
            "truncation_alert": truncation_alert,
            "thought_history": thought_history,
        }

    def research(self, question: str, evidence_only: bool = False) -> AgentState:
        """Execute full research cycle synchronously."""
        initial_state: AgentState = {
            "question": question,
            "iteration": 0,
            "max_iterations": self.vault.config.max_iterations,
            "max_hops": self.vault.config.max_hops,
            "current_query": question,
            "visited_links": [],
            "pending_links": [],
            "sub_goals": [],
            "sub_questions": [],
            "visited_queries": [],
            "recomposed_topics": [],
            "evidence": [],
            "thought_history": [],
            "is_sufficient": False,
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
            "governor_alert": None,
            "evidence_only": evidence_only,
        }

        result = self.graph.invoke(initial_state)
        return result

    def stream_research(self, question: str, evidence_only: bool = False) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Stream intermediate state changes for live CLI and SSE Web UI."""
        initial_state: AgentState = {
            "question": question,
            "iteration": 0,
            "max_iterations": self.vault.config.max_iterations,
            "max_hops": self.vault.config.max_hops,
            "current_query": question,
            "visited_links": [],
            "pending_links": [],
            "sub_goals": [],
            "sub_questions": [],
            "visited_queries": [],
            "recomposed_topics": [],
            "evidence": [],
            "thought_history": [],
            "is_sufficient": False,
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
            "governor_alert": None,
            "evidence_only": evidence_only,
        }

        for step in self.graph.stream(initial_state):
            for node_name, state_update in step.items():
                yield node_name, state_update
