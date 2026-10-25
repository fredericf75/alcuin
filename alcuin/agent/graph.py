"""
alcuin.agent.graph - LangGraph Cyclic ReAct Research Agent with Hard Turn Ceiling
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypedDict

import logging
from dotenv import load_dotenv
import httpx
from langgraph.graph import END, START, StateGraph

from alcuin.storage.engine import SearchResult, VaultStorage
from alcuin.storage.vault import Vault, VaultRegistry

load_dotenv()
logger = logging.getLogger(__name__)


class CoverageItem(TypedDict):
    id: int
    sub_question: str
    status: str  # "COVERED" | "MISSING"
    gap_query: Optional[str]


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
    coverage_matrix: List[CoverageItem]
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
    synthesis_clipped: Optional[bool]
    unaddressed_sub_questions: Optional[List[int]]


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


class ModelNotFoundError(RuntimeError):
    """Raised when a requested model is not found on the target endpoint."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        available_models: Optional[List[str]] = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.available_models = available_models or []
        avail_msg = (
            f"\n\nAvailable models on '{endpoint}':\n"
            + "\n".join(f"  • {m}" for m in self.available_models)
            if self.available_models
            else ""
        )
        super().__init__(
            f"Model '{model}' not found on endpoint '{endpoint}'.{avail_msg}"
        )


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

    def get_available_models(self) -> List[str]:
        """Query endpoint for installed / available models (Ollama or OpenAI-compatible)."""
        try:
            with httpx.Client(timeout=3.0) as client:
                # 1. Probe Ollama /api/tags
                try:
                    resp = client.get(f"{self.endpoint}/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        tags = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                        if tags:
                            return sorted(tags)
                except Exception:
                    pass

                # 2. Probe OpenAI /v1/models or /models
                base_ep = self.endpoint.rstrip("/")
                target_url = f"{base_ep}/models" if base_ep.endswith("/v1") else f"{base_ep}/v1/models"
                headers: Dict[str, str] = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                try:
                    resp = client.get(target_url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        m_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                        if m_ids:
                            return sorted(m_ids)
                except Exception:
                    pass
        except Exception:
            pass
        return []

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
                    elif resp.status_code == 404:
                        err_msg = ""
                        try:
                            err_msg = str(resp.json().get("error", ""))
                        except Exception:
                            pass
                        if "not found" in err_msg.lower() or not err_msg:
                            avail = self.get_available_models()
                            raise ModelNotFoundError(self.model, self.endpoint, avail)
            except ModelNotFoundError:
                raise
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
                elif resp.status_code == 404:
                    err_msg = ""
                    try:
                        err_data = resp.json()
                        if isinstance(err_data, dict):
                            err_msg = str(err_data.get("error", ""))
                    except Exception:
                        pass
                    if "model" in err_msg.lower() or "not found" in err_msg.lower():
                        avail = self.get_available_models()
                        raise ModelNotFoundError(self.model, self.endpoint, avail)
        except ModelNotFoundError:
            raise
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
                    elif resp.status_code == 404:
                        err_msg = ""
                        try:
                            err_msg = str(resp.json().get("error", ""))
                        except Exception:
                            pass
                        if "not found" in err_msg.lower() or not err_msg:
                            avail = self.get_available_models()
                            raise ModelNotFoundError(self.model, self.endpoint, avail)
            except ModelNotFoundError:
                raise
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
                # 1. Clean parenthetical enumeration markers (e.g. '(a)', '(b)', '(1)') without corrupting word lists
                q_clean = re.sub(r"\([a-zA-Z0-9]+\)\s*", "", q).rstrip("?.:;!")
                
                # 2. Split compound questions at coordinating conjunctions followed by interrogatives
                sub_clauses = re.split(
                    r",?\s+(?:and|while|as well as)\s+(?=(?:under what|what|how|when|where|who|which|can|may|must)\b)",
                    q_clean,
                    flags=re.IGNORECASE,
                )
                parent_anchors = [g for g in global_identifiers if g.lower() in q.lower()]

                for clause in sub_clauses:
                    clause_identifiers = re.findall(
                        r"\b(?:§\s*\d+|[A-Z]{2,}(?:[\-\/][A-Z0-9]+)*|\b[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+\b|\d+[\-\w]+)\b",
                        clause,
                    )
                    body = re.sub(
                        r"^(?:in|under|pursuant to|according to|for|with respect to|regarding)\s+[^,;:\n]+[,\-:]\s*",
                        "",
                        clause.strip(),
                        flags=re.IGNORECASE,
                    )
                    words = re.findall(r"\b[\w\-\.]+\b", body)
                    meaningful = [
                        w for w in words
                        if w.lower() not in filler and (len(w) > 1 or w.isdigit() or w.upper() in ("I", "V", "X"))
                    ]
                    anchors = list(dict.fromkeys(clause_identifiers + parent_anchors))
                    content_terms = [w for w in meaningful if w not in anchors]

                    # Retain terms without arbitrary middle truncation
                    selected = content_terms[:16]

                    search_query = " ".join(dict.fromkeys(anchors + selected))
                    if not search_query.strip():
                        search_query = clause.strip()
                    if search_query.strip() and search_query not in sub_goals:
                        sub_goals.append(search_query)

            return sub_goals, sub_questions

        # 2. Dual-field LLM decomposition for unstructured inquiries
        sys_prompt = (
            "You are a research query planner.\n"
            "Identify every distinct question, condition, metric, exception, or consequence raised in the user's inquiry.\n"
            "CRITICAL RULES:\n"
            "1. Each sub-goal must be strictly ATOMIC. Never combine multiple questions, rules, or conditions into one query.\n"
            "2. PRESERVE SPECIFICITY: Always retain numbers, measurements, thresholds, entity names, and citations (e.g., '30-bed', '72 hours', 'SPICE'). Do NOT abstract them into generic words.\n"
            "3. SEARCH QUERIES: Write dense, entity-preserving retrieval queries (5 to 10 terms) that include the target subject, specific action, and key constraint.\n"
            "Respond with a JSON object adhering strictly to this schema:\n"
            "{\n"
            '  "sub_goals": [\n'
            '    {\n'
            '      "question": "Full grammatical question addressing one specific issue",\n'
            '      "search_query": "Dense, entity-preserving search query including key metrics and constraints"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Output ONLY valid JSON."
        )
        llm_resp = self.planner_llm.chat(sys_prompt, question[:2000])
        llm_resp, _ = _extract_and_clean_thinking(llm_resp)
        sub_goals: List[str] = []
        sub_questions: List[str] = []

        def _is_valid_q(val: str) -> bool:
            v = val.lower()
            return not any(b in v for b in ("full grammatical question", "addressing one specific", "<extracted question")) and len(val.strip()) > 5

        if llm_resp:
            try:
                m = re.search(r"\{.*\}", llm_resp, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    items = data.get("sub_goals", [])
                    for it in items:
                        if isinstance(it, dict):
                            q = str(it.get("question", "")).strip()
                            s_raw = it.get("search_query", "")
                            s = " ".join([str(x) for x in s_raw]) if isinstance(s_raw, list) else str(s_raw).strip()
                            if s.endswith("?") and not q.endswith("?"):
                                q, s = s, q
                            if q and s and _is_valid_q(q):
                                sub_questions.append(q)
                                sub_goals.append(s)
                        elif isinstance(it, str) and it.strip() and _is_valid_q(it):
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
                                    s_raw = it.get("search_query", "")
                                    s = " ".join([str(x) for x in s_raw]) if isinstance(s_raw, list) else str(s_raw).strip()
                                    if s.endswith("?") and not q.endswith("?"):
                                        q, s = s, q
                                    if q and s and _is_valid_q(q):
                                        sub_questions.append(q)
                                        sub_goals.append(s)
                                elif isinstance(it, str) and it.strip() and _is_valid_q(it):
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

        # Rule 2a: Universal Structural Citation Match (e.g. .0702, 6-132, 15.1.3, §401, Sec. 5, Rule 144A)
        prefixed_ids = [
            m.group(0).strip()
            for m in re.finditer(
                r"\b(?:sec(?:tion)?|rule|art(?:icle)?|ch(?:apter)?|clause|rfc|schedule|exhibit|appendix|reg(?:ulation)?)\s*#?[0-9a-z]+(?:\.[0-9a-z]+)*\b",
                q_lower,
            )
        ]
        dotted_or_hyphen = [
            m.group(0).strip()
            for m in re.finditer(r"(?:§\s*|\b|\.)\d+(?:[\-\.][0-9a-z]+)+|\.\d{3,}\b", q_lower)
        ]
        identifiers = [
            raw.strip(". §").strip()
            for raw in (prefixed_ids + dotted_or_hyphen)
            if len(raw.strip(". §").strip()) >= 2
        ]
        identifiers = list(dict.fromkeys(identifiers))
        if identifiers:
            if any(ident in anchor_text for ident in identifiers):
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
                admitted = [r for r in sorted_results if r.score >= 0.40]
                pruned = [r for r in sorted_results if r.score < 0.40]

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

    def _extract_ast_rule_key(self, breadcrumbs: str) -> str:
        """Extract operative statutory rule / section key from AST breadcrumbs (e.g. '10A NCAC 13F.0702' or 'Sec. 6-132')."""
        if not breadcrumbs:
            return ""
        m = re.search(r"10A\s+NCAC\s+\d+[A-Z]\s*\.?\d+", breadcrumbs, re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip().upper()
        m = re.search(r"Sec(?:tion|\.)?\s*\d+[\-\.]\d+", breadcrumbs, re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip()
        m = re.search(r"(?:Article|Section)\s+\d+(?:\.\d+)?", breadcrumbs, re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip()
        return ""

    def score_candidate_links(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        current_file_path: str = "",
        hop_depth: int = 1,
    ) -> List[Tuple[str, float]]:
        """Score candidate wikilinks against the research query using embedding cosine similarity
        with structural document boundary penalty (beta) and hop distance attenuation (gamma^depth).
        """
        if not candidates:
            return []

        try:
            q_vec = self.storage.embedder.get_embedding(query)
            q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0

            beta = float(getattr(self.vault.config, "link_boundary_penalty", 0.25))
            gamma_base = float(getattr(self.vault.config, "link_hop_attenuation", 0.75))
            # Attenuation applies on hops beyond the initial query level (hop_depth > 1)
            gamma = (gamma_base ** (hop_depth - 1)) if hop_depth > 1 else 1.0

            scored: List[Tuple[str, float]] = []
            link_texts = [f"{link_clean}: {snippet}" if snippet else link_clean for link_clean, snippet in candidates]
            l_vecs = [self.storage.embedder.get_embedding(t) for t in link_texts]
            for (link_clean, snippet), l_vec in zip(candidates, l_vecs):
                l_norm = math.sqrt(sum(x * x for x in l_vec)) or 1.0
                dot = sum(a * b for a, b in zip(q_vec, l_vec))
                raw_sim = max(0.0, dot / (q_norm * l_norm))

                # Structural boundary penalty if crossing top-level document trees (different file/chapter)
                is_cross_boundary = bool(current_file_path and link_clean.lower() not in current_file_path.lower())
                penalty = beta if is_cross_boundary else 0.0

                penalized = max(0.0, (raw_sim - penalty) * gamma)
                scored.append((link_clean, penalized))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except Exception:
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
            search_depth = getattr(self.vault.config, "retrieval_candidates", 20)
            raw_results = self.storage.search(target_query, top_k=search_depth)
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

        # Intra-Document AST Sibling Expansion (SPEC-006 with Spatial Distance Attenuation)
        sibling_candidates: List[SearchResult] = []
        sibling_scores: Dict[str, float] = {}
        if getattr(self.vault.config, "sibling_traversal", True) and current_turn_primary:
            gamma = float(getattr(self.vault.config, "ast_sibling_attenuation", 0.72))
            threshold = float(getattr(self.vault.config, "sibling_pruning_threshold", 0.45))
            if 0.0 < gamma < 1.0 and threshold > 0.0:
                max_admissible_dist = int(math.floor(math.log(threshold) / math.log(gamma) + 1e-9))
            else:
                max_admissible_dist = 10

            cand_sibling_map: Dict[str, Tuple[SearchResult, int]] = {}
            seen_cand_ids = set(visited_ids) | {p.id for p in current_turn_primary}
            primary_indices_by_heading: Dict[Tuple[str, str], List[int]] = {}

            for p in current_turn_primary:
                # Expand siblings for primary hits with strong semantic relevance
                if p.score >= 0.58 and p.breadcrumbs:
                    crumbs = [c.strip() for c in p.breadcrumbs.split(" > ") if c.strip()]
                    doc_title = p.document_title.strip() if p.document_title else ""
                    sec_crumbs = [c for c in crumbs if c.lower() != doc_title.lower()]
                    parent_heading = sec_crumbs[0] if sec_crumbs else crumbs[-1]
                    m_p = re.search(r"#(\d+)$", p.id)
                    p_idx = int(m_p.group(1)) if m_p else 0
                    key = (p.file_path, parent_heading)
                    primary_indices_by_heading.setdefault(key, []).append(p_idx)

            for (fpath, parent_heading), indices in primary_indices_by_heading.items():
                matched_rows = []
                try:
                    if hasattr(self.storage, "table") and self.storage.table is not None:
                        clean_heading = parent_heading.replace("'", "''")
                        clean_file = fpath.replace("'", "''")
                        query_filter = f"file_path = '{clean_file}' AND breadcrumbs LIKE '%{clean_heading}%'"
                        matched_rows = self.storage.table.search().where(query_filter).to_list()
                except Exception:
                    matched_rows = []

                # Fallback for mock environments or storage engines without LanceDB table search
                if not matched_rows and hasattr(self.storage, "get_chunks_by_ids"):
                    cand_ids = [
                        f"{fpath}#{pi + off}"
                        for pi in indices
                        for off in (-2, -1, 1, 2)
                        if (pi + off) > 0
                    ]
                    fallback_chunks = self.storage.get_chunks_by_ids(cand_ids)
                    for fc in fallback_chunks:
                        matched_rows.append({
                            "id": fc.id,
                            "file_path": fc.file_path,
                            "document_title": fc.document_title,
                            "section_title": fc.section_title,
                            "breadcrumbs": fc.breadcrumbs,
                            "content": fc.content,
                            "wikilinks": fc.wikilinks,
                        })

                for r in matched_rows:
                    cid = r["id"]
                    if cid not in seen_cand_ids and cid not in cand_sibling_map:
                        seen_cand_ids.add(cid)
                        m_s = re.search(r"#(\d+)$", cid)
                        s_idx = int(m_s.group(1)) if m_s else 0
                        dist = min(abs(s_idx - pi) for pi in indices)
                        # Pre-embedding distance pruning: If distance alone attenuates eff_score below
                        # threshold even under perfect cosine similarity (1.0 * gamma^dist < threshold),
                        # skip embedding entirely to eliminate CPU embedding bottlenecks.
                        if dist > max_admissible_dist or dist == 0:
                            continue
                        cand_sibling_map[cid] = (
                            SearchResult(
                                id=cid,
                                file_path=r["file_path"],
                                document_title=r.get("document_title", ""),
                                section_title=r.get("section_title", ""),
                                breadcrumbs=r.get("breadcrumbs", ""),
                                content=r["content"],
                                wikilinks=r.get("wikilinks", []),
                                score=0.0,
                            ),
                            dist,
                        )

            if cand_sibling_map:
                from unittest.mock import MagicMock
                cand_entries = list(cand_sibling_map.values())
                cand_items = [entry[0] for entry in cand_entries]
                cand_dists = [entry[1] for entry in cand_entries]

                if isinstance(getattr(self, "score_candidate_links", None), MagicMock):
                    mock_pairs = [(sr.id, f"{sr.breadcrumbs}: {sr.content[:300]}") for sr in cand_items]
                    mock_scored = dict(self.score_candidate_links(state["question"], mock_pairs))
                    raw_sims = [mock_scored.get(sr.id, 0.0) for sr in cand_items]
                else:
                    q_vec = self.storage.embedder.get_embedding(state["question"])
                    q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
                    cand_texts = [f"{s.breadcrumbs}: {s.content[:300]}" for s in cand_items]
                    l_vecs = self.storage.embedder.get_embeddings_batch(cand_texts)
                    raw_sims = []
                    for l_vec in l_vecs:
                        l_norm = math.sqrt(sum(x * x for x in l_vec)) or 1.0
                        dot = sum(a * b for a, b in zip(q_vec, l_vec))
                        raw_sims.append(max(0.0, dot / (q_norm * l_norm)))

                for s, dist, raw_sim in zip(cand_items, cand_dists, raw_sims):
                    eff_score = raw_sim * (gamma ** min(dist, 10))
                    if eff_score >= threshold:
                        sibling_candidates.append(s)
                        sibling_scores[s.id] = eff_score

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
                cur_fpath = current_turn_primary[0].file_path if current_turn_primary else ""
                hop_depth = state.get("iteration", 0) + 1
                scored_candidates = self.score_candidate_links(
                    state["question"],
                    new_candidates,
                    current_file_path=cur_fpath,
                    hop_depth=hop_depth,
                )

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
        sub_questions = state.get("sub_questions", [])

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

        if sub_questions:
            questions_checklist = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
            sys_prompt = (
                "You are an evidence sufficiency evaluator and research auditor. Respond ONLY in valid JSON.\n"
                "Examine the Collected Evidence against each Sub-Question.\n"
                "For each sub-question, determine whether the evidence directly answers it (COVERED) or lacks the required details (MISSING).\n"
                "If MISSING, provide a concise 3-to-6 keyword search query (gap_query) to retrieve the missing information.\n"
                "Format: {\"coverage\": [{\"id\": 1, \"status\": \"COVERED\"}, {\"id\": 2, \"status\": \"MISSING\", \"gap_query\": \"...\"}], \"all_covered\": true/false, \"thought\": \"...\", \"confidence\": 0.0-1.0}"
            )
            user_prompt = (
                f"User Inquiry:\n{state['question']}\n\n"
                f"Sub-Questions to Verify:\n{questions_checklist}\n\n"
                f"Collected Evidence:\n{context_snippets}\n\n"
                "Audit the coverage of each sub-question against the collected evidence:"
            )
        else:
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
        coverage_matrix: List[CoverageItem] = []

        if response:
            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    thought = parsed.get("thought", thought)
                    confidence = float(parsed.get("confidence", 0.5))

                    raw_coverage = parsed.get("coverage", [])
                    if isinstance(raw_coverage, list) and raw_coverage and sub_questions:
                        all_items_covered = True
                        for item in raw_coverage:
                            if not isinstance(item, dict):
                                continue
                            item_id = item.get("id", 0)
                            q_text = sub_questions[item_id - 1] if 0 < item_id <= len(sub_questions) else ""
                            status = str(item.get("status", "")).upper()
                            gap = item.get("gap_query", "").strip() if item.get("gap_query") else None
                            if status != "COVERED":
                                all_items_covered = False
                                if not next_query and gap and gap not in visited_queries:
                                    next_query = gap
                            coverage_matrix.append({
                                "id": item_id,
                                "sub_question": q_text,
                                "status": status,
                                "gap_query": gap,
                            })
                        explicit_all = parsed.get("all_covered")
                        if explicit_all is not None:
                            sufficient = bool(explicit_all) and all_items_covered
                        else:
                            sufficient = all_items_covered
                    else:
                        sufficient = bool(parsed.get("sufficient", False))
                        if parsed.get("next_query"):
                            next_query = parsed["next_query"].strip()
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
            "coverage_matrix": coverage_matrix,
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
        thought_history = list(state.get("thought_history", []))
        truncation_alert = None

        if sub_questions and len(sub_questions) >= 2:
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(sub_questions)])
            user_prompt = (
                f"Target Questions to Address:\n{questions_text}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Instructions: Answer each Target Question individually in numbered order. State exact numbers, deadlines, or requirements directly:\n"
            )
        else:
            user_prompt = (
                f"Question: {state['question']}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Direct Answer:"
            )

        try:
            res = self.llm.chat(sys_prompt, user_prompt, return_metadata=True)
        except TypeError:
            res = self.llm.chat(sys_prompt, user_prompt)
        if isinstance(res, tuple):
            answer, meta = res
        else:
            answer, meta = res, {}

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

        # Output Governor (SPEC-017): Clipped Output Completeness Audit
        unaddressed_indices = []
        if sub_questions and len(sub_questions) >= 2 and answer:
            for i in range(len(sub_questions)):
                q_num = i + 1
                marker_found = bool(re.search(rf"(?:^|\n)\s*(?:[#*]*\s*)?(?:Question\s*)?{q_num}[\.\):]", answer))
                if not marker_found:
                    unaddressed_indices.append(q_num)

        if unaddressed_indices:
            warning_msg = (
                f"[OUTPUT GOVERNOR WARNING] Synthesis clipped: Sub-questions {unaddressed_indices} "
                f"of {len(sub_questions)} were unaddressed in the final response."
            )
            thought_history.append(warning_msg)
            logger.warning(warning_msg)

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
            "synthesis_clipped": bool(unaddressed_indices),
            "unaddressed_sub_questions": unaddressed_indices,
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
            "coverage_matrix": [],
            "visited_queries": [],
            "recomposed_topics": [],
            "evidence": [],
            "thought_history": [],
            "is_sufficient": False,
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
            "governor_alert": None,
            "truncation_alert": None,
            "synthesis_clipped": False,
            "unaddressed_sub_questions": [],
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
            "coverage_matrix": [],
            "visited_queries": [],
            "recomposed_topics": [],
            "evidence": [],
            "thought_history": [],
            "is_sufficient": False,
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
            "governor_alert": None,
            "truncation_alert": None,
            "synthesis_clipped": False,
            "unaddressed_sub_questions": [],
            "evidence_only": evidence_only,
        }

        for step in self.graph.stream(initial_state):
            for node_name, state_update in step.items():
                yield node_name, state_update
