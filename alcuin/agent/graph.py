"""
alcuin.agent.graph - LangGraph Cyclic ReAct Research Agent with Hard Turn Ceiling
"""

from __future__ import annotations

import json
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
    visited_queries: List[str]
    recomposed_topics: List[str]
    evidence: List[Dict[str, Any]]
    thought_history: List[str]
    is_sufficient: bool
    final_answer: str
    confidence: float
    sources: List[str]


class LLMClient:
    """Interacts with LLM endpoint (Ollama / Google AI Studio / OpenAI-compatible / Fallback)."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
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
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        is_local = "localhost" in self.endpoint or "127.0.0.1" in self.endpoint
        is_google = (
            "generativelanguage.googleapis.com" in self.endpoint
            or "google" in self.endpoint
            or (self.api_key and (self.model.startswith(("gemini", "gemma")) and not is_local))
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
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                non_thought = [p.get("text", "") for p in parts if not p.get("thought", False) and p.get("text")]
                                if non_thought:
                                    return "\n".join(non_thought).strip()
                                return parts[-1].get("text", "").strip()
            except Exception:
                pass

        # 2. If local endpoint (Ollama /api/chat)
        if is_local:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.endpoint}/api/chat",
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "stream": False,
                            "options": {"temperature": self.temperature},
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        msg = data.get("message", {}).get("content", "")
                        if msg:
                            return msg.strip()
            except Exception:
                pass

        # 3. OpenAI-compatible /v1/chat/completions (hosted or local)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                target_url = (
                    self.endpoint
                    if self.endpoint.endswith("/chat/completions")
                    else f"{self.endpoint}/v1/chat/completions"
                )
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
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass

        # 4. Fallback to Ollama if not already tried
        if not is_local:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.endpoint}/api/chat",
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "stream": False,
                            "options": {"temperature": self.temperature},
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        msg = data.get("message", {}).get("content", "")
                        if msg:
                            return msg.strip()
            except Exception:
                pass

        return ""


class ResearchAgent:
    """Autonomous multi-hop knowledge research agent."""

    def __init__(self, vault: Vault, llm: Optional[LLMClient] = None):
        self.vault = vault
        self.storage = VaultStorage(vault)
        self.llm = llm or LLMClient(
            endpoint=vault.config.model_endpoint,
            model=vault.config.model_name,
            temperature=vault.config.temperature,
        )
        self.graph = self._build_graph()

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
            {"retrieve": "retrieve", "synthesize": "synthesize"},
        )
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def decompose_query(self, question: str) -> List[str]:
        """Decompose a complex or multi-part query into targeted retrieval queries."""
        has_subparts = bool(re.search(r"(?:[A-Z]\.|\b\d+\.)", question))
        is_long = len(question.split()) > 25

        if not (has_subparts or is_long):
            return [question]

        queries: List[str] = []

        # 1. Deterministic extraction: Full labeled clauses (e.g. A. Right of Entry... B. Specific Violations...)
        clause_matches = re.findall(r"([A-Z]\.\s*[^\n]+(?:\n(?![A-Z]\.|\d+\.)[^\n]+)?)", question)
        for c in clause_matches:
            cleaned = " ".join(c.split())
            cleaned = re.sub(r"^[A-Z]\.\s*", "", cleaned)
            shortened = cleaned[:160].strip()
            if shortened and shortened not in queries:
                queries.append(shortened)

        # 2. Numbered items (e.g. 1. The property owner..., 2. The primary dwelling...)
        num_matches = re.findall(r"\b\d+\.\s*([^\n\.\?]{8,120})", question)
        for n in num_matches:
            n_clean = " ".join(n.split())[:120].strip()
            if n_clean and n_clean not in queries:
                queries.append(n_clean)

        # 3. Augment with LLM decomposition if needed for unformatted queries
        if len(queries) < 2:
            sys_prompt = (
                "You are a research query planner.\n"
                "Break down the user's inquiry into 2 to 4 concise search queries (3 to 6 keywords each) "
                "designed to retrieve the relevant statutes, sections, and facts from a document database.\n"
                "Output each search query on a new line. Do not number or prefix them."
            )
            llm_resp = self.llm.chat(sys_prompt, question[:2000])
            if llm_resp:
                for line in llm_resp.strip().splitlines():
                    cleaned = re.sub(r"^[\d\.\-\*\•\s]+", "", line).strip()
                    if cleaned and len(cleaned.split()) >= 2 and cleaned not in queries:
                        queries.append(cleaned)

        if not queries:
            queries = [question]

        return queries[:8]

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
        llm_resp = self.llm.chat(sys_prompt, f"Context:\n{context_str}\n\nHigh-level topic: {topic}")
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

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        evidence_list = list(state.get("evidence", []))
        visited_ids = {e["id"] for e in evidence_list}
        new_sources: Set[str] = set(state.get("sources", []))
        pending_links = list(state.get("pending_links", []))
        visited_links = list(state.get("visited_links", []))
        visited_queries = list(state.get("visited_queries", []))
        sub_goals = list(state.get("sub_goals", []))
        recomposed_topics = list(state.get("recomposed_topics", []))
        thought_history = list(state.get("thought_history", []))

        # 1. On turn 0, decompose question into discrete sub-goals
        if state["iteration"] == 0:
            sub_goals = self.decompose_query(state["question"])
            # Batch first 2-3 sub_goals for initial search
            search_targets = [g for g in sub_goals if g not in visited_queries][:3]
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

        for target_query in search_targets:
            if target_query in visited_queries:
                continue
            visited_queries.append(target_query)
            results = self.storage.search(target_query, top_k=4)
            new_chunks_count = 0
            for r in results:
                if r.id not in visited_ids:
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
                    # Add newly discovered cross-references to pending_links if in scope
                    for link in r.wikilinks:
                        link_clean = " ".join(link.split()).strip()
                        if re.match(r"^§\s*\d+(?:\.\d+)?$", link_clean):
                            continue
                        if self._is_in_inquiry_scope(link_clean, state["question"]):
                            if link_clean not in visited_links and link_clean not in pending_links:
                                pending_links.append(link_clean)

            # Targeted Topic Re-Decomposition Guard (Depth 1):
            # If query yielded zero new chunks and hasn't been re-decomposed yet, drill down!
            if new_chunks_count == 0 and target_query not in recomposed_topics:
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
            "recomposed_topics": recomposed_topics,
            "thought_history": thought_history,
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

        # If LLM said sufficient but we haven't visited primary sub-goals yet, don't stop prematurely
        if sufficient and unvisited_goals and len(state["evidence"]) < 8:
            sufficient = False

        # Deterministic Anti-Stagnation Fallback: Never repeat the same query!
        if not sufficient:
            if not next_query or next_query in visited_queries or next_query == state["question"]:
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

        # Budget evidence to fit model context window (dynamic by model endpoint)
        is_local = "localhost" in self.llm.endpoint or "127.0.0.1" in self.llm.endpoint
        max_evidence_chars = 16000 if is_local else 120000
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
            "based strictly on the provided evidence. "
            "If the user asks for a specific fact (such as a date, name, birthday, amount, or threshold), give only the direct answer. "
            "Do not recite unnecessary background, filler, or raw evidence dumps. "
            "When citing tables, schedules, deadlines, or fine structures, transcribe the row values and numbers verbatim. "
            "If the evidence does not contain the answer, state clearly and concisely that the information is not found."
        )
        user_prompt = (
            f"Question: {state['question']}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Direct Answer:"
        )

        answer = self.llm.chat(sys_prompt, user_prompt)
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
        return {
            "final_answer": answer,
            "confidence": confidence,
        }

    def research(self, question: str) -> AgentState:
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
            "visited_queries": [],
            "recomposed_topics": [],
            "evidence": [],
            "thought_history": [],
            "is_sufficient": False,
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
        }

        result = self.graph.invoke(initial_state)
        return result

    def stream_research(self, question: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
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
            "visited_queries": [],
            "recomposed_topics": [],
            "evidence": [],
            "thought_history": [],
            "is_sufficient": False,
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
        }

        for step in self.graph.stream(initial_state):
            for node_name, state_update in step.items():
                yield node_name, state_update
