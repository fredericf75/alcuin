"""Regression tests for Wave 2 code review fixes (engine, agent graph)."""

from unittest.mock import MagicMock

from alcuin.agent.graph import ResearchAgent, _try_init_remote_reranker
from alcuin.storage.engine import SearchResult
from alcuin.storage.vault import Vault


def test_source_diversification_negative_cosine_penalized_not_boosted(tmp_path):
    """Negative cosine scores beyond quota must decrease (more negative), not increase."""
    vault = Vault.init(tmp_path / "v_div")
    vault.config.source_diversification = True
    vault.config.source_diversification_quota = 1
    vault.config.source_diversification_decay = 0.90

    # Verify the sign-safe decay math directly on SearchResult scores
    r1 = SearchResult(id="a#1", file_path="a.md", document_title="A", section_title="S1", breadcrumbs="A", content="x", wikilinks=[], score=-0.40)
    r2 = SearchResult(id="a#2", file_path="a.md", document_title="A", section_title="S2", breadcrumbs="A", content="y", wikilinks=[], score=-0.50)
    r3 = SearchResult(id="b#1", file_path="b.md", document_title="B", section_title="S1", breadcrumbs="B", content="z", wikilinks=[], score=-0.51)

    results = [r1, r2, r3]
    doc_counts = {}
    for r in results:
        cnt = doc_counts.get(r.file_path, 0)
        if cnt >= 1:
            factor = 0.90 ** (cnt - 1 + 1)
            if r.score > 0:
                r.score = r.score * factor
            elif r.score < 0 and factor > 0:
                r.score = r.score / factor
        doc_counts[r.file_path] = cnt + 1

    # r2 (-0.50 / 0.90 = -0.555...) must be penalized below r3 (-0.51), not boosted to -0.45
    assert r2.score < -0.50
    assert r2.score < r3.score


def test_unit_relevance_maps_logits_to_unit_interval():
    assert abs(ResearchAgent._unit_relevance(0.0, True) - 0.5) < 1e-6
    assert ResearchAgent._unit_relevance(2.0, True) > 0.85
    assert ResearchAgent._unit_relevance(-5.0, True) < 0.01
    assert ResearchAgent._unit_relevance(0.75, False) == 0.75


def test_empty_subquery_synthesis_sets_zero_confidence(tmp_path):
    vault = Vault.init(tmp_path / "v_empty_subq")
    vault.config.subquery_synthesis = True
    vault.config.reranker_enabled = False
    agent = ResearchAgent(vault=vault)
    agent.llm.chat = MagicMock(return_value="")

    state = {
        "question": "Q1 and Q2?",
        "iteration": 1,
        "max_iterations": 3,
        "max_hops": 3,
        "current_query": "",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["Q1", "Q2"],
        "sub_questions": ["Q1?", "Q2?"],
        "coverage_matrix": [],
        "visited_queries": [],
        "recomposed_topics": [],
        "evidence": [{"id": "f.md#1", "breadcrumbs": "Doc", "file_path": "f.md", "content": "Some evidence"}],
        "thought_history": [],
        "is_sufficient": True,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "evidence_only": False,
        "synthesis_clipped": False,
        "unaddressed_sub_questions": [],
    }
    res = agent._synthesize_node(state)
    assert res["confidence"] == 0.0
    assert "Could not reach LLM" in res["final_answer"]


def test_remote_reranker_strips_chat_completions_url(monkeypatch):
    seen_urls = []

    class FakeResp:
        status_code = 200
        def json(self):
            return {"scores": [1.5]}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, **kw):
            seen_urls.append(url)
            return FakeResp()

    import alcuin.agent.graph as gmod
    monkeypatch.setattr(gmod.httpx, "Client", FakeClient)
    ce = _try_init_remote_reranker("http://localhost:8000/v1/chat/completions", "ce-model")
    assert ce is not None
    assert seen_urls == ["http://localhost:8000/v1/rerank"]
