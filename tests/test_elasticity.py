"""
tests/test_elasticity.py - Unit tests for SPEC-034 Dynamic Score Elasticity admission.
"""
from unittest.mock import MagicMock
import pytest
from alcuin.storage.vault import VaultConfig, Vault
from alcuin.storage.engine import SearchResult
from alcuin.agent.graph import ResearchAgent


@pytest.fixture
def test_vault(tmp_path):
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir()
    (vault_dir / ".alcuin").mkdir()
    config_file = vault_dir / ".alcuin" / "config.toml"
    config_file.write_text(
        '[vault]\nname = "test"\n'
    )
    return Vault.load(str(vault_dir))


def _make_result(doc_id: str, score: float, sec_num: int = 1) -> SearchResult:
    return SearchResult(
        id=doc_id,
        file_path=f"test_{doc_id}.md",
        document_title="Test Document",
        section_title=f"Section {sec_num}",
        breadcrumbs=f"Test > Section {sec_num}.0",
        content=f"Content for {doc_id}",
        wikilinks=[],
        score=score,
    )


def test_elasticity_floor_guaranteed_even_with_negative_scores(test_vault):
    """Verify that k_min=2 chunks are guaranteed even when scores are deeply negative."""
    test_vault.config.retrieval_elasticity = True
    test_vault.config.subquery_top_k_min = 2
    test_vault.config.subquery_top_k = 4

    agent = ResearchAgent(vault=test_vault, llm=MagicMock())
    agent.decompose_query = MagicMock(return_value=(["sub q1"], ["sub q1"]))
    # Mock CrossEncoder
    agent.cross_encoder = MagicMock()
    # Scores for 4 chunks: all negative
    agent.cross_encoder.predict.return_value = [-3.0, -4.0, -5.0, -6.0]

    agent.storage = MagicMock()
    raw = [
        _make_result("c1", 0.0, sec_num=1),
        _make_result("c2", 0.0, sec_num=2),
        _make_result("c3", 0.0, sec_num=3),
        _make_result("c4", 0.0, sec_num=4),
    ]
    agent.storage.search.return_value = raw

    # Run _plan_node logic for a single target query
    state = {
        "question": "test question",
        "sub_questions": ["sub q1"],
        "sub_goals": ["sub q1"],
        "visited_queries": [],
        "evidence": [],
        "visited_ids": set(),
        "visited_links": set(),
        "pending_links": [],
        "thought_history": [],
        "sources": set(),
        "new_sources": set(),
        "iteration": 0,
        "max_iterations": 1,
    }
    new_state = agent._retrieve_node(state)
    ev_map = new_state.get("subquery_evidence_map", {})
    admitted = ev_map.get("sub q1", [])

    # Exactly 2 chunks should be admitted (the k_min floor)
    # Since c3 and c4 have scores -5.0 and -6.0 (below floor -0.5 and delta > 2.5), they are pruned
    primary_admitted = [c for c in admitted if c.get("is_primary")]
    assert len(primary_admitted) == 2
    assert primary_admitted[0]["id"] == "c1"
    assert primary_admitted[1]["id"] == "c2"


def test_elasticity_admits_high_confidence_and_elbow(test_vault):
    """Verify that chunks with CE >= 1.0 or within elbow delta are admitted up to subquery_top_k."""
    test_vault.config.retrieval_elasticity = True
    test_vault.config.subquery_top_k_min = 2
    test_vault.config.subquery_top_k = 4
    test_vault.config.elasticity_score_threshold = 1.0
    test_vault.config.elasticity_elbow_floor = -0.5
    test_vault.config.elasticity_elbow_delta = 2.5

    agent = ResearchAgent(vault=test_vault, llm=MagicMock())
    agent.decompose_query = MagicMock(return_value=(["sub q1"], ["sub q1"]))
    agent.cross_encoder = MagicMock()
    # Rank 1: 4.5, Rank 2: 3.5, Rank 3: 2.5 (passes >= 1.0), Rank 4: 2.1 (passes >= 1.0), Rank 5: 1.8 (passes >= 1.0, but capped by sub_k=4)
    agent.cross_encoder.predict.return_value = [4.5, 3.5, 2.5, 2.1, 1.8]

    agent.storage = MagicMock()
    raw = [
        _make_result("c1", 0.0, sec_num=1),
        _make_result("c2", 0.0, sec_num=2),
        _make_result("c3", 0.0, sec_num=3),
        _make_result("c4", 0.0, sec_num=4),
        _make_result("c5", 0.0, sec_num=5),
    ]
    agent.storage.search.return_value = raw

    state = {
        "question": "test question",
        "sub_questions": ["sub q1"],
        "sub_goals": ["sub q1"],
        "visited_queries": [],
        "evidence": [],
        "visited_ids": set(),
        "visited_links": set(),
        "pending_links": [],
        "thought_history": [],
        "sources": set(),
        "new_sources": set(),
        "iteration": 0,
        "max_iterations": 1,
    }
    new_state = agent._retrieve_node(state)
    ev_map = new_state.get("subquery_evidence_map", {})
    admitted = ev_map.get("sub q1", [])
    primary_admitted = [c for c in admitted if c.get("is_primary")]

    # Should admit 4 chunks (c1, c2, c3, c4), capped by subquery_top_k=4
    assert len(primary_admitted) == 4
    assert [c["id"] for c in primary_admitted] == ["c1", "c2", "c3", "c4"]


def test_elasticity_disabled_falls_back_to_static_top_k(test_vault):
    """Verify that when retrieval_elasticity=False, static subquery_top_k is used directly."""
    test_vault.config.retrieval_elasticity = False
    test_vault.config.subquery_top_k = 3

    agent = ResearchAgent(vault=test_vault, llm=MagicMock())
    agent.decompose_query = MagicMock(return_value=(["sub q1"], ["sub q1"]))
    agent.cross_encoder = MagicMock()
    # Rank 1: 5.0, Rank 2: -4.0, Rank 3: -5.0, Rank 4: -6.0
    agent.cross_encoder.predict.return_value = [5.0, -4.0, -5.0, -6.0]

    agent.storage = MagicMock()
    raw = [
        _make_result("c1", 0.0, sec_num=1),
        _make_result("c2", 0.0, sec_num=2),
        _make_result("c3", 0.0, sec_num=3),
        _make_result("c4", 0.0, sec_num=4),
    ]
    agent.storage.search.return_value = raw

    state = {
        "question": "test question",
        "sub_questions": ["sub q1"],
        "sub_goals": ["sub q1"],
        "visited_queries": [],
        "evidence": [],
        "visited_ids": set(),
        "visited_links": set(),
        "pending_links": [],
        "thought_history": [],
        "sources": set(),
        "new_sources": set(),
        "iteration": 0,
        "max_iterations": 1,
    }
    new_state = agent._retrieve_node(state)
    ev_map = new_state.get("subquery_evidence_map", {})
    admitted = ev_map.get("sub q1", [])
    primary_admitted = [c for c in admitted if c.get("is_primary")]

    # With elasticity disabled, exactly top 3 are taken blindly
    assert len(primary_admitted) == 3
    assert [c["id"] for c in primary_admitted] == ["c1", "c2", "c3"]

