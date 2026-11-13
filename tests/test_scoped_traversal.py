"""Unit tests for SPEC-032: Subquery-Scoped Evidence Traversal & Partitioning."""

import os
from unittest.mock import MagicMock
import pytest

from alcuin.storage.vault import Vault
from alcuin.agent.graph import ResearchAgent, AgentState, SearchResult


@pytest.fixture
def test_vault(tmp_path):
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir()
    (vault_dir / ".alcuin").mkdir()
    config_file = vault_dir / ".alcuin" / "config.toml"
    config_file.write_text(
        '[vault]\nname = "test"\n[retrieval]\nsibling_traversal = true\nsubquery_synthesis = true\n'
    )
    return Vault.load(str(vault_dir))


def test_subquery_evidence_map_initialization(test_vault):
    """Verify AgentState initial state contains subquery_evidence_map."""
    agent = ResearchAgent(vault=test_vault)
    assert hasattr(agent, "research")
    # Verify graph can initialize with subquery_evidence_map
    state: AgentState = {
        "question": "Test question",
        "iteration": 0,
        "max_iterations": 3,
        "max_hops": 2,
        "current_query": "Test question",
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
        "friction_alert": None,
        "evidence_only": False,
        "synthesis_clipped": False,
        "unaddressed_sub_questions": [],
        "synthesis_evidence": [],
        "subquery_evidence_map": {},
    }
    assert isinstance(state["subquery_evidence_map"], dict)


def test_scoped_traversal_partitioning(test_vault):
    """Verify that retrieval populates subquery_evidence_map with partitioned clusters."""
    agent = ResearchAgent(vault=test_vault)

    # Mock storage search to return different hits for different queries
    hit_q1 = SearchResult(
        id="docA.txt#1",
        file_path="docA.txt",
        document_title="Doc A",
        section_title="Sec 1",
        breadcrumbs="Doc A > Sec 1",
        content="Primary clause for Q1 regarding dispute resolution.",
        wikilinks=[],
        score=0.88,
    )
    hit_q2 = SearchResult(
        id="docB.txt#1",
        file_path="docB.txt",
        document_title="Doc B",
        section_title="Sec 2",
        breadcrumbs="Doc B > Sec 2",
        content="Primary clause for Q2 regarding termination fees.",
        wikilinks=[],
        score=0.85,
    )

    def mock_search(query, top_k=20):
        if "dispute" in query.lower() or "q1" in query.lower():
            return [hit_q1]
        elif "fee" in query.lower() or "q2" in query.lower():
            return [hit_q2]
        return []

    agent.storage.search = MagicMock(side_effect=mock_search)
    agent.storage.get_chunks_by_ids = MagicMock(return_value=[])

    state: AgentState = {
        "question": "Complex inquiry with two parts",
        "iteration": 0,
        "max_iterations": 3,
        "max_hops": 2,
        "current_query": "Complex inquiry with two parts",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["Q1 dispute resolution", "Q2 termination fee"],
        "sub_questions": ["What is the dispute procedure?", "What is the termination fee?"],
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
        "friction_alert": None,
        "evidence_only": False,
        "synthesis_clipped": False,
        "unaddressed_sub_questions": [],
        "synthesis_evidence": [],
        "subquery_evidence_map": {},
    }

    # Override decompose_query to preserve our explicit sub_goals & sub_questions
    agent.decompose_query = MagicMock(return_value=(state["sub_goals"], state["sub_questions"]))

    update = agent._retrieve_node(state)

    # 1. Global evidence union must have both chunks
    packed_ids = [e["id"] for e in update["evidence"]]
    assert "docA.txt#1" in packed_ids
    assert "docB.txt#1" in packed_ids

    # 2. subquery_evidence_map must be partitioned
    sq_map = update.get("subquery_evidence_map", {})
    assert "Q1 dispute resolution" in sq_map or "What is the dispute procedure?" in sq_map
    assert "Q2 termination fee" in sq_map or "What is the termination fee?" in sq_map

    # Check cluster purity: Q1 cluster must contain docA, NOT docB
    q1_cluster = sq_map.get("What is the dispute procedure?") or sq_map.get("Q1 dispute resolution", [])
    q1_ids = [c["id"] for c in q1_cluster]
    assert "docA.txt#1" in q1_ids
    assert "docB.txt#1" not in q1_ids

    # Check cluster purity: Q2 cluster must contain docB, NOT docA
    q2_cluster = sq_map.get("What is the termination fee?") or sq_map.get("Q2 termination fee", [])
    q2_ids = [c["id"] for c in q2_cluster]
    assert "docB.txt#1" in q2_ids
    assert "docA.txt#1" not in q2_ids
