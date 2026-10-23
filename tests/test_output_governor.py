from unittest.mock import MagicMock
from alcuin.agent.graph import AgentState, ResearchAgent
from alcuin.storage.vault import Vault


def _make_mock_agent(tmp_path) -> ResearchAgent:
    vault = Vault.init(tmp_path / "v_outgov", name="v_outgov")
    agent = ResearchAgent(vault=vault)
    return agent


def test_output_governor_clipped_warning(tmp_path):
    """Verify that omitting a sub-question triggers synthesis_clipped=True and warning."""
    agent = _make_mock_agent(tmp_path)
    # Mock LLM returns only answers 1 and 2, missing 3
    agent.llm.chat = MagicMock(return_value="1. First answer details.\n\n2. Second answer details.")

    state: AgentState = {
        "question": "Multi-question query",
        "iteration": 1,
        "max_iterations": 3,
        "max_hops": 3,
        "current_query": "test",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["q1", "q2", "q3"],
        "sub_questions": ["Question 1?", "Question 2?", "Question 3?"],
        "coverage_matrix": [],
        "visited_queries": [],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc > Sec", "file_path": "f.md", "content": "Evidence text"}],
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

    result = agent._synthesize_node(state)
    assert result["synthesis_clipped"] is True
    assert result["unaddressed_sub_questions"] == [3]
    assert any("[OUTPUT GOVERNOR WARNING]" in th for th in result["thought_history"])


def test_output_governor_all_addressed(tmp_path):
    """Verify that when all sub-questions are answered, synthesis_clipped=False."""
    agent = _make_mock_agent(tmp_path)
    agent.llm.chat = MagicMock(return_value="1. First answer.\n\n2. Second answer.\n\n3. Third answer.")

    state: AgentState = {
        "question": "Multi-question query",
        "iteration": 1,
        "max_iterations": 3,
        "max_hops": 3,
        "current_query": "test",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["q1", "q2", "q3"],
        "sub_questions": ["Question 1?", "Question 2?", "Question 3?"],
        "coverage_matrix": [],
        "visited_queries": [],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc > Sec", "file_path": "f.md", "content": "Evidence text"}],
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

    result = agent._synthesize_node(state)
    assert result["synthesis_clipped"] is False
    assert result["unaddressed_sub_questions"] == []



