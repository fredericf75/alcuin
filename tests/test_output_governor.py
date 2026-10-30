from unittest.mock import MagicMock
from alcuin.agent.graph import AgentState, ResearchAgent
from alcuin.storage.vault import Vault


def _make_mock_agent(tmp_path) -> ResearchAgent:
    vault = Vault.init(tmp_path / "v_outgov", name="v_outgov")
    agent = ResearchAgent(vault=vault)
    return agent


def test_output_governor_clipped_warning(tmp_path):
    """Verify that omitting a sub-question in monolithic synthesis triggers synthesis_clipped=True and warning."""
    agent = _make_mock_agent(tmp_path)
    agent.vault.config.subquery_synthesis = False
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
    """Verify that when all sub-questions are answered in monolithic synthesis, synthesis_clipped=False."""
    agent = _make_mock_agent(tmp_path)
    agent.vault.config.subquery_synthesis = False
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


def test_output_governor_subquery_synthesis_all_addressed(tmp_path):
    """Verify that isolated sub-query synthesis answers all sub-questions with synthesis_clipped=False."""
    agent = _make_mock_agent(tmp_path)
    assert agent.vault.config.subquery_synthesis is True
    agent.llm.chat = MagicMock(side_effect=[
        "First subquery direct finding.",
        "Second subquery direct finding.",
        "Third subquery direct finding.",
    ])

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
    assert "1. First subquery direct finding." in result["final_answer"]
    assert "2. Second subquery direct finding." in result["final_answer"]
    assert "3. Third subquery direct finding." in result["final_answer"]


def test_output_governor_subquery_synthesis_clipped(tmp_path):
    """Verify that if a sub-query returns empty, it is omitted and triggers synthesis_clipped=True."""
    agent = _make_mock_agent(tmp_path)
    assert agent.vault.config.subquery_synthesis is True
    # Q1 and Q2 succeed, Q3 returns empty response
    agent.llm.chat = MagicMock(side_effect=[
        "First subquery direct finding.",
        "Second subquery direct finding.",
        "",
    ])

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
