from unittest.mock import MagicMock
from alcuin.agent.graph import AgentState, ResearchAgent
from alcuin.storage.vault import Vault


def _make_mock_agent(tmp_path, response_text: str) -> ResearchAgent:
    vault = Vault.init(tmp_path / "v_cov", name="v_cov")
    agent = ResearchAgent(vault=vault)
    agent.llm.chat = MagicMock(return_value=response_text)
    return agent


def test_coverage_matrix_all_covered(tmp_path):
    mock_json = """{
        "coverage": [
            {"id": 1, "status": "COVERED"},
            {"id": 2, "status": "COVERED"}
        ],
        "all_covered": true,
        "thought": "All sub-questions addressed in evidence.",
        "confidence": 0.95
    }"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "What is the notice deadline and default decision maker?",
        "iteration": 0,
        "max_iterations": 3,
        "max_hops": 3,
        "current_query": "test",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["notice deadline", "decision maker"],
        "sub_questions": ["What is the notice deadline?", "Who is the default decision maker?"],
        "coverage_matrix": [],
        "visited_queries": ["notice deadline", "decision maker"],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc > Sec", "file_path": "f.md", "content": "Evidence text"}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    assert result["is_sufficient"] is True
    assert len(result["coverage_matrix"]) == 2
    assert result["coverage_matrix"][0]["status"] == "COVERED"
    assert result["coverage_matrix"][1]["status"] == "COVERED"


def test_coverage_matrix_missing_forces_turn_2(tmp_path):
    mock_json = """{
        "coverage": [
            {"id": 1, "status": "COVERED"},
            {"id": 2, "status": "MISSING", "gap_query": "AIA claim for delay deadline additional cost"}
        ],
        "all_covered": false,
        "thought": "Evidence is missing delay claim details.",
        "confidence": 0.5
    }"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "Notice for rock and deadline for delay claim?",
        "iteration": 0,
        "max_iterations": 3,
        "max_hops": 3,
        "current_query": "rock notice",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["rock notice"],
        "sub_questions": ["What is the notice for rock?", "What is the deadline for delay claim?"],
        "coverage_matrix": [],
        "visited_queries": ["rock notice"],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc > Sec", "file_path": "f.md", "content": "14 days for rock."}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    assert result["is_sufficient"] is False
    assert result["current_query"] == "AIA claim for delay deadline additional cost"
    assert len(result["coverage_matrix"]) == 2
    assert result["coverage_matrix"][1]["status"] == "MISSING"
    assert result["coverage_matrix"][1]["gap_query"] == "AIA claim for delay deadline additional cost"


def test_coverage_matrix_turn_ceiling_terminates(tmp_path):
    mock_json = """{"coverage": [{"id": 1, "status": "MISSING", "gap_query": "search query"}], "all_covered": false}"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "Unsolvable inquiry",
        "iteration": 4,  # Hard ceiling reached (max_iter = min(5, 5) -> 5)
        "max_iterations": 5,
        "max_hops": 3,
        "current_query": "query",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": [],
        "sub_questions": ["What is X?"],
        "coverage_matrix": [],
        "visited_queries": [],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc", "file_path": "f.md", "content": "Some text"}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    # Iteration becomes 5 >= max_iter (5) -> must terminate to synthesis
    assert result["is_sufficient"] is True


def test_coverage_matrix_governor_saturation_precedence(tmp_path):
    mock_json = """{
        "coverage": [{"id": 1, "status": "MISSING", "gap_query": "more evidence"}],
        "all_covered": false
    }"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "Big query",
        "iteration": 0,
        "max_iterations": 3,
        "max_hops": 3,
        "current_query": "query",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["goal 1"],
        "sub_questions": ["What is X?"],
        "coverage_matrix": [],
        "visited_queries": ["goal 1"],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc", "file_path": "f.md", "content": "Massive context"}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": "Context Governor Capacity Reached: 8000 tokens",
        "truncation_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    # Saturation forces termination to synthesis even if coverage is MISSING
    assert result["is_sufficient"] is True


def test_coverage_matrix_atomic_single_question_evaluates_directly(tmp_path):
    mock_json = """{
        "sufficient": true,
        "thought": "Riley was born on August 8, 2011 as stated in profile.",
        "confidence": 0.95
    }"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "What is riley's birthday?",
        "iteration": 0,
        "max_iterations": 4,
        "max_hops": 3,
        "current_query": "what is riley's birthday",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["what is riley's birthday"],
        "sub_questions": ["What is riley's birthday?"],
        "coverage_matrix": [],
        "visited_queries": ["what is riley's birthday"],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Riley > Profile", "file_path": "riley.md", "content": "Birthday: 2011-08-08"}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    assert result["is_sufficient"] is True
    assert result["coverage_matrix"] == []
    # Verify the prompt sent to LLM was the direct single-question prompt, not the sub-question checklist
    call_sys_prompt = agent.llm.chat.call_args[0][0]
    assert "Sub-Questions to Verify" not in agent.llm.chat.call_args[0][1]
    assert "sufficient" in call_sys_prompt


def test_coverage_matrix_ignores_out_of_bounds_phantom_ids(tmp_path):
    mock_json = """{
        "coverage": [
            {"id": 1, "status": "COVERED"},
            {"id": 2, "status": "COVERED"},
            {"id": 3, "status": "MISSING", "gap_query": "hallucinated phantom query"}
        ],
        "all_covered": true,
        "thought": "Both genuine questions answered.",
        "confidence": 0.9
    }"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "What is the penalty and the grace period?",
        "iteration": 0,
        "max_iterations": 4,
        "max_hops": 3,
        "current_query": "penalty",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["penalty", "grace period"],
        "sub_questions": ["What is the penalty?", "What is the grace period?"],
        "coverage_matrix": [],
        "visited_queries": ["penalty", "grace period"],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Doc", "file_path": "f.md", "content": "Penalty: $500. Grace period: 30 days."}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    assert result["is_sufficient"] is True
    # Phantom id 3 must be filtered out
    assert len(result["coverage_matrix"]) == 2
    assert [item["id"] for item in result["coverage_matrix"]] == [1, 2]
    assert result["coverage_matrix"][0]["status"] == "COVERED"
    assert result["coverage_matrix"][1]["status"] == "COVERED"


def test_coverage_matrix_filters_hallucinated_web_queries(tmp_path):
    mock_json = """{
        "sufficient": false,
        "thought": "Need to find more external details.",
        "next_query": "Celebrity birthdays Ava",
        "confidence": 0.4
    }"""
    agent = _make_mock_agent(tmp_path, mock_json)

    state: AgentState = {
        "question": "What is Ava's birthday?",
        "iteration": 2,
        "max_iterations": 4,
        "max_hops": 3,
        "current_query": "Ava's birthdate",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": ["Ava's birthdate"],
        "sub_questions": ["What is Ava's birthday?"],
        "coverage_matrix": [],
        "visited_queries": ["Ava's birthdate"],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Ava > Profile", "file_path": "ava.md", "content": "Date of Birth: 2009-11-12"}],
        "thought_history": [],
        "is_sufficient": False,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "governor_alert": None,
        "truncation_alert": None,
        "friction_alert": None,
        "evidence_only": False,
    }

    result = agent._evaluate_node(state)
    # The hallucinated web query must be rejected, and because no unvisited goals remain, it must terminate to synthesis
    assert result["is_sufficient"] is True
    assert result["current_query"] == ""


def test_friction_alert_generated_on_multi_turn_synthesis(tmp_path):
    agent = _make_mock_agent(tmp_path, "Ava was born on November 12, 2009.")

    state: AgentState = {
        "question": "What is Ava's birthday?",
        "iteration": 2,  # Turn 3
        "max_iterations": 4,
        "max_hops": 3,
        "current_query": "query",
        "visited_links": [],
        "pending_links": [],
        "sub_goals": [],
        "sub_questions": [],
        "coverage_matrix": [],
        "visited_queries": [],
        "recomposed_topics": [],
        "evidence": [{"breadcrumbs": "Ava > Profile", "file_path": "ava.md", "content": "Date of Birth: 2009-11-12"}],
        "thought_history": [],
        "is_sufficient": True,
        "final_answer": "",
        "confidence": 0.95,
        "sources": ["ava.md"],
        "governor_alert": None,
        "truncation_alert": None,
        "friction_alert": None,
        "evidence_only": False,
    }

    result = agent._synthesize_node(state)
    assert result["friction_alert"] is not None
    assert "High retrieval friction detected" in result["friction_alert"]
    assert "3 turns" in result["friction_alert"]
