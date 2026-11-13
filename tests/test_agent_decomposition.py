from unittest.mock import MagicMock
from alcuin.agent.graph import ResearchAgent
from alcuin.storage.vault import Vault


def _make_mock_agent(tmp_path, response_text: str = "") -> ResearchAgent:
    vault = Vault.init(tmp_path / "v_decomp", name="v_decomp")
    agent = ResearchAgent(vault=vault)
    agent.planner_llm.chat = MagicMock(return_value=response_text)
    return agent


def test_structured_query_bypasses_llm(tmp_path):
    agent = _make_mock_agent(tmp_path)
    question = "Under 10A NCAC 13F:\n1. What are personal care aide duty hours for a 30-bed facility?\n2. When must an aide remain awake at all times?"
    sub_goals, sub_questions = agent.decompose_query(question)
    assert len(sub_questions) == 2
    assert "personal care aide duty hours" in sub_questions[0]
    assert "remain awake at all times" in sub_questions[1]
    assert agent.planner_llm.chat.call_count == 0


def test_short_query_bypasses_llm(tmp_path):
    agent = _make_mock_agent(tmp_path)
    question = "What is the capital of North Carolina?"
    sub_goals, sub_questions = agent.decompose_query(question)
    assert sub_questions == [question]
    assert sub_goals == [question]
    assert agent.planner_llm.chat.call_count == 0


def test_unstructured_query_user_framing_and_few_shot(tmp_path):
    llm_output = (
        "1. What is the maximum permitted size for a detached accessory garage in Cary?\n"
        "2. What are the minimum setback distances from the property line for a detached garage in Cary?\n"
        "3. Is a separate electrical permit required for a detached accessory structure?"
    )
    agent = _make_mock_agent(tmp_path, llm_output)
    question = (
        "I want to understand the zoning rules in Cary for building a detached garage behind my house. "
        "Can I build a 600 sq ft structure, how close can it sit to my neighbor's boundary line, and "
        "do I need an electrical permit if I run a subpanel out there?"
    )
    sub_goals, sub_questions = agent.decompose_query(question)

    assert agent.planner_llm.chat.call_count == 1
    sys_prompt, user_prompt = agent.planner_llm.chat.call_args[0]

    # Verify dual-role framing and few-shot exemplar presence
    assert "You are a research query planner." in sys_prompt
    assert "ENTITY PRESERVATION" in sys_prompt
    assert "[Task]" in user_prompt
    assert "[Examples]" in user_prompt
    assert "[Inquiry]" in user_prompt
    assert "[Numbered Questions]" in user_prompt
    assert question in user_prompt

    # Verify decomposed sub-questions
    assert len(sub_questions) == 3
    assert "maximum permitted size" in sub_questions[0]
    assert "setback distances" in sub_questions[1]
    assert "separate electrical permit" in sub_questions[2]


def test_unstructured_query_markdown_bold_and_echoed_tags(tmp_path):
    llm_output = (
        "[Numbered Questions]\n"
        "Here are the deconstructed questions:\n"
        "**1.** How many daily personal care aide duty hours are required for a 30-bed adult care home?\n"
        "**2.** Under what conditions must an adult care home personal care aide remain awake at all times?"
    )
    agent = _make_mock_agent(tmp_path, llm_output)
    question = (
        "What are the staffing requirements under North Carolina adult care regulations for an adult care home "
        "with 30 beds in terms of aide duty hours, and do aides need to stay awake on third shift?"
    )
    sub_goals, sub_questions = agent.decompose_query(question)

    assert len(sub_questions) == 2
    assert sub_questions[0].startswith("How many daily personal care aide duty hours")
    assert sub_questions[1].startswith("Under what conditions must an adult care home")


def test_unstructured_query_fallback_on_unparseable(tmp_path):
    agent = _make_mock_agent(tmp_path, "Certainly! Facilities must follow rules.")
    question = (
        "Please provide a comprehensive summary of all applicable zoning standards for accessory dwellings, "
        "including height limits, setback requirements, parking spaces, and utility connections."
    )
    sub_goals, sub_questions = agent.decompose_query(question)
    assert sub_questions == [question]
    assert sub_goals == [question]
