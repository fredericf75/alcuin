"""Unit tests for LangGraph cyclic ResearchAgent."""

from pathlib import Path
import pytest
from alcuin.agent.graph import AgentState, LLMClient, ResearchAgent
from alcuin.storage.engine import VaultStorage
from alcuin.storage.vault import Vault


class MockLLM(LLMClient):
    """Deterministic mock LLM for testing."""

    def __init__(self, answer: str = "Mock answer with evidence."):
        super().__init__()
        self.answer = answer
        self.calls = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if "sufficiency evaluator" in system_prompt:
            return '{"sufficient": true, "thought": "Mock evaluated as complete", "confidence": 0.9}'
        return self.answer


@pytest.fixture
def indexed_vault(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    vault_dir = tmp_path / "agent_vault"
    vault = Vault.init(vault_dir, name="agent-test")

    # Add knowledge documents
    doc1 = vault_dir / "clinic.md"
    doc1.write_text(
        "# Charlie's Pediatrician\n\nDr. Robinson works at 742 Evergreen Terrace, Springfield.\nPhone: 555-4321.",
        encoding="utf-8",
    )
    storage = VaultStorage(vault)
    storage.index()
    return vault


def test_research_agent_end_to_end(indexed_vault):
    mock_llm = MockLLM(answer="Charlie's pediatrician Dr. Robinson is located at 742 Evergreen Terrace.")
    agent = ResearchAgent(indexed_vault, llm=mock_llm)

    result = agent.research("What is the address for Charlie's pediatrician?")

    assert result["final_answer"] != ""
    assert "clinic.md" in result["sources"]
    assert len(result["evidence"]) > 0
    assert result["confidence"] > 0.0
    assert result["iteration"] >= 1


def test_research_agent_turn_ceiling_guardrail(indexed_vault):
    # Mock LLM that always claims insufficiency
    class AlwaysInsufficientLLM(LLMClient):
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            if "sufficiency evaluator" in system_prompt:
                return '{"sufficient": false, "thought": "Need more info", "next_query": "more info"}'
            return "Synthesized after max turns."

    # Set vault max_iterations to 2
    indexed_vault.config.max_iterations = 2
    agent = ResearchAgent(indexed_vault, llm=AlwaysInsufficientLLM())

    result = agent.research("Complex query that never finishes")
    # Must terminate deterministically at max_iterations
    assert result["iteration"] == 2
    assert result["is_sufficient"] is True


def test_research_agent_streaming(indexed_vault):
    mock_llm = MockLLM(answer="Streaming test answer.")
    agent = ResearchAgent(indexed_vault, llm=mock_llm)

    nodes_visited = []
    for node_name, state_update in agent.stream_research("Where is Dr. Robinson?"):
        nodes_visited.append(node_name)

    assert "retrieve" in nodes_visited
    assert "evaluate" in nodes_visited
    assert "synthesize" in nodes_visited
