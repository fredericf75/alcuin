"""Unit tests for LangGraph cyclic ResearchAgent."""

from pathlib import Path
import pytest
from alcuin.agent.graph import AgentState, LLMClient, ModelNotFoundError, ResearchAgent
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
    indexed_vault.config.multi_hop = True
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

    # Set vault max_iterations to 2 and multi_hop to True
    indexed_vault.config.max_iterations = 2
    indexed_vault.config.multi_hop = True
    agent = ResearchAgent(indexed_vault, llm=AlwaysInsufficientLLM())

    result = agent.research("Complex query that never finishes")
    # Must terminate deterministically at max_iterations
    assert result["iteration"] == 2
    assert result["is_sufficient"] is True


def test_research_agent_streaming(indexed_vault):
    indexed_vault.config.multi_hop = True
    mock_llm = MockLLM(answer="Streaming test answer.")
    agent = ResearchAgent(indexed_vault, llm=mock_llm)

    nodes_visited = []
    for node_name, state_update in agent.stream_research("Where is Dr. Robinson?"):
        nodes_visited.append(node_name)

    assert "retrieve" in nodes_visited
    assert "evaluate" in nodes_visited
    assert "synthesize" in nodes_visited


def test_model_not_found_raises_structured_error():
    import unittest.mock
    client = LLMClient(endpoint="http://localhost:11434", model="non-existent-model")

    with unittest.mock.patch("httpx.Client.post") as mock_post, \
         unittest.mock.patch("httpx.Client.get") as mock_get:

        mock_post.return_value = unittest.mock.MagicMock(
            status_code=404,
            json=lambda: {"error": "model 'non-existent-model' not found"},
        )
        mock_get.return_value = unittest.mock.MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3.2:3b"}, {"name": "qwen3:8b"}]},
        )

        with pytest.raises(ModelNotFoundError) as exc_info:
            client.chat("sys", "user")

        assert exc_info.value.model == "non-existent-model"
        assert exc_info.value.endpoint == "http://localhost:11434"
        assert exc_info.value.available_models == ["llama3.2:3b", "qwen3:8b"]
        assert "llama3.2:3b" in str(exc_info.value)


def test_cli_handles_model_not_found_cleanly(indexed_vault):
    import unittest.mock
    from click.testing import CliRunner
    from alcuin.cli.main import cli

    with unittest.mock.patch(
        "alcuin.agent.graph.LLMClient.chat",
        side_effect=ModelNotFoundError("fake:model", "http://localhost:11434", ["llama3.2:3b", "qwen3:8b"]),
    ):
        runner = CliRunner()
        res = runner.invoke(cli, ["research", "-v", str(indexed_vault.root_path), "What is the address?"])
        assert res.exit_code == 1
        assert "Model 'fake:model' not found" in res.output
        assert "llama3.2:3b" in res.output


def test_subquery_document_title_affinity(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    vault_dir = tmp_path / "affinity_vault"
    vault = Vault.init(vault_dir, name="affinity-test")

    # Note A: Adult Care (uses word "awake", lacks word "sleep")
    doc_a = vault_dir / "Adult_Care_Homes.md"
    doc_a.write_text("# Adult Care Homes\n\nStaff must remain awake on third shift at all times.", encoding="utf-8")

    # Note B: Family Care (uses word "sleep")
    doc_b = vault_dir / "Family_Care_Homes.md"
    doc_b.write_text("# Family Care Homes\n\nOvernight staff may sleep if emergency signaling system is operational.", encoding="utf-8")

    storage = VaultStorage(vault)
    storage.index()

    vault.config.subquery_synthesis = True
    mock_llm = MockLLM(answer="Adult care staff must remain awake.")
    agent = ResearchAgent(vault, llm=mock_llm)

    chunks = [
        {
            "id": "Family_Care_Homes.md#0",
            "file_path": "Family_Care_Homes.md",
            "document_title": "Family Care Homes",
            "section_title": "Family Care Homes",
            "breadcrumbs": "Family Care Homes",
            "content": "Overnight staff may sleep if emergency signaling system is operational.",
        },
        {
            "id": "Adult_Care_Homes.md#0",
            "file_path": "Adult_Care_Homes.md",
            "document_title": "Adult Care Homes",
            "section_title": "Adult Care Homes",
            "breadcrumbs": "Adult Care Homes",
            "content": "Staff must remain awake on third shift at all times.",
        },
    ]

    # Test with sub-questions explicitly referencing adult care and family care
    state = {
        "question": "What are the overnight sleep rules for adult care homes vs family care homes?",
        "sub_questions": [
            "Are overnight staff permitted to sleep in an adult care home?",
            "Are overnight staff permitted to sleep in a family care home?",
        ],
        "evidence": chunks,
        "thought_history": [],
    }

    res = agent._synthesize_node(state)

    # Verify that the LLM received the Adult Care chunk for sub-question 1 due to title affinity
    calls = mock_llm.calls
    assert len(calls) == 2
    first_user_prompt = calls[0][1]
    assert "Adult Care Homes" in first_user_prompt
    assert "awake" in first_user_prompt

    second_user_prompt = calls[1][1]
    assert "Family Care Homes" in second_user_prompt
    assert "sleep" in second_user_prompt
