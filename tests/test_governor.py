"""
tests/test_governor.py - Unit tests for SPEC-006: Context-Governed Traversal & Sizing Telemetry
"""

from unittest.mock import MagicMock
import pytest
from alcuin.agent.graph import LLMClient, ResearchAgent
from alcuin.chunker.chunker import chunk_document_text
from alcuin.storage.engine import SearchResult
from alcuin.storage.vault import Vault, VaultConfig


@pytest.fixture
def mock_vault(tmp_path):
    vault_dir = tmp_path / "test_governor_vault"
    vault_dir.mkdir()
    config = VaultConfig(
        name="test_gov",
        model_name="llama3.2:3b",
        model_endpoint="http://localhost:11434",
        context_governor=True,
        sibling_traversal=True,
        sibling_pruning_threshold=0.45,
        sibling_max_per_hit=2,
    )
    vault = Vault(root_path=vault_dir, config=config)
    return vault


def test_chunker_wrapped_sentence_no_split():
    """Verify that a wrapped line containing 'Appendix A' inside a sentence does not create a false Schedule heading."""
    text = (
        "Sec. 6-62. Domestic fowl prohibited.\n"
        "(a) Chickens and ducks are banned from sale.\n"
        "(b) Livestock is prohibited except as permitted in the Land Development Ordinance,\n"
        "Appendix A of the Town Code.\n"
        "(c) On-site slaughter of chickens is prohibited.\n"
    )
    chunks = chunk_document_text("Town Code", text)
    # None of the chunks should be falsely titled as a Schedule or Appendix
    for c in chunks:
        assert "Schedule" not in c.section_title
        assert "Appendix" not in c.breadcrumbs
    # Clause (b) must contain both the sentence start and the wrapped Appendix reference
    b_chunk = [c for c in chunks if "(b)" in c.section_title][0]
    assert "Land Development Ordinance," in b_chunk.content
    assert "Appendix A of the Town Code." in b_chunk.content


def test_context_governor_budget_calculation(mock_vault):
    agent = ResearchAgent(vault=mock_vault)

    # 1. 8k local instruct model (llama) -> dynamic 1500 token headroom
    agent.llm.model = "llama3.2:3b"
    agent.llm.endpoint = "http://localhost:11434"
    assert agent.get_model_context_window() == 8192
    assert agent.resolve_model_headroom(8192) == 1500
    assert agent.get_context_governor_budget() == min(8192 - 1500, int(0.65 * 8192))

    # 2. 32k reasoning model (gemma-4-31b) -> dynamic 5000 token headroom
    agent.llm.model = "gemma-4-31b"
    agent.llm.endpoint = "https://generativelanguage.googleapis.com"
    assert agent.get_model_context_window() == 32768
    assert agent.resolve_model_headroom(32768) == 5000
    assert agent.get_context_governor_budget() == min(32768 - 5000, int(0.65 * 32768))

    # 3. 50% safety cap on small context windows (e.g. 4k window caps 5000 headroom at 2000)
    assert agent.resolve_model_headroom(4000) == 2000

    # 4. Custom user-defined headroom override (e.g. 1500 tokens for short-output tasks)
    mock_vault.config.context_headroom_tokens = 1500
    assert agent.resolve_model_headroom(32768) == 1500
    assert agent.get_context_governor_budget() == min(32768 - 1500, int(0.65 * 32768))

    # 5. Explicit max tokens override
    mock_vault.config.context_governor_max_tokens = 3500
    assert agent.get_context_governor_budget() == 3500


def test_context_governor_token_cap_and_telemetry(mock_vault):
    mock_vault.config.context_governor_max_tokens = 500  # Tight budget for testing
    agent = ResearchAgent(vault=mock_vault)

    state = {
        "question": "What are the rules?",
        "iteration": 0,
        "max_iterations": 4,
        "max_hops": 3,
        "current_query": "",
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
        "governor_alert": None,
    }

    # Create 3 large chunks (300 tokens / 1200 chars each)
    mock_results = [
        SearchResult(
            id="doc.txt#1",
            file_path="doc.txt",
            document_title="Doc",
            section_title="Rule 1",
            breadcrumbs="Doc > Rule 1",
            content="A" * 1000,  # ~250 tokens
            wikilinks=[],
            score=0.9,
        ),
        SearchResult(
            id="doc.txt#2",
            file_path="doc.txt",
            document_title="Doc",
            section_title="Rule 2",
            breadcrumbs="Doc > Rule 2",
            content="B" * 1000,  # ~250 tokens
            wikilinks=[],
            score=0.85,
        ),
        SearchResult(
            id="doc.txt#3",
            file_path="doc.txt",
            document_title="Doc",
            section_title="Rule 3",
            breadcrumbs="Doc > Rule 3",
            content="C" * 1000,  # ~250 tokens
            wikilinks=[],
            score=0.80,
        ),
    ]

    agent.storage.search = MagicMock(return_value=mock_results)
    agent.storage.get_chunks_by_ids = MagicMock(return_value=[])

    update = agent._retrieve_node(state)

    # Total tokens in evidence must be <= 500
    ev_tokens = sum(int(len(e["content"]) / 4) for e in update["evidence"])
    assert ev_tokens <= 500
    # Exactly 2 chunks should fit (250 * 2 = 500), 3rd must be pruned
    assert len(update["evidence"]) == 2
    # Alert must be emitted naming the pruned chunk and recommending sizing up
    assert update["governor_alert"] is not None
    assert "[CONTEXT_GOVERNOR_ALERT]" in update["governor_alert"]
    assert "doc.txt#3" in update["governor_alert"]
    assert "Sizing up context window is empirically justified" in update["governor_alert"]


def test_intra_document_sibling_traversal(mock_vault):
    mock_vault.config.context_governor_max_tokens = 5000
    mock_vault.config.sibling_traversal = True
    agent = ResearchAgent(vault=mock_vault)

    state = {
        "question": "chicken slaughter rules",
        "iteration": 0,
        "max_iterations": 4,
        "max_hops": 3,
        "current_query": "",
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
        "governor_alert": None,
    }

    # Primary vector search hit: chunk 1
    mock_primary = [
        SearchResult(
            id="ordinance.txt#1",
            file_path="ordinance.txt",
            document_title="Code",
            section_title="Sec. 6-62 Domestic fowl",
            breadcrumbs="Code > Animals > Sec. 6-62",
            content="Keeping chickens is prohibited in residential zones.",
            wikilinks=[],
            score=0.9,
        )
    ]

    # Sibling chunk: chunk 2 (contains the specific slaughter ban)
    mock_sibling = SearchResult(
        id="ordinance.txt#2",
        file_path="ordinance.txt",
        document_title="Code",
        section_title="Sec. 6-62 Slaughter ban",
        breadcrumbs="Code > Animals > Sec. 6-62",
        content="On-site slaughter of chickens is strictly prohibited.",
        wikilinks=[],
        score=0.85,
    )

    agent.storage.search = MagicMock(return_value=mock_primary)
    agent.storage.get_chunks_by_ids = MagicMock(return_value=[mock_sibling])
    # Mock System 1 scorer to give high similarity to the sibling
    agent.score_candidate_links = MagicMock(return_value=[("ordinance.txt#2", 0.78)])

    update = agent._retrieve_node(state)

    # Both primary hit and relevant sibling chunk must be packed into evidence
    packed_ids = [e["id"] for e in update["evidence"]]
    assert "ordinance.txt#1" in packed_ids
    assert "ordinance.txt#2" in packed_ids


def test_llm_client_truncation_detection_ollama(monkeypatch):
    """Verify that LLMClient correctly flags truncation when Ollama returns done_reason: length."""
    client = LLMClient(endpoint="http://localhost:11434", model="llama3.2:3b")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "llama3.2:3b",
        "done": True,
        "done_reason": "length",
        "message": {"role": "assistant", "content": "Partial answer cut off mid-sent"},
    }

    mock_post = MagicMock(return_value=mock_resp)
    monkeypatch.setattr("httpx.Client.post", mock_post)

    text, meta = client.chat("sys", "user", return_metadata=True)
    assert text == "Partial answer cut off mid-sent"
    assert meta["truncated"] is True
    assert meta["finish_reason"] == "length"
    assert meta["provider"] == "ollama"


def test_llm_client_truncation_detection_google(monkeypatch):
    """Verify that LLMClient correctly flags truncation when Google returns finishReason: MAX_TOKENS."""
    client = LLMClient(
        endpoint="https://generativelanguage.googleapis.com",
        model="gemma-4-31b",
        api_key="test-fake-key",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Partial answer cut off mid-thought"}]},
                "finishReason": "MAX_TOKENS",
            }
        ]
    }

    mock_post = MagicMock(return_value=mock_resp)
    monkeypatch.setattr("httpx.Client.post", mock_post)

    text, meta = client.chat("sys", "user", return_metadata=True)
    assert text == "Partial answer cut off mid-thought"
    assert meta["truncated"] is True
    assert meta["finish_reason"] == "MAX_TOKENS"
    assert meta["provider"] == "google"


def test_synthesize_node_emits_generation_truncation_alert(mock_vault):
    """Verify that _synthesize_node attaches [GENERATION_TRUNCATION_ALERT] when generation is truncated."""
    agent = ResearchAgent(vault=mock_vault)

    # Mock chat to return truncated metadata
    agent.llm.chat = MagicMock(
        return_value=(
            "The civil penalty for the animal violation is $75.00, but regarding housing",
            {"truncated": True, "finish_reason": "length", "provider": "ollama"},
        )
    )

    state = {
        "question": "What is the penalty?",
        "evidence": [
            {
                "id": "chunk#1",
                "file_path": "doc.md",
                "breadcrumbs": "Doc > Sec 1",
                "content": "Penalty is $75.",
            }
        ],
        "thought_history": [],
        "confidence": 0.5,
        "governor_alert": None,
    }

    res = agent._synthesize_node(state)

    assert "[GENERATION_TRUNCATION_ALERT]" in res["final_answer"]
    assert "finish_reason: length" in res["final_answer"]
    assert res["truncation_alert"] is not None
    assert any("[GENERATION_TRUNCATION_ALERT]" in t for t in res["thought_history"])
