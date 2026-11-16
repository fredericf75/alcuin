"""Unit tests for SPEC-036 Adaptive Fast-Path Intent Routing."""

import pytest
from unittest.mock import MagicMock
from alcuin.storage.vault import VaultConfig
from alcuin.agent.graph import ResearchAgent, AgentState


class DummyVault:
    def __init__(
        self,
        adaptive_fast_path: bool = True,
        fast_path_max_words: int = 32,
        adaptive_fast_path_semantic: bool = True,
        fast_path_embedding_threshold: float = 0.0,
    ):
        self.config = VaultConfig(
            adaptive_fast_path=adaptive_fast_path,
            fast_path_max_words=fast_path_max_words,
            adaptive_fast_path_semantic=adaptive_fast_path_semantic,
            fast_path_embedding_threshold=fast_path_embedding_threshold,
        )


def test_atomic_inquiry_classification():
    vault = DummyVault()
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault

    atomic_queries = [
        "What is the statutory deadline for filing an appeal after notice of discharge?",
        "When is monthly base rent due under the lease?",
        "What is the minimum age requirement for a personal care aide in North Carolina?",
        "How many female chickens can a Cary resident keep in their backyard?",
        "What is Apple's total right-of-use asset balance in the 2026 Form 10-K?",
        "Under what conditions may a contractor give notice of differing site conditions?",
        "What HTTP status code is required when a method is not supported?",
        "Is mediation mandatory before arbitration under AIA A201?",
        "What is the required frequency for fire drills in adult care homes?",
        "What is the penalty for a second animal violation in Cary?",
    ]

    for q in atomic_queries:
        assert agent.is_atomic_inquiry(q) is True, f"Failed on atomic query: {q}"


def test_composite_inquiry_rejection():
    vault = DummyVault()
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault

    composite_queries = [
        # Multiple question marks
        "What is base rent? And what late fee applies?",
        # Semicolon
        "What is the grace period; and what is the late penalty?",
        # List / bullet markers
        "1. How many hens can I keep? 2. Are roosters permitted?",
        "- What is the setback? - What is the fine?",
        # Comparative markers
        "Compare the staffing ratios between 13F and 13G facilities.",
        "What is the difference between safe and idempotent HTTP methods?",
        "Should we choose arbitration versus mediation under AIA A201?",
        # Compound conjunctions
        "What are the aide hours as well as the night shift rules?",
        "What is the base rent in addition to the late fee percentage?",
        "Explain the right-of-use assets along with the lease liabilities.",
        # Multi-question conjunctions
        "What is the grace period and what late fee applies?",
        "How many hens are permitted and can we keep roosters?",
        "When is rent due and how should payment be delivered?",
        "What is the admission protocol and who performs the physical exam?",
        # Word count > 32
        (
            "A resident in Cary wants to keep chickens in their backyard and needs to know the exact "
            "rules regarding coops, setbacks from adjacent property lines, and previous violation penalties "
            "under local town ordinances and state administrative regulations."
        ),
    ]

    for q in composite_queries:
        assert agent.is_atomic_inquiry(q) is False, f"Failed to reject composite query: {q}"


def test_adaptive_fast_path_config_toggle():
    vault = DummyVault(adaptive_fast_path=False)
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault

    q = "When is base rent due?"
    assert agent.is_atomic_inquiry(q) is False


def test_decompose_query_fast_path_bypasses_llm():
    vault = DummyVault(adaptive_fast_path=True)
    mock_llm = MagicMock()
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault
    agent.planner_llm = mock_llm

    atomic_q = "When is base rent due under the lease?"
    sgs, sqs = agent.decompose_query(atomic_q)

    assert sgs == [atomic_q]
    assert sqs == [atomic_q]
    # Verify planner LLM was NEVER called
    mock_llm.chat.assert_not_called()


def test_decompose_query_composite_invokes_llm():
    vault = DummyVault(adaptive_fast_path=True)
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "1. What is base rent?\n2. What late fee applies?"
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault
    agent.planner_llm = mock_llm

    composite_q = (
        "A tenant signed a lease and wants to know when base rent is due, "
        "and what late fee applies if they pay after the grace period?"
    )
    sgs, sqs = agent.decompose_query(composite_q)

    # Verify planner LLM was called because query is composite
    mock_llm.chat.assert_called_once()
    assert len(sqs) >= 2


def test_short_composite_invokes_llm():
    """SPEC-038: Short composite queries (<25 words) must not be aborted by legacy length guard."""
    vault = DummyVault(adaptive_fast_path=True)
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "1. What is Charlie's insurance?\n2. Who has custody?"
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault
    agent.planner_llm = mock_llm

    short_composite = "Send Charlie ER info along with insurance and custody"
    sgs, sqs = agent.decompose_query(short_composite)

    mock_llm.chat.assert_called_once()
    assert len(sqs) == 2


def test_semantic_embedding_routes_implicit_composite():
    """SPEC-038: Zero-shot embedding centroid routes implicit multi-domain queries to planner."""
    vault = DummyVault(adaptive_fast_path=True, adaptive_fast_path_semantic=True)
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "1. What is Charlie's insurance?\n2. What are Charlie's allergies?"

    mock_embedder = MagicMock()
    # Prototype vectors: atomic = [1, 0], composite = [0, 1]
    def fake_batch(texts, is_query=True):
        if any("single specific fact" in t for t in texts):
            return [[1.0, 0.0] for _ in texts]
        return [[0.0, 1.0] for _ in texts]

    def fake_single(text, is_query=True):
        if "ER" in text:
            return [0.1, 0.99]  # Closer to composite centroid
        return [0.99, 0.1]      # Closer to atomic centroid

    mock_embedder.get_embeddings_batch.side_effect = fake_batch
    mock_embedder.get_embedding.side_effect = fake_single

    mock_storage = MagicMock()
    mock_storage.embedder = mock_embedder

    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault
    agent.storage = mock_storage
    agent.planner_llm = mock_llm

    # 1. Atomic lookup stays on fast path
    assert agent.is_atomic_inquiry("What is Charlie's pediatrician phone number?") is True
    agent.decompose_query("What is Charlie's pediatrician phone number?")
    mock_llm.chat.assert_not_called()

    # 2. Implicit multi-domain ER query routes to planner
    er_query = "I need to check charlie into the ER send me all necessary info"
    assert agent.is_atomic_inquiry(er_query) is False
    sgs, sqs = agent.decompose_query(er_query)
    mock_llm.chat.assert_called_once()
    assert len(sqs) == 2


def test_semantic_fast_path_toggle():
    """SPEC-038: Disabling adaptive_fast_path_semantic falls back strictly to syntactic rules."""
    vault = DummyVault(adaptive_fast_path=True, adaptive_fast_path_semantic=False)
    mock_storage = MagicMock()
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.vault = vault
    agent.storage = mock_storage

    er_query = "I need to check charlie into the ER send me all necessary info"
    # Without semantic gate, ER query has no syntactic markers so returns True
    assert agent.is_atomic_inquiry(er_query) is True
    mock_storage.embedder.get_embedding.assert_not_called()
