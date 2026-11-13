"""Unit tests for SPEC-036 Adaptive Fast-Path Intent Routing."""

import pytest
from unittest.mock import MagicMock
from alcuin.storage.vault import VaultConfig
from alcuin.agent.graph import ResearchAgent, AgentState


class DummyVault:
    def __init__(self, adaptive_fast_path: bool = True, fast_path_max_words: int = 32):
        self.config = VaultConfig(
            adaptive_fast_path=adaptive_fast_path,
            fast_path_max_words=fast_path_max_words,
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
