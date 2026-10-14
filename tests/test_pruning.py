"""
Unit tests for System 1 Link Pruning (SPEC-005).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from alcuin.agent.graph import LLMClient, ResearchAgent
from alcuin.storage.vault import Vault


@pytest.fixture
def test_vault(tmp_path: Path) -> Vault:
    """Create a temporary vault with sample notes containing relevant and distractor wikilinks."""
    vault_dir = tmp_path / "test_pruning_vault"
    vault = Vault.init(vault_dir, name="pruning_vault")
    
    # 1. Main medical note with links
    (vault_dir / "Charlie Profile.md").write_text(
        "# Charlie Profile\n\n"
        "Charlie was born in 2020. For medical care, see [[Dr. Sarah Miller Pediatrics]]. "
        "Also see [[Charlie Pediatric Clinic Log]]. "
        "For vehicle records, see [[Toyota Sienna Maintenance Log]].\n",
        encoding="utf-8",
    )
    # 2. Pediatrician Note
    (vault_dir / "Dr. Sarah Miller Pediatrics.md").write_text(
        "# Dr. Sarah Miller Pediatrics\n\n"
        "Phone: 919-555-0199. Address: 123 Health Ave, Suite 200, Cary, NC.\n",
        encoding="utf-8",
    )
    # 3. Clinic Log
    (vault_dir / "Charlie Pediatric Clinic Log.md").write_text(
        "# Charlie Pediatric Clinic Log\n\n"
        "Annual wellness checkup notes and vaccination status at Cary clinic.\n",
        encoding="utf-8",
    )
    # 4. Car Maintenance Log (Distractor)
    (vault_dir / "Toyota Sienna Maintenance Log.md").write_text(
        "# Toyota Sienna Maintenance Log\n\n"
        "Oil change at 45,000 miles. Tire rotation and brake inspection.\n",
        encoding="utf-8",
    )
    
    # Configure link pruning
    vault.config.link_pruning = True
    vault.config.link_pruning_threshold = 0.40
    vault.config.link_pruning_max_queue = 5
    
    return vault


def test_score_candidate_links_ordering(test_vault: Vault):
    agent = ResearchAgent(vault=test_vault)
    query = "What is the phone number for Charlie's pediatrician?"
    
    candidates = [
        ("Toyota Sienna Maintenance Log", "Oil change at 45,000 miles."),
        ("Dr. Sarah Miller Pediatrics", "Primary pediatrician contact and phone number."),
        ("Charlie Pediatric Clinic Log", "Annual checkup and medical history."),
    ]
    
    scored = agent.score_candidate_links(query, candidates)
    
    assert len(scored) == 3
    # Scored results should be sorted descending by similarity
    assert scored[0][1] >= scored[1][1] >= scored[2][1]
    # The pediatrician link should be highest ranked
    top_link, top_score = scored[0]
    assert "Pediatrics" in top_link
    assert top_score > 0.20


def test_link_pruning_threshold_filtering(test_vault: Vault):
    agent = ResearchAgent(vault=test_vault)
    
    # Mock embedding responses to explicitly test threshold gating
    def mock_embedding(text: str):
        if "pediatrician" in text.lower() or "pediatrics" in text.lower():
            return [1.0, 0.0, 0.0]
        elif "clinic" in text.lower():
            return [0.7, 0.7, 0.0]
        else: # Unrelated distractor (orthogonal vector)
            return [0.0, 1.0, 0.0]
            
    agent.storage.embedder.get_embedding = MagicMock(side_effect=mock_embedding)
    
    query = "What is the pediatrician phone number?"
    candidates = [
        ("Dr. Sarah Miller Pediatrics", "Primary clinic contact"), # Sim = 1.0
        ("Charlie Pediatric Clinic Log", "Visit notes"),          # Sim = 0.707
        ("Toyota Sienna Maintenance Log", "Oil change records"),  # Sim = 0.0
    ]
    
    scored = agent.score_candidate_links(query, candidates)
    threshold = test_vault.config.link_pruning_threshold # 0.40
    
    kept = [l for l, score in scored if score >= threshold]
    pruned = [l for l, score in scored if score < threshold]
    
    assert "Dr. Sarah Miller Pediatrics" in kept
    assert "Charlie Pediatric Clinic Log" in kept
    assert "Toyota Sienna Maintenance Log" in pruned


def test_link_pruning_disabled_mode(test_vault: Vault):
    test_vault.config.link_pruning = False
    agent = ResearchAgent(vault=test_vault)
    
    state = {
        "question": "Where is the clinic?",
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
    }
    
    # Mock search returning a result with a distractor link
    mock_result = MagicMock(
        id="c1",
        file_path="Charlie Profile.md",
        breadcrumbs="Charlie Profile",
        content="Details about [[Toyota Sienna Maintenance Log]].",
        wikilinks=["Toyota Sienna Maintenance Log"],
        score=0.9,
    )
    agent.storage.search = MagicMock(return_value=[mock_result])
    agent._is_in_inquiry_scope = MagicMock(return_value=True)
    
    update = agent._retrieve_node(state)
    # When link_pruning is disabled, distractor is preserved in pending_links
    assert "Toyota Sienna Maintenance Log" in update["pending_links"]


def test_link_pruning_fallback_on_embedding_error(test_vault: Vault):
    agent = ResearchAgent(vault=test_vault)
    
    # Mock embedder raising runtime exception
    agent.storage.embedder.get_embedding = MagicMock(side_effect=RuntimeError("Embedding API timeout"))
    
    candidates = [
        ("Dr. Sarah Miller Pediatrics", "Primary clinic contact"),
        ("Toyota Sienna Maintenance Log", "Oil change records"),
    ]
    
    # Should not raise exception; falls back gracefully
    scored = agent.score_candidate_links("What is the address?", candidates)
    assert len(scored) == 2
    assert all(score == 1.0 for _, score in scored)


def test_link_pruning_max_queue_cap(test_vault: Vault):
    test_vault.config.link_pruning = True
    test_vault.config.link_pruning_threshold = 0.0
    test_vault.config.link_pruning_max_queue = 2
    
    agent = ResearchAgent(vault=test_vault)
    
    state = {
        "question": "What is the pediatrician phone?",
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
    }
    
    mock_result = MagicMock(
        id="c1",
        file_path="Profile.md",
        breadcrumbs="Profile",
        content="See [[Link A]], [[Link B]], and [[Link C]].",
        wikilinks=["Link A", "Link B", "Link C"],
        score=0.9,
    )
    agent.storage.search = MagicMock(return_value=[mock_result])
    agent._is_in_inquiry_scope = MagicMock(return_value=True)
    
    update = agent._retrieve_node(state)
    # Must be capped at max_queue = 2
    assert len(update["pending_links"]) == 2
