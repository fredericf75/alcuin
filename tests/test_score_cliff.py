from alcuin.agent.graph import ResearchAgent
from alcuin.storage.engine import SearchResult
from alcuin.storage.vault import Vault


def _make_result(cid: str, score: float) -> SearchResult:
    return SearchResult(
        id=cid,
        file_path="test.md",
        document_title="Test Doc",
        section_title="Test Section",
        breadcrumbs="Test > Section",
        content="Test content",
        wikilinks=[],
        score=score,
    )


def test_score_cliff_singular_fact(tmp_path):
    vault = Vault.init(tmp_path / "v1", name="v1")
    agent = ResearchAgent(vault=vault)

    # Singular fact pattern: [0.88, 0.50, 0.45, 0.40]
    hits = [
        _make_result("c1", 0.88),
        _make_result("c2", 0.50),
        _make_result("c3", 0.45),
        _make_result("c4", 0.40),
    ]
    pruned = agent.prune_by_score_cliff(hits)
    assert len(pruned) == 1
    assert pruned[0].id == "c1"


def test_score_cliff_multi_clause(tmp_path):
    vault = Vault.init(tmp_path / "v2", name="v2")
    agent = ResearchAgent(vault=vault)

    # Multi-clause pattern: [0.86, 0.84, 0.82, 0.45]
    hits = [
        _make_result("c1", 0.86),
        _make_result("c2", 0.84),
        _make_result("c3", 0.82),
        _make_result("c4", 0.45),
    ]
    pruned = agent.prune_by_score_cliff(hits)
    assert len(pruned) == 3
    assert [p.id for p in pruned] == ["c1", "c2", "c3"]


def test_score_cliff_flat_cluster(tmp_path):
    vault = Vault.init(tmp_path / "v3", name="v3")
    agent = ResearchAgent(vault=vault)

    # Flat cluster with no cliff: [0.85, 0.84, 0.83, 0.82]
    hits = [
        _make_result("c1", 0.85),
        _make_result("c2", 0.84),
        _make_result("c3", 0.83),
        _make_result("c4", 0.82),
    ]
    pruned = agent.prune_by_score_cliff(hits)
    assert len(pruned) == 4


def test_score_cliff_below_guardrail(tmp_path):
    vault = Vault.init(tmp_path / "v4", name="v4")
    agent = ResearchAgent(vault=vault)

    # Low noise plateau: [0.38, 0.35, 0.32] -> Below 0.40 guardrail
    hits = [
        _make_result("c1", 0.38),
        _make_result("c2", 0.35),
        _make_result("c3", 0.32),
    ]
    pruned = agent.prune_by_score_cliff(hits)
    assert len(pruned) == 0
