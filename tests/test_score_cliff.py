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


def test_score_cliff_structural_identifier_rescue_ignores_body_mentions(tmp_path):
    vault = Vault.init(tmp_path / "v5", name="v5")
    agent = ResearchAgent(vault=vault)

    hits = [
        SearchResult(
            id="c1",
            file_path="code.md",
            document_title="Town Code",
            section_title="Sec. 6-10: Overview",
            breadcrumbs="Town Code > Sec. 6-10: Overview",
            content="General animal control rules.",
            wikilinks=[],
            score=0.78,
        ),
        SearchResult(
            id="c2",
            file_path="code.md",
            document_title="Town Code",
            section_title="Sec. 6-28: Tethering",
            breadcrumbs="Town Code > Sec. 6-28: Tethering",
            content="Violations of tethering are subject to penalties under Sec. 6-132.",
            wikilinks=["Sec. 6-132"],
            score=0.48,
        ),
        SearchResult(
            id="c3",
            file_path="code.md",
            document_title="Town Code",
            section_title="Sec. 6-132: Escalating Penalties",
            breadcrumbs="Town Code > Sec. 6-132: Escalating Penalties",
            content="1st offense: $20. 2nd offense: $50.",
            wikilinks=[],
            score=0.44,
        ),
    ]

    # Query targets Sec. 6-132: c2 only mentions 6-132 in body prose (must NOT be rescued),
    # whereas c3 has 6-132 in section_title (MUST be rescued).
    pruned = agent.prune_by_score_cliff(hits, query="Sec. 6-132")
    admitted_ids = [p.id for p in pruned]
    assert "c1" in admitted_ids
    assert "c3" in admitted_ids
    assert "c2" not in admitted_ids


def test_score_cliff_split_section_and_table_rescue(tmp_path):
    vault = Vault.init(tmp_path / "v6", name="v6")
    agent = ResearchAgent(vault=vault)

    hits = [
        SearchResult(
            id="c1",
            file_path="code.md",
            document_title="Town Code",
            section_title="Sec. 6-132: Penalties (Part 1)",
            breadcrumbs="Town Code > Sec. 6-132: Penalties",
            content="Civil penalties for animal control violations shall escalate by offense.",
            wikilinks=[],
            score=0.84,
        ),
        SearchResult(
            id="c2",
            file_path="code.md",
            document_title="Town Code",
            section_title="Sec. 6-132: Penalties (Part 2)",
            breadcrumbs="Town Code > Sec. 6-132: Penalties",
            content="| Offense | Fine |\n| 1st | $20 |\n<!-- TABLE_DATA:\n{\"1st\": \"$20\"}\n-->",
            wikilinks=[],
            score=0.45,
        ),
        SearchResult(
            id="c3",
            file_path="code.md",
            document_title="Town Code",
            section_title="Sec. 8-10: Housing",
            breadcrumbs="Town Code > Sec. 8-10: Housing",
            content="Unrelated housing clause.",
            wikilinks=[],
            score=0.42,
        ),
    ]

    # Even without section number in query, c2 is the companion TABLE_DATA chunk for admitted section c1
    pruned = agent.prune_by_score_cliff(hits, query="escalating fine schedule for animal violations")
    admitted_ids = [p.id for p in pruned]
    assert admitted_ids == ["c1", "c2"]


def test_score_cliff_graph_hop_title_rescue(tmp_path):
    vault = Vault.init(tmp_path / "v7", name="v7")
    agent = ResearchAgent(vault=vault)

    # Multi-file vault where fallback or short hop query scores < 0.40 in vector space
    hits = [
        SearchResult(
            id="c1",
            file_path="pediatrician.md",
            document_title="Dr. Sarah Miller Pediatrics",
            section_title="Clinic Contact",
            breadcrumbs="Dr. Sarah Miller Pediatrics > Clinic Contact",
            content="Phone: 555-0199.",
            wikilinks=[],
            score=0.22,
        ),
        SearchResult(
            id="c2",
            file_path="car.md",
            document_title="Toyota Maintenance",
            section_title="Oil Change",
            breadcrumbs="Toyota Maintenance > Oil Change",
            content="Changed oil at 45k miles.",
            wikilinks=[],
            score=0.15,
        ),
    ]

    pruned = agent.prune_by_score_cliff(hits, query="Dr. Sarah Miller Pediatrics")
    assert [p.id for p in pruned] == ["c1"]

