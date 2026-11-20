"""Unit tests for SPEC-043 Assisted Curation (--review) & Knowledge Gap Resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from alcuin.cli.main import cli
from alcuin.storage.curator import (
    VaultCurator,
    apply_conflict_resolution,
    apply_entity_merge,
    collect_conflicts,
    collect_merge_candidates,
    collect_open_gaps,
    load_curated_overrides,
    record_not_merge,
    resolve_gap,
)
from alcuin.storage.vault import Vault


def _setup_vault_with_conflicts_and_gaps(tmp_path: Path) -> Vault:
    v = Vault.init(tmp_path, name="assist_test")
    people_dir = tmp_path / "_curated" / "people"
    people_dir.mkdir(parents=True, exist_ok=True)
    (people_dir / "Ava Emick.md").write_text(
        "# Ava Emick — Canonical Profile\n\n"
        "> Curated automatically by Alcuin (`SPEC-041`). Formal records take precedence over informal mentions.\n\n"
        "## Canonical Facts\n"
        "- **Date of Birth**: 1990-11-12 *(Source: `scan1.pdf`)*\n\n"
        "## Conflicting / Alternate Mentions\n"
        "- **Date of Birth**: `2009-11-12` *(Source: `scan2.pdf`)* — differs from canonical `1990-11-12`\n",
        encoding="utf-8",
    )
    (people_dir / "Daniel.md").write_text(
        "# Daniel — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Address**: 100 Old Rd *(Source: `doc1.md`)*\n\n"
        "## Conflicting / Alternate Mentions\n"
        "- **Address**: `200 New Rd` *(Source: `doc2.md`)* — differs from canonical `100 Old Rd`\n",
        encoding="utf-8",
    )
    gaps_data = {
        "version": 1,
        "updated_at": "2026-11-19T00:00:00+00:00",
        "gaps": {
            "marriage_date_1": {
                "id": "marriage_date_1",
                "query": "When did Tara and Daniel get married?",
                "status": "unresolved",
                "ask_count": 2,
            }
        },
    }
    (tmp_path / ".alcuin" / "knowledge_gaps.json").write_text(
        json.dumps(gaps_data, indent=2), encoding="utf-8"
    )
    return v


def test_collect_and_resolve_conflict_and_gap(tmp_path):
    v = _setup_vault_with_conflicts_and_gaps(tmp_path)
    conflicts = collect_conflicts(v.root_path)
    assert len(conflicts) == 2

    ava = next(c for c in conflicts if c["entity"] == "Ava Emick")
    apply_conflict_resolution(v.root_path, ava, "2009-11-12", "scan2.pdf")

    # Immediately persisted in overrides & removed from remaining conflicts
    overrides = load_curated_overrides(v.root_path)["overrides"]
    assert overrides["people/Ava Emick::Date of Birth"]["value"] == "2009-11-12"
    remaining = collect_conflicts(v.root_path)
    assert len(remaining) == 1
    assert remaining[0]["entity"] == "Daniel"

    # Profile card updated and empty Conflicting section stripped
    ava_md = (tmp_path / "_curated" / "people" / "Ava Emick.md").read_text(encoding="utf-8")
    assert "2009-11-12" in ava_md
    assert "Conflicting / Alternate Mentions" not in ava_md

    # Resolve open knowledge gap
    gaps = collect_open_gaps(v.root_path)
    assert len(gaps) == 1
    assert resolve_gap(v.root_path, "marriage_date_1", answer="November 21, 2008") is True
    assert collect_open_gaps(v.root_path) == []
    assert "November 21, 2008" in (tmp_path / "_curated" / "verified_facts.md").read_text(encoding="utf-8")


def test_curate_review_cli_quit_after_one_answer_persists(tmp_path, monkeypatch):
    v = _setup_vault_with_conflicts_and_gaps(tmp_path)

    # Avoid loading real embedder during test re-index
    monkeypatch.setattr("alcuin.cli.main.VaultStorage", lambda vault_obj: MagicMock(
        index=lambda: MagicMock(duration_seconds=0.1, added_files=[], modified_files=[], total_chunks_indexed=0)
    ))

    runner = CliRunner()
    # Select section [1] Conflicts, pick option [2] for Ava Emick, then [q] on Daniel
    res = runner.invoke(cli, ["curate", "-v", str(tmp_path), "--review"], input="1\n2\nq\n")
    assert res.exit_code == 0, res.output
    assert "Resolved this session: 1 conflicts" in res.output
    assert "Remaining backlog:     1 conflicts" in res.output

    # Ava persisted, Daniel untouched for next session
    remaining = collect_conflicts(v.root_path)
    assert len(remaining) == 1
    assert remaining[0]["entity"] == "Daniel"


def test_overrides_preserved_on_subsequent_curate(tmp_path):
    v = _setup_vault_with_conflicts_and_gaps(tmp_path)
    ava = next(c for c in collect_conflicts(v.root_path) if c["entity"] == "Ava Emick")
    apply_conflict_resolution(v.root_path, ava, "2009-11-12", "Human Verified")

    curator = VaultCurator(v)
    state = {
        "chunk_1": {
            "hash": "abc",
            "file_path": "scan1.pdf",
            "authority": 2,
            "facts": [{"entity": "Ava Emick", "type": "person", "field": "Date of Birth", "value": "1990-11-12"}],
        },
        "chunk_2": {
            "hash": "def",
            "file_path": "scan2.pdf",
            "authority": 1,
            "facts": [{"entity": "Ava Emick", "type": "person", "field": "Date of Birth", "value": "2009-11-12"}],
        },
    }
    profiles, conflicts_count = curator._consolidate_profiles(state)
    assert conflicts_count == 0
    assert "2009-11-12" in profiles["people/Ava Emick.md"]
    assert "Conflicting / Alternate Mentions" not in profiles["people/Ava Emick.md"]


def test_entity_merge_detection_and_alias_persistence(tmp_path):
    v = Vault.init(tmp_path, name="merge_test")
    people_dir = tmp_path / "_curated" / "people"
    people_dir.mkdir(parents=True, exist_ok=True)
    (people_dir / "Tara Marie Emick.md").write_text(
        "# Tara Marie Emick — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Date of Birth**: 1983-07-23 *(Source: `license.pdf`)*\n"
        "- **Email**: tara@example.com *(Source: `notes.md`)*\n",
        encoding="utf-8",
    )
    (people_dir / "Tara Marie Matthews.md").write_text(
        "# Tara Marie Matthews — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Date of Birth**: 1983-07-23 *(Source: `marriage.pdf`)*\n",
        encoding="utf-8",
    )

    cands = collect_merge_candidates(v.root_path)
    assert len(cands) == 1
    assert "Shared Date of Birth: 1983-07-23" in cands[0]["reason"]

    # Merge into Tara Marie Emick
    out_path = apply_entity_merge(v.root_path, cands[0], "Tara Marie Emick")
    assert out_path.is_file()
    assert not (people_dir / "Tara Marie Matthews.md").exists()
    merged_md = out_path.read_text(encoding="utf-8")
    assert "Aliases / Former Names" in merged_md
    assert "Tara Marie Matthews" in merged_md
    assert "tara@example.com" in merged_md

    # Future curation remaps Tara Marie Matthews -> Tara Marie Emick and keeps alias
    curator = VaultCurator(v)
    state = {
        "c1": {
            "hash": "h1",
            "file_path": "marriage.pdf",
            "authority": 2,
            "facts": [{"entity": "Tara Marie Matthews", "type": "person", "field": "Date of Birth", "value": "1983-07-23"}],
        },
        "c2": {
            "hash": "h2",
            "file_path": "notes.md",
            "authority": 1,
            "facts": [{"entity": "Tara Marie Emick", "type": "person", "field": "Email", "value": "tara@example.com"}],
        },
    }
    profiles, _ = curator._consolidate_profiles(state)
    assert "people/Tara Marie Matthews.md" not in profiles
    assert "people/Tara Marie Emick.md" in profiles
    assert "Tara Marie Matthews" in profiles["people/Tara Marie Emick.md"]


def test_record_not_merge_suppresses_pair(tmp_path):
    v = Vault.init(tmp_path, name="not_merge_test")
    people_dir = tmp_path / "_curated" / "people"
    people_dir.mkdir(parents=True, exist_ok=True)
    (people_dir / "John Smith.md").write_text(
        "# John Smith — Canonical Profile\n\n## Canonical Facts\n- **Phone**: 555-0100 *(Source: `a.md`)*\n",
        encoding="utf-8",
    )
    (people_dir / "John Smith Jr.md").write_text(
        "# John Smith Jr — Canonical Profile\n\n## Canonical Facts\n- **Phone**: 555-0100 *(Source: `b.md`)*\n",
        encoding="utf-8",
    )
    cands = collect_merge_candidates(v.root_path)
    assert len(cands) == 1
    record_not_merge(v.root_path, cands[0]["a"]["rel_path"], cands[0]["b"]["rel_path"])
    assert collect_merge_candidates(v.root_path) == []


def test_keep_all_multi_value_conflict_and_persistence(tmp_path):
    from alcuin.storage.curator import apply_keep_all_resolution

    v = Vault.init(tmp_path, name="keep_all_test")
    people_dir = tmp_path / "_curated" / "people"
    people_dir.mkdir(parents=True, exist_ok=True)
    (people_dir / "Frederic Faizon.md").write_text(
        "# Frederic Faizon — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Case Number**: 26CVD3578-910 *(Source: `qdro.pdf`)*\n\n"
        "## Conflicting / Alternate Mentions\n"
        "- **Case Number**: `24CVD00123` *(Source: `custody.pdf`)* — differs from canonical `26CVD3578-910`\n",
        encoding="utf-8",
    )
    conflicts = collect_conflicts(v.root_path)
    assert len(conflicts) == 1
    apply_keep_all_resolution(v.root_path, conflicts[0])

    assert collect_conflicts(v.root_path) == []
    md = (people_dir / "Frederic Faizon.md").read_text(encoding="utf-8")
    assert "26CVD3578-910" in md and "24CVD00123" in md
    assert "Conflicting / Alternate Mentions" not in md

    curator = VaultCurator(v)
    state = {
        "c1": {
            "hash": "h1",
            "file_path": "qdro.pdf",
            "authority": 2,
            "facts": [{"entity": "Frederic Faizon", "type": "person", "field": "Case Number", "value": "26CVD3578-910"}],
        },
        "c2": {
            "hash": "h2",
            "file_path": "custody.pdf",
            "authority": 2,
            "facts": [
                {"entity": "Frederic Faizon", "type": "person", "field": "Case Number", "value": "24CVD00123"},
                {"entity": "Frederic Faizon", "type": "person", "field": "Date of Birth", "value": "1975-08-14"},
            ],
        },
    }
    profiles, count = curator._consolidate_profiles(state)
    assert count == 0
    assert "26CVD3578-910" in profiles["people/Frederic Faizon.md"]
    assert "24CVD00123" in profiles["people/Frederic Faizon.md"]


def test_multi_value_fields_do_not_raise_conflicts_by_default(tmp_path):
    v = Vault.init(tmp_path, name="mv_default_test")
    curator = VaultCurator(v)
    state = {
        "c1": {
            "hash": "h1",
            "file_path": "qdro.pdf",
            "authority": 2,
            "facts": [
                {"entity": "Frederic Faizon", "type": "person", "field": "Key Date", "value": "2004-03-05"},
                {"entity": "Frederic Faizon", "type": "person", "field": "Date of Birth", "value": "1975-08-14"},
            ],
        },
        "c2": {
            "hash": "h2",
            "file_path": "separation.pdf",
            "authority": 2,
            "facts": [{"entity": "Frederic Faizon", "type": "person", "field": "Key Date", "value": "2024-11-01"}],
        },
    }
    profiles, count = curator._consolidate_profiles(state)
    assert count == 0
    assert "2004-03-05" in profiles["people/Frederic Faizon.md"]
    assert "2024-11-01" in profiles["people/Frederic Faizon.md"]
    assert "Conflicting / Alternate Mentions" not in profiles["people/Frederic Faizon.md"]
