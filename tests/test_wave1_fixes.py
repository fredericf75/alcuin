"""Regression tests for Wave 1 code review fixes (security, data loss, outages)."""

import json
import re
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from alcuin.api.server import HTML_DASHBOARD, app
from alcuin.connectors.gdrive import sanitize_name, sync_gdrive_folder
from alcuin.storage.curator import (
    apply_entity_merge,
    collect_merge_candidates,
    load_curated_overrides,
)
from alcuin.storage.engine import CHUNKS_TABLE_NAME, VaultStorage
from alcuin.storage.gaps import KnowledgeGapTracker
from alcuin.storage.linter import VaultLinter
from alcuin.storage.vault import Vault, VaultConfig, VaultRegistry


def test_dashboard_no_duplicate_latest_finding_and_no_inline_onclick_xss():
    assert HTML_DASHBOARD.count("let latestFinding = null;") == 1
    assert "onclick=\"resolveGap(" not in HTML_DASHBOARD
    assert "onclick=\"setDefaultVault(" not in HTML_DASHBOARD
    assert "onclick=\"deleteVault(" not in HTML_DASHBOARD
    assert "g.last_confidence" in HTML_DASHBOARD
    assert "info.chunk_count" in HTML_DASHBOARD


def test_create_note_blocks_traversal_before_mkdir_and_blocks_alcuin_dir(tmp_path, monkeypatch):
    vault_dir = tmp_path / "vault_a"
    Vault.init(vault_dir, name="vault_a")
    monkeypatch.setattr(VaultRegistry, "get_vault_path", classmethod(lambda cls, name: vault_dir if name == "vault_a" else None))

    client = TestClient(app)
    evil_dir = tmp_path / "escaped_dir"
    resp = client.post(
        "/api/v1/vaults/vault_a/notes",
        json={"title": "Bad", "content": "nope", "subfolder": "../escaped_dir"},
    )
    assert resp.status_code == 400
    assert not evil_dir.exists()

    resp2 = client.post(
        "/api/v1/vaults/vault_a/notes",
        json={"title": "Meta", "content": "nope", "subfolder": ".alcuin"},
    )
    assert resp2.status_code == 400


def test_lancedb_open_error_does_not_overwrite_existing_table(tmp_path):
    vault = Vault.init(tmp_path / "v_lance")
    storage = VaultStorage(vault)
    # Simulate a transient open_table error when table already exists
    storage.db = MagicMock()
    storage.db.list_tables.return_value = [CHUNKS_TABLE_NAME]
    storage.db.open_table.side_effect = OSError("Transient file lock")
    with pytest.raises(OSError, match="Transient file lock"):
        storage._get_or_create_table()
    storage.db.create_table.assert_not_called()


def test_entity_merge_preserves_multivalue_conflicts_overrides_and_chained_aliases(tmp_path):
    vault = Vault.init(tmp_path / "v_merge")
    people = vault.root_path / "_curated" / "people"
    people.mkdir(parents=True)

    (people / "Jane Maiden.md").write_text(
        "# Jane Maiden — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Date of Birth**: 1990-01-01 *(Source: `id.pdf`)*\n"
        "- **Key Date**: 2010-05-01 *(Source: `doc1.pdf`)*\n"
        "- **Key Date**: 2015-06-02 *(Source: `doc2.pdf`)*\n"
        "- **Address**: 100 Old St *(Source: `doc1.pdf`)*\n",
        encoding="utf-8",
    )
    (people / "Jane Married.md").write_text(
        "# Jane Married — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Date of Birth**: 1990-01-01 *(Source: `id.pdf`)*\n"
        "- **Key Date**: 2020-07-03 *(Source: `doc3.pdf`)*\n"
        "- **Address**: 200 New Ave *(Source: `doc3.pdf`)*\n",
        encoding="utf-8",
    )
    # Pre-existing override on Jane Maiden + pre-existing alias pointing to Jane Maiden
    ov_path = vault.root_path / ".alcuin" / "curated_overrides.json"
    ov_path.write_text(
        json.dumps(
            {
                "version": 1,
                "overrides": {"people/Jane Maiden::Date of Birth": {"value": "1990-01-01", "source": "Human Verified"}},
                "aliases": {"janie": "Jane Maiden"},
                "not_merges": [],
            }
        ),
        encoding="utf-8",
    )

    cands = collect_merge_candidates(vault.root_path)
    assert len(cands) == 1
    out = apply_entity_merge(vault.root_path, cands[0], "Jane Married")
    txt = out.read_text(encoding="utf-8")

    # All 3 Key Dates preserved
    assert "2010-05-01" in txt
    assert "2015-06-02" in txt
    assert "2020-07-03" in txt
    # Differing single-value Address preserved in Conflicting section
    assert "## Conflicting / Alternate Mentions" in txt
    assert "200 New Ave" in txt

    ov = load_curated_overrides(vault.root_path)
    assert "people/Jane Married::Date of Birth" in ov["overrides"]
    assert "people/Jane Maiden::Date of Birth" not in ov["overrides"]
    # Chained alias updated directly to Jane Married
    assert ov["aliases"]["janie"] == "Jane Married"


def test_gdrive_sanitize_blocks_alcuin_and_prune_blocks_traversal(tmp_path):
    assert sanitize_name(".alcuin") == "_alcuin"
    assert sanitize_name(".git") == "_git"
    assert sanitize_name("..") == "untitled"

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("keep me", encoding="utf-8")

    state_path = vault_dir / ".alcuin" / "gdrive_sync_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"bad_id": {"local_path": "../outside_secret.txt"}}),
        encoding="utf-8",
    )

    fake_service = MagicMock()
    fake_service.files.return_value.list.return_value.execute.return_value = {"files": []}
    report = sync_gdrive_folder(fake_service, "root_id", vault_dir, prune=True)
    assert outside.exists()
    assert report.pruned == []


def test_linter_blocks_host_filesystem_probing_via_wikilink(tmp_path):
    vault = Vault.init(tmp_path / "v_lint")
    secret = tmp_path / "outside.md"
    secret.write_text("# Secret", encoding="utf-8")
    linter = VaultLinter(vault)
    empty_targets = {"stems_lower": set(), "exact_paths": set(), "titles_lower": set()}
    assert linter._resolve_wikilink_target("../outside.md", "note.md", empty_targets) is False
    assert linter._resolve_wikilink_target(str(secret), "note.md", empty_targets) is False


def test_gaps_resolve_punctuation_only_does_not_resolve_all(tmp_path):
    vault = Vault.init(tmp_path / "v_gaps")
    tracker = KnowledgeGapTracker(vault)
    tracker.record_query("What is the statutory deadline?", 0.0, 0)
    tracker.record_query("What is the filing fee?", 0.0, 0)

    assert tracker.resolve_gap("???") is False
    assert tracker.resolve_gap("a") is False
    assert len([g for g in tracker.list_gaps() if g.status == "unresolved"]) == 2


def test_vault_config_toml_roundtrip_escapes_reranker_and_preserves_glossary(tmp_path):
    cfg = VaultConfig(
        name="win_vault",
        reranker_model=r"C:\models\cross-encoder",
        auto_ingest_inbox=False,
        inbox_check_interval_seconds=15,
        glossary={"ACH": "Adult Care Home", "SCU": "Special Care Unit"},
    )
    toml_str = cfg.to_toml()
    parsed = tomllib.loads(toml_str)
    reloaded = VaultConfig.from_dict(parsed)
    assert reloaded.reranker_model == r"C:\models\cross-encoder"
    assert reloaded.auto_ingest_inbox is False
    assert reloaded.inbox_check_interval_seconds == 15
    assert reloaded.glossary == {"ACH": "Adult Care Home", "SCU": "Special Care Unit"}
