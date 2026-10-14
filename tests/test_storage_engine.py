"""Unit tests for LanceDB VaultStorage engine and incremental ingestion."""

import os
from pathlib import Path
import pytest
from alcuin.storage.vault import Vault
from alcuin.storage.engine import VaultStorage, extract_wikilinks


@pytest.fixture
def temp_vault(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    vault_dir = tmp_path / "docs_vault"
    vault = Vault.init(vault_dir, name="engine-test")
    return vault


def test_extract_wikilinks():
    text = "Refer to [[pediatrician]] and [[Charlie's School|school]] for details."
    links = extract_wikilinks(text)
    assert "pediatrician" in links
    assert "Charlie's School" in links


def test_incremental_indexing_lifecycle(temp_vault):
    storage = VaultStorage(temp_vault)

    # 1. Add first document
    doc1 = temp_vault.root_path / "pediatrician.md"
    doc1.write_text(
        "# Dr. Smith Pediatrics\n\nAddress: 123 Health Ave, Suite 400.\nPhone: 555-0199.\nSee [[Vaccines]]",
        encoding="utf-8",
    )

    report1 = storage.index()
    assert "pediatrician.md" in report1.added_files
    assert report1.total_chunks_indexed > 0
    assert storage.table.count_rows() > 0

    # 2. Re-indexing without changes should detect unchanged
    report2 = storage.index()
    assert "pediatrician.md" in report2.unchanged_files
    assert len(report2.added_files) == 0
    assert len(report2.modified_files) == 0

    # 3. Add second document and modify first
    doc2 = temp_vault.root_path / "vaccines.md"
    doc2.write_text(
        "# Vaccine Schedule\n\nAdministered by [[Dr. Smith Pediatrics]]. Next due: October.",
        encoding="utf-8",
    )
    doc1.write_text(
        "# Dr. Smith Pediatrics\n\nAddress: 123 Health Ave, Suite 500 (Relocated).\nSee [[Vaccines]]",
        encoding="utf-8",
    )

    report3 = storage.index()
    assert "vaccines.md" in report3.added_files
    assert "pediatrician.md" in report3.modified_files

    # 4. Search verification
    results = storage.search("pediatrician address", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert "Health Ave" in top.content
    assert "Vaccines" in top.wikilinks

    # 5. Wikilink graph resolution
    links = storage.get_linked_pages("Vaccines")
    assert "pediatrician.md" in links

    # 6. Delete document
    doc2.unlink()
    report4 = storage.index()
    assert "vaccines.md" in report4.deleted_files
