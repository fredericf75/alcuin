"""Unit tests for VaultLinter diagnostic and remediation engine."""

import json
from pathlib import Path
from click.testing import CliRunner
import pytest

from alcuin.cli.main import cli
from alcuin.storage.engine import VaultStorage
from alcuin.storage.linter import LintCategory, LintSeverity, VaultLinter
from alcuin.storage.vault import Vault


@pytest.fixture
def test_vault(tmp_path, monkeypatch):
    """Fixture providing an isolated vault and global config."""
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "test_vault"
    v_dir.mkdir()

    # Create notes
    note1 = v_dir / "Charlie.md"
    note1.write_text(
        "# Charlie\n\nCharlie's health records link to [[Pediatrician]] and [[Hospital#Emergency]].",
        encoding="utf-8",
    )

    note2 = v_dir / "Hospital.md"
    note2.write_text(
        "# Hospital\n\n## Emergency\nEmergency department is on 1st floor. See [[Charlie|Patient]].",
        encoding="utf-8",
    )

    vault = Vault.init(v_dir, name="test_vault")
    storage = VaultStorage(vault)
    storage.index()

    return vault, storage, v_dir


def test_lint_clean_vault_detects_dead_link(test_vault):
    vault, storage, v_dir = test_vault
    linter = VaultLinter(vault)

    # Note1 references [[Pediatrician]], which does not exist yet.
    # Note2 references [[Charlie|Patient]], which exists.
    # Note1 references [[Hospital#Emergency]], which exists.
    report = linter.lint()

    dead_link_issues = [i for i in report.issues if i.category == LintCategory.DEAD_LINK]
    assert len(dead_link_issues) == 1
    issue = dead_link_issues[0]
    assert issue.file_path == "Charlie.md"
    assert "[[Pediatrician]]" in issue.message
    assert issue.fixable is True
    assert issue.details["target"] == "Pediatrician"


def test_lint_sync_drift_unindexed_and_modified(test_vault):
    vault, storage, v_dir = test_vault
    linter = VaultLinter(vault)

    # 1. Add unindexed file
    unindexed = v_dir / "Insurance.md"
    unindexed.write_text("# Insurance\nPolicy 12345.", encoding="utf-8")

    # 2. Modify existing file
    note1 = v_dir / "Charlie.md"
    note1.write_text("# Charlie\nUpdated notes content.", encoding="utf-8")

    report = linter.lint()

    unindexed_issues = [
        i for i in report.issues if i.category == LintCategory.SYNC_DRIFT and i.details.get("type") == "unindexed"
    ]
    assert len(unindexed_issues) == 1
    assert unindexed_issues[0].file_path == "Insurance.md"

    modified_issues = [
        i for i in report.issues if i.category == LintCategory.SYNC_DRIFT and i.details.get("type") == "modified"
    ]
    assert len(modified_issues) == 1
    assert modified_issues[0].file_path == "Charlie.md"


def test_lint_document_hygiene_empty_and_frontmatter(test_vault):
    vault, storage, v_dir = test_vault
    linter = VaultLinter(vault)

    # Empty file
    empty_file = v_dir / "Empty.md"
    empty_file.write_text("", encoding="utf-8")

    # Headerless file
    headerless = v_dir / "Notes.md"
    headerless.write_text("Just some text without any heading.", encoding="utf-8")

    # Unclosed frontmatter
    malformed = v_dir / "Malformed.md"
    malformed.write_text("---\ntitle: Malformed\n\nNo closing delimiter here.", encoding="utf-8")

    report = linter.lint()

    empty_issues = [
        i for i in report.issues if i.category == LintCategory.DOCUMENT_HYGIENE and i.details.get("type") == "empty_file"
    ]
    assert len(empty_issues) == 1
    assert empty_issues[0].file_path == "Empty.md"

    missing_headings = [
        i for i in report.issues if i.category == LintCategory.DOCUMENT_HYGIENE and i.details.get("type") == "missing_heading"
    ]
    assert len(missing_headings) >= 1

    frontmatter_issues = [
        i for i in report.issues if i.category == LintCategory.DOCUMENT_HYGIENE and i.details.get("type") == "unclosed_frontmatter"
    ]
    assert len(frontmatter_issues) == 1
    assert frontmatter_issues[0].file_path == "Malformed.md"


def test_lint_storage_drift_orphan_chunks(test_vault):
    vault, storage, v_dir = test_vault
    # Inject an orphan record into LanceDB table
    storage.table.add(
        [
            {
                "id": "deleted_file.md#0",
                "file_path": "deleted_file.md",
                "document_title": "Deleted",
                "section_title": "Deleted",
                "breadcrumbs": "Deleted",
                "content": "Orphan content",
                "wikilinks": [],
                "vector": [0.0] * 768,
                "updated_at": "2026-10-14T00:00:00Z",
            }
        ]
    )

    linter = VaultLinter(vault)
    report = linter.lint()
    orphan_issues = [
        i for i in report.issues if i.category == LintCategory.STORAGE_DRIFT and i.details.get("type") == "orphan_chunks"
    ]
    assert len(orphan_issues) == 1
    assert orphan_issues[0].file_path == "deleted_file.md"
    assert orphan_issues[0].fixable is True


def test_remediation_dry_run(test_vault):
    vault, storage, v_dir = test_vault
    linter = VaultLinter(vault)

    # Create unindexed file
    new_doc = v_dir / "NewDoc.md"
    new_doc.write_text("# New Doc\nUnindexed.", encoding="utf-8")

    # Run dry run fix with create_stubs
    fix_result = linter.fix(create_stubs=True, dry_run=True)

    assert any("Pediatrician.md" in a for a in fix_result.repaired)
    assert any("NewDoc.md" in a for a in fix_result.repaired)

    # Verify that files were NOT created on disk during dry run
    assert not (v_dir / "Pediatrician.md").exists()
    assert "NewDoc.md" not in storage.load_manifest()["files"]


def test_remediation_apply_fixes_and_verify_clean(test_vault):
    vault, storage, v_dir = test_vault
    linter = VaultLinter(vault)

    # 1. Add unindexed file
    unindexed = v_dir / "Insurance.md"
    unindexed.write_text("# Insurance\nPolicy details.", encoding="utf-8")

    # 2. Add headerless file
    headerless = v_dir / "Todo.md"
    headerless.write_text("Call the clinic tomorrow.", encoding="utf-8")

    # 2b. Add unclosed frontmatter file
    malformed = v_dir / "Malformed.md"
    malformed.write_text("---\ntitle: Malformed\nauthor: Alice\n\nBody paragraph outside frontmatter.", encoding="utf-8")

    # 3. Inject orphan chunk into LanceDB
    storage.table.add(
        [
            {
                "id": "orphan.md#0",
                "file_path": "orphan.md",
                "document_title": "Orphan",
                "section_title": "Orphan",
                "breadcrumbs": "Orphan",
                "content": "Ghost content",
                "wikilinks": [],
                "vector": [0.0] * 768,
                "updated_at": "2026-10-14T00:00:00Z",
            }
        ]
    )

    linter = VaultLinter(vault)

    # Apply fixes including stub creation for [[Pediatrician]]
    fix_result = linter.fix(create_stubs=True, dry_run=False)
    assert len(fix_result.errors) == 0

    # Verify stub was created
    stub_path = v_dir / "Pediatrician.md"
    assert stub_path.is_file()
    assert "Stub created by Alcuin lint" in stub_path.read_text(encoding="utf-8")

    # Verify headerless file had title injected
    assert "# Todo" in (v_dir / "Todo.md").read_text(encoding="utf-8")

    # Verify unclosed frontmatter was closed immediately after metadata and before body text
    repaired_malformed = malformed.read_text(encoding="utf-8")
    assert repaired_malformed.startswith("---\ntitle: Malformed\nauthor: Alice\n---\n")
    assert repaired_malformed.index("\n---\n") < repaired_malformed.index("Body paragraph outside frontmatter.")

    # Verify post-fix audit is clean
    clean_report = linter.lint()
    assert clean_report.error_count == 0
    assert clean_report.warning_count == 0
    assert clean_report.is_clean is True


def test_cli_lint_commands(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "cli_vault"
    v_dir.mkdir()

    doc = v_dir / "Main.md"
    doc.write_text("# Main\nLinks to [[MissingLink]].", encoding="utf-8")

    runner = CliRunner(env={"COLUMNS": "160"})
    runner.invoke(cli, ["init", str(v_dir), "--name", "cli_vault"])
    runner.invoke(cli, ["index", "-v", "cli_vault"])

    # 1. Audit reports missing link
    res = runner.invoke(cli, ["lint", "-v", "cli_vault"])
    assert "MissingLink" in res.output
    assert "WARN" in res.output

    # 2. JSON output
    res_json = runner.invoke(cli, ["lint", "-v", "cli_vault", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["vault_name"] == "cli_vault"
    assert data["warning_count"] >= 1

    # 3. Dry run
    res_dry = runner.invoke(cli, ["lint", "-v", "cli_vault", "--fix", "--create-stubs", "--dry-run"])
    assert "Remediation Plan (Dry Run)" in res_dry.output
    assert "MissingLink.md" in res_dry.output

    # 4. Apply fix non-interactively
    res_fix = runner.invoke(cli, ["lint", "-v", "cli_vault", "--fix", "--create-stubs", "-y"])
    assert res_fix.exit_code == 0
    assert "Fixed" in res_fix.output or "Created stub" in res_fix.output

    # 5. Verify healthy vault
    res_clean = runner.invoke(cli, ["lint", "-v", "cli_vault"])
    assert res_clean.exit_code == 0
    assert "healthy" in res_clean.output
