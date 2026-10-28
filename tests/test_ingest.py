"""Unit tests for Multimodal Inbox Ingestion Pipeline (alcuin ingest)."""

import json
from pathlib import Path
from click.testing import CliRunner
import pytest

from alcuin.cli.main import cli
from alcuin.storage.engine import VaultStorage
from alcuin.storage.ingest import InboxIngestionEngine
from alcuin.storage.linter import LintCategory, VaultLinter
from alcuin.storage.vault import Vault


@pytest.fixture
def ingest_vault(tmp_path, monkeypatch):
    """Fixture providing a fresh vault with inbox and archive."""
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "test_intake_vault"
    vault = Vault.init(v_dir, name="intake_vault")
    storage = VaultStorage(vault)
    storage.index()
    return vault, storage, v_dir


def test_vault_init_scaffolds_inbox_and_archive(ingest_vault):
    vault, storage, v_dir = ingest_vault
    assert (v_dir / "inbox").is_dir()
    assert (v_dir / "archive").is_dir()
    assert "inbox/*" in vault.config.exclude_patterns
    assert "archive/*" in vault.config.exclude_patterns


def test_ingest_warns_when_folders_missing(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "bare_vault"
    v_dir.mkdir()

    # Create vault manually without inbox/archive
    alcuin_dir = v_dir / ".alcuin"
    alcuin_dir.mkdir()
    vault = Vault.init(v_dir, name="bare_vault")

    # Remove scaffolded inbox and archive to simulate legacy vault
    (v_dir / "inbox").rmdir()
    (v_dir / "archive").rmdir()

    runner = CliRunner()
    res = runner.invoke(cli, ["ingest", "-v", "bare_vault"])
    assert res.exit_code == 1
    assert "Warning:" in res.output
    assert "lint --fix" in res.output


def test_linter_detects_and_fixes_missing_intake_folders(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "legacy_vault"
    vault = Vault.init(v_dir, name="legacy_vault")

    # Remove intake folders
    (v_dir / "inbox").rmdir()
    (v_dir / "archive").rmdir()

    linter = VaultLinter(vault)
    report = linter.lint()

    missing_issues = [
        i for i in report.issues if i.details.get("type") == "missing_intake_folders"
    ]
    assert len(missing_issues) == 1
    assert missing_issues[0].fixable is True

    # Fix via linter
    fix_res = linter.fix(dry_run=False)
    assert len(fix_res.errors) == 0
    assert (v_dir / "inbox").is_dir()
    assert (v_dir / "archive").is_dir()

    # Re-audit is clean of folder warnings
    clean_report = linter.lint()
    assert not any(i.details.get("type") == "missing_intake_folders" for i in clean_report.issues)


def test_ingest_file_creates_markdown_and_archives(ingest_vault):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    # Place raw document in inbox
    sample = v_dir / "inbox" / "prescription_refill.txt"
    sample.write_text("Patient: Charlie\nMedication: Amoxicillin 250mg\nRefills: 2", encoding="utf-8")

    res = engine.ingest_inbox(dry_run=False)
    assert len(res.errors) == 0
    assert len(res.ingested_notes) == 1

    # Verify markdown note created in vault root
    note_path = v_dir / "prescription_refill.md"
    assert note_path.is_file()
    content = note_path.read_text(encoding="utf-8")
    assert "# Prescription Refill" in content
    assert "archive/prescription_refill.txt" in content
    assert "Amoxicillin 250mg" in content

    # Verify raw file moved to archive
    assert not sample.exists()
    assert (v_dir / "archive" / "prescription_refill.txt").is_file()

    # Verify history recorded
    history = engine.load_history()
    assert len(history) == 1
    rec = list(history.values())[0]
    assert rec.original_name == "prescription_refill.txt"
    assert rec.vault_note_path == "prescription_refill.md"


def test_archive_removal_does_not_affect_vault_note(ingest_vault):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    # Ingest document
    raw = v_dir / "inbox" / "invoice.txt"
    raw.write_text("Invoice #9876\nTotal: $150.00", encoding="utf-8")
    engine.ingest_inbox()

    note_path = v_dir / "invoice.md"
    archive_path = v_dir / "archive" / "invoice.txt"
    assert note_path.is_file()
    assert archive_path.is_file()

    # Delete raw file from archive
    archive_path.unlink()
    assert not archive_path.exists()

    # Vault note remains fully intact and indexed
    assert note_path.is_file()
    assert "Invoice #9876" in note_path.read_text(encoding="utf-8")

    # Indexing runs cleanly
    report = storage.index()
    assert "invoice.md" in report.unchanged_files or "invoice.md" in report.added_files


def test_ingest_deduplication(ingest_vault):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    # Ingest first time
    raw1 = v_dir / "inbox" / "receipt.txt"
    raw1.write_text("Home Depot total: $45.10", encoding="utf-8")
    res1 = engine.ingest_inbox()
    assert len(res1.ingested_notes) == 1

    # Drop exact same file again with same or different name
    raw2 = v_dir / "inbox" / "receipt_copy.txt"
    raw2.write_text("Home Depot total: $45.10", encoding="utf-8")
    res2 = engine.ingest_inbox()

    assert len(res2.ingested_notes) == 0
    assert len(res2.skipped_duplicates) == 1
    assert "Skipped: Already ingested" in res2.skipped_duplicates[0]

    # File remains in inbox so user can decide what to do
    assert raw2.is_file()


def test_ingest_dry_run(ingest_vault):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    raw = v_dir / "inbox" / "whiteboard.txt"
    raw.write_text("Architecture notes: A -> B", encoding="utf-8")

    res = engine.ingest_inbox(dry_run=True)
    assert len(res.ingested_notes) == 1
    assert "[Dry Run]" in res.ingested_notes[0]

    # No files modified on disk
    assert raw.is_file()
    assert not (v_dir / "whiteboard.md").exists()
    assert not (v_dir / "archive" / "whiteboard.txt").exists()


def test_cli_ingest(ingest_vault):
    vault, storage, v_dir = ingest_vault

    raw = v_dir / "inbox" / "lab_results.txt"
    raw.write_text("Blood test: Normal", encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(cli, ["ingest", "-v", "intake_vault"])
    assert res.exit_code == 0
    assert "Ingested 'lab_results.txt'" in res.output

    assert (v_dir / "lab_results.md").is_file()
    assert (v_dir / "archive" / "lab_results.txt").is_file()


def test_ingest_quarantines_rejected_files_and_writes_reason(ingest_vault):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    # Place rejected formats in inbox
    sheet = v_dir / "inbox" / "finances.xlsx"
    sheet.write_bytes(b"PK fake excel binary content")
    csv_file = v_dir / "inbox" / "data.csv"
    csv_file.write_text("a,b,c\n1,2,3", encoding="utf-8")
    archive_zip = v_dir / "inbox" / "bundle.zip"
    archive_zip.write_bytes(b"PK fake zip content")

    res = engine.ingest_inbox(dry_run=False)

    assert len(res.rejected_files) == 3
    assert len(res.ingested_notes) == 0

    # Verify original files removed from inbox root
    assert not sheet.exists()
    assert not csv_file.exists()
    assert not archive_zip.exists()

    # Verify quarantined to _rejected/ with companion .reason.txt
    rejected_dir = v_dir / "inbox" / "_rejected"
    assert (rejected_dir / "finances.xlsx").is_file()
    assert (rejected_dir / "finances.xlsx.reason.txt").is_file()
    assert "Spreadsheet" in (rejected_dir / "finances.xlsx.reason.txt").read_text(encoding="utf-8")

    assert (rejected_dir / "data.csv").is_file()
    assert (rejected_dir / "data.csv.reason.txt").is_file()
    assert "Spreadsheet" in (rejected_dir / "data.csv.reason.txt").read_text(encoding="utf-8")

    assert (rejected_dir / "bundle.zip").is_file()
    assert (rejected_dir / "bundle.zip.reason.txt").is_file()
    assert "Unsupported format (.zip)" in (rejected_dir / "bundle.zip.reason.txt").read_text(encoding="utf-8")

    # Vault root has no unwanted notes
    assert not (v_dir / "finances.md").exists()
    assert not (v_dir / "data.md").exists()
    assert not (v_dir / "bundle.md").exists()


def test_ingest_quarantine_dry_run(ingest_vault):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    ppt = v_dir / "inbox" / "presentation.pptx"
    ppt.write_bytes(b"PK fake pptx content")

    res = engine.ingest_inbox(dry_run=True)
    assert len(res.rejected_files) == 1
    assert "[Dry Run] Would reject 'presentation.pptx'" in res.rejected_files[0]

    # File remains untouched in inbox
    assert ppt.is_file()
    assert not (v_dir / "inbox" / "_rejected").exists()


def test_image_gating_sparse_text_creates_clean_asset_note(ingest_vault, monkeypatch):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    # Mock Tesseract to return sparse noise (< 20 alnum chars)
    monkeypatch.setattr(engine, "extract_image_with_tesseract", lambda p: "... 123 !")
    monkeypatch.setattr(engine, "find_local_vision_model", lambda: None)

    img = v_dir / "inbox" / "icon.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")

    res = engine.ingest_inbox(dry_run=False)
    assert len(res.ingested_notes) == 1

    note_path = v_dir / "icon.md"
    assert note_path.is_file()
    content = note_path.read_text(encoding="utf-8")
    assert "Visual asset ingested: `icon.png`" in content
    assert "No legible document text detected" in content
    assert "## Document Text (OCR" not in content


def test_image_gating_sufficient_text_creates_ocr_transcription(ingest_vault, monkeypatch):
    vault, storage, v_dir = ingest_vault
    engine = InboxIngestionEngine(vault)

    # Mock Tesseract returning substantive document text (>= 20 alnum chars)
    ocr_sample = "INVOICE #4092 Date: 2026-10-25 Total Amount Due: $1,420.00"
    monkeypatch.setattr(engine, "extract_image_with_tesseract", lambda p: ocr_sample)

    img = v_dir / "inbox" / "bill.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")

    res = engine.ingest_inbox(dry_run=False)
    assert len(res.ingested_notes) == 1

    note_path = v_dir / "bill.md"
    assert note_path.is_file()
    content = note_path.read_text(encoding="utf-8")
    assert "## Transcribed Text (OCR)" in content
    assert ocr_sample in content


def test_periodic_indexer_inbox_stabilization(ingest_vault):
    import asyncio
    from alcuin.api.daemon import PeriodicIndexer

    vault, storage, v_dir = ingest_vault
    indexer = PeriodicIndexer(check_interval=1.0)

    # Place file in inbox
    sample = v_dir / "inbox" / "streamed_note.txt"
    sample.write_text("Work in progress content", encoding="utf-8")

    # Poll cycle 1: Observes file and records stat, should NOT ingest yet
    asyncio.run(indexer._process_vault_inbox("intake_vault", vault))
    assert sample.exists()
    assert not (v_dir / "streamed_note.md").exists()
    assert str(sample) in indexer._file_stats

    # Poll cycle 2: Stat is unchanged -> stable -> ingests!
    asyncio.run(indexer._process_vault_inbox("intake_vault", vault))
    assert not sample.exists()
    assert (v_dir / "streamed_note.md").is_file()
    assert str(sample) not in indexer._file_stats

