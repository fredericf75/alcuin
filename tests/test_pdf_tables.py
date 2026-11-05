"""Unit tests for Structured PDF Table Extraction and Docling integration (SPEC-026)."""

import tempfile
from pathlib import Path
from click.testing import CliRunner
import pytest
import pymupdf as fitz

from alcuin.chunker.chunker import extract_pdf_content, chunk_document_text
from alcuin.cli.main import cli
from alcuin.storage.engine import VaultStorage
from alcuin.storage.ingest import InboxIngestionEngine
from alcuin.storage.vault import Vault


def create_sample_pdf_with_table(target_path: Path) -> Path:
    """Helper to generate a PDF containing prose text and a bordered table."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter size

    # Top prose text block
    page.insert_text((50, 50), "Quarterly Financial Overview", fontsize=14)
    page.insert_text((50, 75), "Below is the consolidated performance breakdown for Q3.", fontsize=10)

    # Table geometry: 2 columns, 2 rows (header + 1 data row)
    # Outer rect: (50, 100) to (450, 180)
    page.draw_rect(fitz.Rect(50, 100, 450, 180), color=(0, 0, 0), width=1)
    # Horizontal divider between header and row 1: y = 140
    page.draw_line(fitz.Point(50, 140), fitz.Point(450, 140), color=(0, 0, 0), width=1)
    # Vertical divider between col 1 and col 2: x = 250
    page.draw_line(fitz.Point(250, 100), fitz.Point(250, 180), color=(0, 0, 0), width=1)

    # Insert cell texts inside table rect
    page.insert_text((60, 125), "Department", fontsize=10)
    page.insert_text((260, 125), "Allocated Budget", fontsize=10)
    page.insert_text((60, 165), "Engineering", fontsize=10)
    page.insert_text((260, 165), "$1,250,000", fontsize=10)

    # Bottom prose text block
    page.insert_text((50, 220), "Notes: Figures are pending final external audit verification.", fontsize=10)

    doc.save(str(target_path))
    doc.close()
    return target_path


def test_native_pymupdf_table_extraction(tmp_path):
    pdf_path = tmp_path / "financial_report.pdf"
    create_sample_pdf_with_table(pdf_path)

    title, markdown = extract_pdf_content(pdf_path, engine="auto")

    assert "Financial Report" in title
    assert "Quarterly Financial Overview" in markdown
    assert "Below is the consolidated performance breakdown" in markdown
    assert "Notes: Figures are pending final external audit" in markdown

    # Verify table markdown syntax exists
    assert "| Department | Allocated Budget |" in markdown or "|Department|Allocated Budget|" in markdown
    assert "| Engineering | $1,250,000 |" in markdown or "|Engineering|$1,250,000|" in markdown

    # Verify cell deduplication: cells should not appear as redundant text outside the table
    # Specifically, count occurrences of "$1,250,000" in the markdown
    assert markdown.count("$1,250,000") == 1
    assert markdown.count("Engineering") == 1


def test_docling_engine_missing_raises_informative_error(tmp_path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy content")

    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf_content(pdf_path, engine="docling")

    err_msg = str(exc_info.value)
    assert "docling" in err_msg.lower()
    assert 'alcuin[docling]' in err_msg


def test_extracted_pdf_table_chunks_atomically_with_metadata(tmp_path):
    pdf_path = tmp_path / "budget_plan.pdf"
    create_sample_pdf_with_table(pdf_path)

    _, markdown = extract_pdf_content(pdf_path, engine="pymupdf")
    chunks = chunk_document_text("budget_plan.md", markdown)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) >= 1

    tbl = table_chunks[0]
    assert "TABLE_DATA:" in tbl.content
    assert "Engineering" in tbl.content
    assert "$1,250,000" in tbl.content


def test_inbox_ingest_with_engine_option(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "pdf_test_vault"
    vault = Vault.init(v_dir, name="pdf_vault")
    storage = VaultStorage(vault)
    storage.index()

    inbox_dir = v_dir / "inbox"
    pdf_file = inbox_dir / "audit_metrics.pdf"
    create_sample_pdf_with_table(pdf_file)

    runner = CliRunner()
    res = runner.invoke(cli, ["ingest", "-v", "pdf_vault", "--engine", "pymupdf"])
    assert res.exit_code == 0
    assert "audit_metrics.pdf" in res.output

    # Check generated vault note
    generated_note = v_dir / "audit_metrics.md"
    assert generated_note.exists()
    content = generated_note.read_text(encoding="utf-8")
    assert "| Department | Allocated Budget |" in content or "|Department|Allocated Budget|" in content
    assert "Engineering" in content
    assert "$1,250,000" in content
