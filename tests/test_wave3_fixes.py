"""Regression tests for Wave 3 code review fixes (chunker, ingest, daemon, CLI)."""

from pathlib import Path
from unittest.mock import MagicMock

import click

from alcuin.chunker.chunker import (
    chunk_document_text,
    extract_docx_content,
    serialize_tabular_data_with_metadata,
    slice_massive_table,
)
from alcuin.storage.ingest import InboxIngestionEngine
from alcuin.storage.vault import Vault


def test_chunker_hierarchy_stack_preserves_l2_under_l3_and_sibling_l2s():
    text = (
        "# Chapter 1\n\nIntro.\n\n"
        "## Section 1.1\n\nBody 1.1.\n\n"
        "### Subsection 1.1.1\n\nBody 1.1.1.\n"
    )
    chunks = chunk_document_text("Doc", text)
    sub = [c for c in chunks if c.section_title == "Subsection 1.1.1"][0]
    assert sub.hierarchy == ["Doc", "Chapter 1", "Section 1.1", "Subsection 1.1.1"]

    text_no_l1 = "## First\n\nA.\n\n## Second\n\nB.\n"
    chunks2 = chunk_document_text("Doc", text_no_l1)
    second = [c for c in chunks2 if c.section_title == "Second"][0]
    assert second.hierarchy == ["Doc", "Second"]


def test_chunker_oversized_paragraph_preserves_order():
    text = "# Section\n\nFIRST_SHORT_PARA\n\n" + ("B" * 400) + "\n\nTHIRD_SHORT_PARA\n"
    chunks = chunk_document_text("Doc", text, target_size=150, max_size=200)
    assert len(chunks) >= 4
    assert "FIRST_SHORT_PARA" in chunks[0].content
    assert "BBBB" in chunks[1].content
    assert "THIRD_SHORT_PARA" in chunks[-1].content


def test_chunker_consecutive_non_markdown_headings_not_suppressed():
    text = (
        "ARTICLE I: DEFINITIONS\n"
        "Section 1.1 General Terms\n"
        "(a) First Item body text.\n"
    )
    chunks = chunk_document_text("Doc", text)
    titles = [c.section_title for c in chunks]
    assert any("General Terms" in t or "1.1" in t for t in titles)


def test_chunker_escaped_pipes_and_sliced_table_badge():
    raw_tbl = (
        "| Key | Value |\n"
        "| --- | --- |\n"
        "| A \\| B | 100 \\| 200 |\n"
    )
    serialized, headers, n_rows, n_cols, ctype = serialize_tabular_data_with_metadata(raw_tbl)
    assert headers == ["Key", "Value"]
    assert n_cols == 2
    assert n_rows == 1
    assert ctype == "table"

    # Sliced massive table must not emit duplicate badges or overwrite total rows
    big_lines = ["| Col1 | Col2 | Col3 |", "| --- | --- | --- |"] + [
        f"| Row{i} | Val{i} | Extra{i} |" for i in range(1, 26)
    ]
    slices = slice_massive_table(big_lines, max_chars=400)
    assert len(slices) >= 2
    ser, _, total_rows, total_cols, _ = serialize_tabular_data_with_metadata(slices[0])
    assert total_rows == 25
    assert total_cols == 3
    assert ser.count("> **Table") == 1


def test_chunker_prose_with_showing_rows_still_splits():
    prose = "# Notes\n\nShowing rows in the UI is helpful.\n\n" + ("word " * 200) + "\n\n" + ("more " * 200)
    chunks = chunk_document_text("Doc", prose, target_size=300, max_size=500)
    assert len(chunks) >= 2


def test_extract_docx_preserves_interleaved_table_order(tmp_path):
    import pytest

    docx = pytest.importorskip("docx")

    doc = docx.Document()
    doc.add_heading("Section Alpha", level=1)
    tbl = doc.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "H1"
    tbl.cell(0, 1).text = "H2"
    tbl.cell(1, 0).text = "V1"
    tbl.cell(1, 1).text = "V2"
    doc.add_heading("Section Beta", level=1)
    doc.add_paragraph("Trailing text.")

    path = tmp_path / "ordered.docx"
    doc.save(str(path))
    _, md = extract_docx_content(path)
    assert md.index("# Section Alpha") < md.index("| H1 | H2 |") < md.index("# Section Beta")


def test_ingest_failed_pdf_not_archived_and_only_files_respected(tmp_path, monkeypatch):
    vault = Vault.init(tmp_path / "v_ingest")
    engine = InboxIngestionEngine(vault)
    engine.inbox_dir.mkdir(exist_ok=True)
    engine.archive_dir.mkdir(exist_ok=True)

    bad_pdf = engine.inbox_dir / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a real pdf")

    note_path, status = engine.ingest_file(bad_pdf)
    assert note_path is None
    assert "Extraction failed" in status
    assert bad_pdf.exists()
    assert not list(engine.archive_dir.glob("*.pdf"))
    assert engine.load_history() == {}

    # only_files should ignore other files in inbox
    stable_md = engine.inbox_dir / "stable.md"
    unstable_md = engine.inbox_dir / "unstable.md"
    stable_md.write_text("# Stable\n\nDone.", encoding="utf-8")
    unstable_md.write_text("# Unstable\n\nWriting...", encoding="utf-8")

    # Avoid triggering real embedding index in unit test
    monkeypatch.setattr("alcuin.storage.ingest.VaultStorage", MagicMock())
    res = engine.ingest_inbox(only_files=[stable_md])
    assert len(res.ingested_notes) == 1
    assert not stable_md.exists()
    assert unstable_md.exists()


def test_assisted_review_abort_returns_modified_for_reindex(tmp_path, monkeypatch):
    from alcuin.cli.main import _run_assisted_review

    vault = Vault.init(tmp_path / "v_rev")
    curated = vault.root_path / "_curated" / "people"
    curated.mkdir(parents=True)
    (curated / "Jane_Doe.md").write_text(
        "# Jane Doe — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Date of Birth**: 1990-01-01 *(Source: `a.pdf`)*\n\n"
        "## Conflicting / Alternate Mentions\n"
        "- **Date of Birth**: `1990-01-02` *(Source: `b.pdf`)* — differs from canonical `1990-01-01`\n",
        encoding="utf-8",
    )
    (curated / "John_Doe.md").write_text(
        "# John Doe — Canonical Profile\n\n"
        "## Canonical Facts\n"
        "- **Address**: 100 Old Rd *(Source: `a.pdf`)*\n\n"
        "## Conflicting / Alternate Mentions\n"
        "- **Address**: `200 New Rd` *(Source: `b.pdf`)* — differs from canonical `100 Old Rd`\n",
        encoding="utf-8",
    )

    # First prompt picks section 1, second picks option 1, third raises click.Abort (Ctrl+C)
    answers = iter(["1", "1"])

    def fake_prompt(*args, **kwargs):
        try:
            return next(answers)
        except StopIteration:
            raise click.Abort()

    monkeypatch.setattr(click, "prompt", fake_prompt)
    modified = _run_assisted_review(vault)
    assert modified is True
