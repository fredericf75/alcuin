"""Unit tests for SPEC-042 Scanned PDF per-page OCR fallback."""

from __future__ import annotations

import pymupdf

from alcuin.chunker.chunker import extract_pdf_content
from alcuin.storage.engine import chunk_file_in_memory


def _make_scanned_pdf(tmp_path, name: str = "scanned_marriage_license.pdf"):
    src_doc = pymupdf.open()
    p = src_doc.new_page(width=612, height=300)
    p.insert_text((50, 100), "MARRIAGE LICENSE CERTIFICATE", fontsize=22)
    p.insert_text((50, 150), "Date of Marriage: June 14, 2018", fontsize=18)
    pix = p.get_pixmap(dpi=200)
    src_doc.close()

    scan_pdf = tmp_path / name
    img_doc = pymupdf.open()
    img_page = img_doc.new_page(width=612, height=300)
    img_page.insert_image(img_page.rect, stream=pix.tobytes("png"))
    img_doc.save(str(scan_pdf))
    img_doc.close()
    return scan_pdf


def test_scanned_image_only_pdf_recovers_text_via_primary_ocr(tmp_path):
    scan_pdf = _make_scanned_pdf(tmp_path)

    check_doc = pymupdf.open(str(scan_pdf))
    assert check_doc[0].get_text("text").strip() == ""
    check_doc.close()

    title, raw_text = extract_pdf_content(scan_pdf)
    assert "MARRIAGE" in raw_text.upper()
    assert "2018" in raw_text

    _, _, chunks = chunk_file_in_memory(scan_pdf)
    assert len(chunks) >= 1
    assert any("2018" in c.content for c in chunks)


def test_scanned_image_only_pdf_falls_back_to_rapidocr_when_no_tesseract(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("rapidocr_onnxruntime")
    import alcuin.chunker.chunker as chunker_mod

    monkeypatch.setattr(chunker_mod, "find_tesseract_binary", lambda: None)
    scan_pdf = _make_scanned_pdf(tmp_path, "scanned_fallback.pdf")

    _, raw_text = extract_pdf_content(scan_pdf)
    assert "MARRIAGE" in raw_text.upper()
    assert "2018" in raw_text
