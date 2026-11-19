#!/usr/bin/env python3
"""
chunk_document.py - Clause-Aware Legal & Tech-Spec Document Chunker

Extracts, structures, and chunks contracts, technical specifications, and RFCs
from PDF, Word (.docx), Markdown (.md), Plain Text (.txt), Web URLs, and Clipboard.

Features:
  - Complete legal clauses & tech-spec hierarchy (Articles, Sections, REQ tags)
  - Tables (Markdown / HTML) as atomic unbreakable blocks
  - Code blocks and definitions
  - Hierarchical breadcrumbs baked into every chunk for high-accuracy RAG vector search
  - Scanned PDF Detection & Auto-OCR via Tesseract & PyMuPDF
  - Batch Directory Ingestion with combined Master Dataset generation
  - Clipboard Ingestion (`-c`) for quick URL or text chunking

Outputs:
  - Enriched Markdown (_enriched.md): Master file ready for AnythingLLM drag-and-drop
  - Split Sections (_sections/): Folder of individual section files for targeted inspection / `ask -f`
  - JSONL (_chunks.jsonl): Machine-readable streaming format for vector DBs and Python pipelines
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
# Document Structure Data Models
# =====================================================================

@dataclass
class DocumentChunk:
    chunk_id: int
    document_title: str
    section_title: str
    hierarchy: List[str]
    breadcrumbs: str
    content: str
    char_count: int
    token_estimate: int
    chunk_type: str = "clause"  # clause, table, schedule, code, preamble
    table_headers: List[str] = field(default_factory=list)
    table_rows: int = 0
    table_cols: int = 0


# =====================================================================
# OCR & Tesseract Detection
# =====================================================================

def find_tesseract_binary() -> Optional[str]:
    """Find installed tesseract executable on Windows or Linux."""
    tess = shutil.which("tesseract")
    if tess:
        return tess

    standard_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path("/usr/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
    ]
    for p in standard_paths:
        if p.is_file():
            return str(p)
    return None


# =====================================================================
# Clipboard Utility
# =====================================================================

def get_clipboard_text() -> str:
    """Read text from Linux, macOS, or Windows clipboard."""
    if sys.platform.startswith("linux"):
        if shutil.which("wl-paste"):
            try:
                res = subprocess.run(["wl-paste", "-n"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0 and res.stdout:
                    return res.stdout.strip()
            except Exception:
                pass
        for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]:
            if shutil.which(cmd[0]):
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                    if res.returncode == 0 and res.stdout:
                        return res.stdout.strip()
                except Exception:
                    pass

    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        text = r.clipboard_get()
        r.destroy()
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            if user32.OpenClipboard(None):
                try:
                    handle = user32.GetClipboardData(13)  # CF_UNICODETEXT
                    if handle:
                        ptr = kernel32.GlobalLock(handle)
                        if ptr:
                            try:
                                return ctypes.c_wchar_p(ptr).value or ""
                            finally:
                                kernel32.GlobalUnlock(handle)
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass

        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=0x08000000,
            )
            return res.stdout.strip()
        except Exception:
            pass

    return ""


# =====================================================================
# Extraction Handlers: Local Files, Web URLs, and PDFs with OCR
# =====================================================================

def fetch_url_content(url: str) -> Tuple[str, str]:
    """Fetch content from a web URL, strip clutter, and return (title, markdown_text)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        content_type = resp.headers.get("content-type", "").lower()
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        raw_html = resp.read().decode(charset, errors="replace")

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 is required for URL extraction. Run: pip install beautifulsoup4")

    soup = BeautifulSoup(raw_html, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if not title:
        parsed_url = urllib.parse.urlparse(url)
        path_name = Path(parsed_url.path).name or parsed_url.netloc
        title = path_name.replace("-", " ").replace("_", " ").title()

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form", "iframe"]):
        tag.decompose()

    content_root = soup.find("article") or soup.find("main") or soup.find(role="main") or soup.body or soup
    markdown_lines = []

    def process_node(node):
        if not node:
            return
        if isinstance(node, str):
            text = node.strip()
            if text:
                markdown_lines.append(text)
            return

        name = node.name.lower() if node.name else ""

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            heading_text = node.get_text(separator=" ", strip=True)
            if heading_text:
                markdown_lines.append(f"\n{'#' * level} {heading_text}\n")
        elif name == "p":
            para_text = node.get_text(separator=" ", strip=True)
            if para_text:
                markdown_lines.append(f"\n{para_text}\n")
        elif name in ("ul", "ol"):
            for li in node.find_all("li", recursive=False):
                li_text = li.get_text(separator=" ", strip=True)
                if li_text:
                    markdown_lines.append(f"- {li_text}")
            markdown_lines.append("")
        elif name == "table":
            rows = node.find_all("tr")
            if rows:
                markdown_lines.append("")
                header_processed = False
                for r in rows:
                    ths = r.find_all("th")
                    tds = r.find_all("td")
                    cells = [c.get_text(separator=" ", strip=True).replace("|", "\\|") for c in (ths or tds)]
                    if cells:
                        markdown_lines.append("| " + " | ".join(cells) + " |")
                        if ths and not header_processed:
                            markdown_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                            header_processed = True
                markdown_lines.append("")
        elif name in ("pre", "code"):
            code_text = node.get_text()
            if "\n" in code_text:
                markdown_lines.append(f"\n```\n{code_text.strip()}\n```\n")
            else:
                markdown_lines.append(f"`{code_text.strip()}`")
        elif name == "hr":
            markdown_lines.append("\n---\n")
        elif name in ("div", "section"):
            for child in node.children:
                process_node(child)
        else:
            for child in node.children:
                process_node(child)

    process_node(content_root)
    full_markdown = "\n".join(markdown_lines)
    full_markdown = re.sub(r"\n{3,}", "\n\n", full_markdown)
    return title, full_markdown


_RAPIDOCR_ENGINE = None


def _get_rapidocr_engine():
    """Lazily initialize a singleton RapidOCR engine."""
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


_TESSERACT_LANGS_CACHE: dict[str, str] = {}


def _get_tesseract_langs(tess_exe: str) -> str:
    """Return installed Tesseract language pack string (preferring eng+fra when available)."""
    if tess_exe in _TESSERACT_LANGS_CACHE:
        return _TESSERACT_LANGS_CACHE[tess_exe]
    import subprocess

    lang_arg = "eng"
    try:
        res = subprocess.run(
            [tess_exe, "--list-langs"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        available = {line.strip() for line in (res.stdout or "").splitlines() if line.strip()}
        chosen = [lang for lang in ("eng", "fra") if lang in available]
        if chosen:
            lang_arg = "+".join(chosen)
    except Exception:
        pass
    _TESSERACT_LANGS_CACHE[tess_exe] = lang_arg
    return lang_arg


def _ocr_page_with_tesseract(page, tess_exe: str) -> str:
    """Render a PyMuPDF page at 200 DPI and extract text via Tesseract CLI (no pytesseract required)."""
    import subprocess
    import tempfile

    langs = _get_tesseract_langs(tess_exe)
    png_bytes = page.get_pixmap(dpi=200).tobytes("png")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name
    try:
        res = subprocess.run(
            [tess_exe, tmp_path, "stdout", "-l", langs, "--psm", "3"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if res.returncode != 0 and res.stderr:
            raise RuntimeError(res.stderr.strip())
        return (res.stdout or "").strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _ocr_page_with_rapidocr(page) -> str:
    """Render a PyMuPDF page at 200 DPI and extract text lines via RapidOCR."""
    engine = _get_rapidocr_engine()
    png_bytes = page.get_pixmap(dpi=200).tobytes("png")
    result, _ = engine(png_bytes)
    if not result:
        return ""
    lines = [str(item[1]).strip() for item in result if len(item) >= 2 and str(item[1]).strip()]
    return "\n".join(lines).strip()


def extract_pdf_content(pdf_path: Path, engine: str = "auto") -> Tuple[str, str]:
    """Extract structured text and tables from PDF, with auto-OCR for scanned pages (SPEC-026)."""
    if engine.lower() == "docling":
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()
            title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
            return title, markdown_text
        except ImportError:
            raise RuntimeError(
                "PDF extraction engine 'docling' requested but not installed.\n"
                "To enable vision-based document layout parsing, install the docling extra:\n"
                "  pip install \"alcuin[docling]\""
            )

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            raise RuntimeError("pymupdf is required for PDF extraction. Run: pip install pymupdf")

    tess_exe = find_tesseract_binary()

    doc = fitz.open(str(pdf_path))
    doc_title = doc.metadata.get("title") if doc.metadata else ""
    if not doc_title or len(doc_title.strip()) < 3:
        doc_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    pages_text = []
    scanned_pages = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text").strip()

        # Detect scanned page (less than 50 chars of selectable text with images/drawings)
        if len(raw_text) < 50 and (len(page.get_images()) > 0 or len(page.get_drawings()) > 0):
            scanned_pages.append(page_idx + 1)
            ocr_text = ""
            if tess_exe:
                try:
                    ocr_text = _ocr_page_with_tesseract(page, tess_exe)
                except Exception as e:
                    print(f"  \033[91m[OCR Error]\033[0m Page {page_idx + 1}: {e}")
            if not ocr_text:
                try:
                    ocr_text = _ocr_page_with_rapidocr(page)
                except Exception as e:
                    print(f"  \033[91m[RapidOCR Error]\033[0m Page {page_idx + 1}: {e}")
            if ocr_text:
                print(f"  \033[92m[OCR]\033[0m Page {page_idx + 1}: Scanned image recognized ({len(ocr_text):,} characters)")
                pages_text.append(ocr_text)
            else:
                print(f"  \033[90m[OCR]\033[0m Page {page_idx + 1}: No text detected via OCR")
            continue

        # Native coordinate-aware layout & table extraction (handles borderless tables + 2-column prose)
        assembled_page = _extract_pdf_page_layout(page)
        if assembled_page.strip():
            pages_text.append(assembled_page.strip())
        elif raw_text:
            pages_text.append(raw_text)

    doc.close()
    return doc_title, "\n\n".join(pages_text)


def _cluster_x_columns(
    intervals: List[Tuple[float, float]],
    row_intervals: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Tuple[float, float]]:
    """Cluster segment x-intervals into columns without merging distinct columns."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[1] - x[0], x[0]))
    cols: List[List[float]] = []
    for x0, x1 in sorted_iv:
        overlapping = []
        for ci, (cx0, cx1) in enumerate(cols):
            ov = min(x1, cx1) - max(x0, cx0)
            if ov > 1.0:
                overlapping.append((ov, ci))
        if len(overlapping) == 1:
            ci = overlapping[0][1]
            cols[ci][0] = min(cols[ci][0], x0)
            cols[ci][1] = max(cols[ci][1], x1)
        elif len(overlapping) == 0:
            cols.append([x0, x1])
    cols.sort(key=lambda c: c[0])

    # Merge adjacent clusters that never co-occur on the same row and are within 20pt
    # (e.g. indented row sub-labels like "Total net sales" vs "iPhone").
    if row_intervals and len(cols) >= 2:
        merged = True
        while merged and len(cols) >= 2:
            merged = False
            for ci in range(len(cols) - 1):
                c0, c1 = cols[ci], cols[ci + 1]
                if c1[0] - c0[1] > 20.0:
                    continue
                co_occur = False
                for r_ivs in row_intervals:
                    # Only count as co-occurring if two DISTINCT segments on the same row map to c0 and c1
                    left_segs = [idx_s for idx_s, (rx0, rx1) in enumerate(r_ivs) if min(rx1, c0[1]) - max(rx0, c0[0]) > 0]
                    right_segs = [idx_s for idx_s, (rx0, rx1) in enumerate(r_ivs) if min(rx1, c1[1]) - max(rx0, c1[0]) > 0]
                    if left_segs and right_segs and set(left_segs) != set(right_segs):
                        co_occur = True
                        break
                if not co_occur:
                    cols[ci] = [min(c0[0], c1[0]), max(c0[1], c1[1])]
                    cols.pop(ci + 1)
                    merged = True
                    break

    return [(c[0], c[1]) for c in cols]


def _extract_pdf_segments(
    words: List[Tuple],
    y_tol: float = 3.0,
    gap_tol: float = 9.0,
) -> List[Tuple[float, float, List[Tuple[float, float, str]]]]:
    """Group PyMuPDF word tuples into y-lines and split each line into (x0, x1, text) segments."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: (w[1], w[0]))
    y_lines: List[List[Tuple]] = []
    for w in sorted_w:
        y_mid = (w[1] + w[3]) / 2.0
        if y_lines:
            prev_mid = sum((pw[1] + pw[3]) / 2.0 for pw in y_lines[-1]) / len(y_lines[-1])
            if abs(y_mid - prev_mid) <= y_tol:
                y_lines[-1].append(w)
                continue
        y_lines.append([w])

    result = []
    for line_w in y_lines:
        line_w.sort(key=lambda w: w[0])
        segs: List[Tuple[float, float, str]] = []
        curr_words = [line_w[0][4]]
        sx0, sx1 = line_w[0][0], line_w[0][2]
        for w in line_w[1:]:
            if w[0] - sx1 > gap_tol:
                segs.append((sx0, sx1, " ".join(curr_words)))
                curr_words = [w[4]]
                sx0, sx1 = w[0], w[2]
            else:
                curr_words.append(w[4])
                sx1 = max(sx1, w[2])
        segs.append((sx0, sx1, " ".join(curr_words)))

        # Split trailing currency symbol attached to a label (e.g. "sales $" -> "sales", "$")
        split_segs: List[Tuple[float, float, str]] = []
        for x0, x1, txt in segs:
            if len(txt) > 2 and txt.endswith((" $", " €", " £")):
                sym = txt[-1]
                lbl = txt[:-2].strip()
                split_segs.append((x0, max(x0 + 1.0, x1 - 8.0), lbl))
                split_segs.append((max(x0, x1 - 6.0), x1, sym))
            else:
                split_segs.append((x0, x1, txt))

        # Merge standalone currency prefix with next segment, and '%' suffix with previous segment
        merged: List[Tuple[float, float, str]] = []
        idx = 0
        while idx < len(split_segs):
            x0, x1, txt = split_segs[idx]
            if txt in ("$", "€", "£") and idx + 1 < len(split_segs):
                nx0, nx1, ntxt = split_segs[idx + 1]
                merged.append((nx0, nx1, f"{txt}{ntxt}" if ntxt[0].isdigit() else f"{txt} {ntxt}"))
                idx += 2
                continue
            if txt == "%" and merged:
                px0, px1, ptxt = merged[-1]
                merged[-1] = (px0, x1, f"{ptxt}%")
                idx += 1
                continue
            merged.append((x0, x1, txt))
            idx += 1

        y0_min = min(w[1] for w in line_w)
        y1_max = max(w[3] for w in line_w)
        result.append((y0_min, y1_max, merged))
    return result


def _segments_to_markdown_table(
    band_lines: List[Tuple[float, float, List[Tuple[float, float, str]]]],
    pre_header_line: Optional[Tuple[float, float, List[Tuple[float, float, str]]]] = None,
) -> Tuple[Optional[str], bool]:
    """Reconstruct a Markdown pipe table from coordinate-segmented lines. Returns (md_table, consumed_pre_hdr)."""
    multi_rows = [ln for ln in band_lines if len(ln[2]) >= 2]
    if len(multi_rows) < 2:
        return None, False

    all_ivs = [(s[0], s[1]) for ln in multi_rows for s in ln[2]]
    row_ivs = [[(s[0], s[1]) for s in ln[2]] for ln in multi_rows]
    cols = _cluster_x_columns(all_ivs, row_intervals=row_ivs)
    if len(cols) < 2:
        return None, False

    def _assign_col(x0: float, x1: float) -> int:
        best_c = 0
        best_score = -1e9
        mid = (x0 + x1) / 2.0
        for ci, (cx0, cx1) in enumerate(cols):
            ov = min(x1, cx1) - max(x0, cx0)
            score = ov if ov > 0 else -abs(mid - (cx0 + cx1) / 2.0)
            if score > best_score:
                best_score = score
                best_c = ci
        return best_c

    grid_rows: List[List[str]] = []
    pending_col0: List[str] = []

    for _y0, _y1, segs in band_lines:
        if len(segs) == 1:
            sx0, sx1, stxt = segs[0]
            ci = _assign_col(sx0, sx1)
            if ci == 0:
                if stxt.startswith("(") and stxt.endswith(")") and len(stxt) <= 5 and grid_rows:
                    grid_rows[-1][0] = (grid_rows[-1][0] + " " + stxt).strip()
                else:
                    pending_col0.append(stxt)
                continue

        row = [""] * len(cols)
        for sx0, sx1, stxt in segs:
            ci = _assign_col(sx0, sx1)
            row[ci] = (row[ci] + " " + stxt).strip() if row[ci] else stxt
        if pending_col0:
            prefix = " ".join(pending_col0)
            row[0] = (prefix + " " + row[0]).strip() if row[0] else prefix
            pending_col0 = []
        grid_rows.append(row)

    if pending_col0 and grid_rows:
        grid_rows[-1][0] = (grid_rows[-1][0] + " " + " ".join(pending_col0)).strip()

    if len(grid_rows) < 2:
        return None, False

    non_empty_cols = [ci for ci in range(len(cols)) if any(r[ci].strip() for r in grid_rows)]
    if len(non_empty_cols) < 2:
        return None, False
    cols = [cols[ci] for ci in non_empty_cols]
    grid_rows = [[r[ci] for ci in non_empty_cols] for r in grid_rows]

    consumed_pre_hdr = False
    if pre_header_line and len(pre_header_line[2]) == 1:
        px0, px1, ptxt = pre_header_line[2][0]
        if len(ptxt) <= 30 and not ptxt.endswith((":", ".", ",")):
            pci = _assign_col(px0, px1)
            if pci > 0:
                hdr = [""] * len(cols)
                hdr[0] = "Category / Item"
                hdr[pci] = ptxt
                grid_rows.insert(0, hdr)
                consumed_pre_hdr = True

    if len(grid_rows) >= 3 and not grid_rows[0][-1].strip():
        hdr_end = 0
        while hdr_end + 1 < len(grid_rows) and not grid_rows[hdr_end][-1].strip():
            hdr_end += 1
        if 0 < hdr_end < len(grid_rows) - 1 and grid_rows[hdr_end][-1].strip():
            merged_hdr = [""] * len(cols)
            for hr in grid_rows[: hdr_end + 1]:
                for ci, val in enumerate(hr):
                    if val.strip():
                        merged_hdr[ci] = (merged_hdr[ci] + " " + val.strip()).strip()
            grid_rows = [merged_hdr] + grid_rows[hdr_end + 1 :]

    header = grid_rows[0]
    if not header[0].strip():
        header[0] = "Category / Item"
    md_lines = [
        "| " + " | ".join(c.replace("|", "\\|") for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in grid_rows[1:]:
        md_lines.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
    return "\n".join(md_lines), consumed_pre_hdr


def _extract_pdf_blocks_from_lines(
    lines: List[Tuple[float, float, List[Tuple[float, float, str]]]],
    page_width: float,
) -> List[str]:
    out_blocks: List[str] = []
    i = 0
    while i < len(lines):
        if len(lines[i][2]) >= 2:
            band = [lines[i]]
            j = i + 1
            while j < len(lines):
                y_gap = lines[j][0] - lines[j - 1][1]
                if y_gap > 55.0:
                    break
                if len(lines[j][2]) >= 2:
                    if y_gap > 18.0:
                        # Only bridge a wide vertical gap (18..55pt, e.g. tall padded cells)
                        # if the next row's segments align with existing band columns.
                        band_ivs = [(s[0], s[1]) for ln in band if len(ln[2]) >= 2 for s in ln[2]]
                        aligned = sum(
                            1 for sx0, sx1, _ in lines[j][2]
                            if any(min(sx1, bx1) - max(sx0, bx0) > 0 for bx0, bx1 in band_ivs)
                        )
                        if aligned < 2:
                            break
                    band.append(lines[j])
                    j += 1
                else:
                    if y_gap > 18.0:
                        break
                    sx0, sx1, stxt = lines[j][2][0]
                    if len(stxt) <= 45 and (sx1 - sx0) < page_width * 0.45:
                        found_next = False
                        for k in range(j + 1, min(len(lines), j + 5)):
                            if lines[k][0] - lines[k - 1][1] > 18.0:
                                break
                            if len(lines[k][2]) >= 2:
                                found_next = True
                                break
                        if found_next or (stxt.startswith("(") and stxt.endswith(")") and len(stxt) <= 5):
                            band.append(lines[j])
                            j += 1
                            continue
                    break

            pre_hdr = (
                lines[i - 1]
                if (i > 0 and (lines[i][0] - lines[i - 1][1]) < 18.0 and out_blocks and not out_blocks[-1].startswith("|"))
                else None
            )
            md_tbl, consumed_pre_hdr = _segments_to_markdown_table(band, pre_header_line=pre_hdr)
            if md_tbl:
                if consumed_pre_hdr and out_blocks:
                    last_lines = out_blocks[-1].splitlines()
                    if last_lines and last_lines[-1].strip() == pre_hdr[2][0][2].strip():
                        if len(last_lines) == 1:
                            out_blocks.pop()
                        else:
                            out_blocks[-1] = "\n".join(last_lines[:-1])
                out_blocks.append(md_tbl)
                i = j
                continue

        prose = " ".join(s[2] for s in lines[i][2])
        if out_blocks and not out_blocks[-1].startswith("|") and (lines[i][0] - lines[i - 1][1]) < 8.0:
            out_blocks[-1] += "\n" + prose
        else:
            out_blocks.append(prose)
        i += 1
    return out_blocks


def _extract_pdf_page_layout(page) -> str:
    """Extract clean reading-order Markdown from a PDF page, supporting 2-column prose and borderless tables."""
    words = page.get_text("words")
    if not words:
        return ""
    lines = _extract_pdf_segments(words)

    mid_min = page.rect.width * 0.42
    mid_max = page.rect.width * 0.58
    two_col_lines = []
    for idx, (_y0, _y1, segs) in enumerate(lines):
        if len(segs) == 2:
            s0, s1 = segs
            if s0[1] < mid_max and s1[0] > mid_min and (s1[0] - s0[1]) >= 12.0:
                if (s0[1] - s0[0]) > page.rect.width * 0.18 and (s1[1] - s1[0]) > page.rect.width * 0.18:
                    if not any(ch in s1[2] for ch in ("$", "%")) and sum(c.isalpha() for c in s1[2]) > sum(c.isdigit() for c in s1[2]):
                        two_col_lines.append(idx)

    if len(two_col_lines) >= 5:
        first_2c, last_2c = two_col_lines[0], two_col_lines[-1]
        gutter_x = (
            max(lines[k][2][0][1] for k in two_col_lines) + min(lines[k][2][1][0] for k in two_col_lines)
        ) / 2.0
        blocks: List[str] = []
        if first_2c > 0:
            blocks.extend(_extract_pdf_blocks_from_lines(lines[:first_2c], page.rect.width))
        y_top, y_bot = lines[first_2c][0] - 1.0, lines[last_2c][1] + 1.0
        left_words = [w for w in words if y_top <= w[1] <= y_bot and (w[0] + w[2]) / 2.0 < gutter_x]
        right_words = [w for w in words if y_top <= w[1] <= y_bot and (w[0] + w[2]) / 2.0 >= gutter_x]
        if left_words:
            blocks.extend(_extract_pdf_blocks_from_lines(_extract_pdf_segments(left_words), gutter_x))
        if right_words:
            blocks.extend(_extract_pdf_blocks_from_lines(_extract_pdf_segments(right_words), page.rect.width - gutter_x))
        if last_2c + 1 < len(lines):
            blocks.extend(_extract_pdf_blocks_from_lines(lines[last_2c + 1 :], page.rect.width))
        return "\n\n".join(blocks)

    return "\n\n".join(_extract_pdf_blocks_from_lines(lines, page.rect.width))


def extract_docx_content(docx_path: Path) -> Tuple[str, str]:
    """Extract structured text and tables from Word (.docx) using python-docx."""
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is required for Word document extraction. Run: pip install python-docx")

    doc = docx.Document(str(docx_path))
    doc_title = docx_path.stem.replace("_", " ").replace("-", " ").title()

    lines = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        style_name = p.style.name.lower() if p.style else ""
        if "heading 1" in style_name:
            lines.append(f"\n# {txt}\n")
        elif "heading 2" in style_name:
            lines.append(f"\n## {txt}\n")
        elif "heading 3" in style_name:
            lines.append(f"\n### {txt}\n")
        elif "heading 4" in style_name:
            lines.append(f"\n#### {txt}\n")
        else:
            lines.append(txt)

    for table in doc.tables:
        lines.append("")
        for row_idx, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("\n", " ").replace("|", "\\|") for cell in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
            if row_idx == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        lines.append("")

    return doc_title, "\n\n".join(lines)


def extract_text_file(txt_path: Path) -> Tuple[str, str]:
    """Extract text from plain text or markdown file."""
    doc_title = txt_path.stem.replace("_", " ").replace("-", " ").title()
    try:
        content = txt_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = txt_path.read_text(encoding="latin-1", errors="replace")
    return doc_title, content


# =====================================================================
# Clause & Structural Boundary Parser
# =====================================================================

RE_CHAPTER = re.compile(
    r"^(?:#+\s*)?(?:CHAPTER|Chapter)\s+([0-9IVXLCDM]+)(?:[:.\s\-\–\—]+([A-Z][A-Za-z0-9\s,'\*\-]+))?$",
    re.IGNORECASE,
)
RE_ARTICLE = re.compile(
    r"^(?:#+\s*)?(?:ARTICLE|Article)\s+([0-9IVXLCDM]+)[:.\s\-–—]*(.*)$",
    re.IGNORECASE,
)
RE_SECTION = re.compile(
    r"^(?:#+\s*)?(?:SECTION|Section|SEC\.|Sec\.|Sec|SEC|§|Clause|CLAUSE)\s*([0-9]+(?:[-.][0-9]+)*[a-z]?)[.:\s\-–—]*(.*)$",
    re.IGNORECASE,
)
RE_TECH_SPEC = re.compile(
    r"^(?:#+\s*)?([0-9]+(?:\.[0-9]+)+)\s+([A-Z].*)$"
)
RE_REQ_TAG = re.compile(
    r"^(?:#+\s*)?(\[?(?:REQ|SPEC|RFC)-[0-9]+\]?)[:.\s\-–—]*(.*)$",
    re.IGNORECASE,
)
RE_SCHEDULE = re.compile(
    r"^(?:#+\s*)?(?:SCHEDULE|EXHIBIT|APPENDIX|ATTACHMENT)\s+([A-Z0-9]+)[:.\s\-–—]*(.*)$",
    re.IGNORECASE,
)
RE_SUBSECTION = re.compile(
    r"^(?:#+\s*)?\(([a-zA-Z0-9]+)\)\s+([A-Z].*)$"
)
RE_MD_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")


def identify_heading(line: str) -> Optional[Tuple[int, str]]:
    """Check if a line represents a major structural boundary."""
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return None

    m_md = RE_MD_HEADER.match(stripped)
    if m_md:
        level = len(m_md.group(1))
        title = m_md.group(2).strip()
        return min(level, 3), title

    m_chap = RE_CHAPTER.match(stripped)
    if m_chap:
        chap_num = m_chap.group(1).strip()
        chap_title = (m_chap.group(2) or "").strip()
        title = f"Chapter {chap_num}" + (f": {chap_title.title()}" if chap_title else "")
        return 1, title

    m_art = RE_ARTICLE.match(stripped)
    if m_art:
        art_num = m_art.group(1).strip()
        art_title = m_art.group(2).strip()
        title = f"Article {art_num}" + (f": {art_title}" if art_title else "")
        return 2, title

    m_sch = RE_SCHEDULE.match(stripped)
    if m_sch:
        sch_num = m_sch.group(1).strip()
        sch_title = m_sch.group(2).strip()
        title = f"Schedule {sch_num}" + (f": {sch_title}" if sch_title else "")
        return 1, title

    m_sec = RE_SECTION.match(stripped)
    if m_sec:
        sec_num = m_sec.group(1).strip()
        sec_title = m_sec.group(2).strip()
        depth = sec_num.count(".") + sec_num.count("-")
        level = 2 if depth <= 1 else 3
        title = f"Sec. {sec_num}" + (f": {sec_title}" if sec_title else "")
        return level, title

    m_tech = RE_TECH_SPEC.match(stripped)
    if m_tech:
        num = m_tech.group(1).strip()
        title_text = m_tech.group(2).strip()
        depth = num.count(".")
        level = 2 if depth == 1 else 3
        return level, f"{num} {title_text}"

    m_req = RE_REQ_TAG.match(stripped)
    if m_req:
        tag = m_req.group(1).strip()
        desc = m_req.group(2).strip()
        title = f"{tag}: {desc}" if desc else tag
        return 3, title

    m_sub = RE_SUBSECTION.match(stripped)
    if m_sub:
        marker = m_sub.group(1).strip()
        text = m_sub.group(2).strip()
        # Truncate long outline descriptions to a neat heading title
        clean_text = text.split(".")[0].strip() if "." in text else text
        if len(clean_text) > 60:
            clean_text = clean_text[:57] + "..."
        title = f"({marker}) {clean_text}"
        return 4, title

    return None


# =====================================================================
# Clause-Aware Chunking Engine & Tabular Serialization
# =====================================================================

def enrich_wikilinks_appositive(text: str) -> str:
    """Enrich piped wikilinks [[Target|Alias]] into appositive prose 'Alias (Target)', and [[Target]] into 'Target'."""
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        if "|" in inner:
            target, alias = inner.split("|", 1)
            target = target.strip()
            alias = alias.strip()
            if target and alias and target.lower() != alias.lower():
                return f"{alias} ({target})"
            return alias or target
        return inner

    return re.sub(r"\[\[([^\]]+)\]\]", repl, text)


def is_pipe_table_row(line: str) -> bool:
    """Return True if line is a Markdown pipe table row."""
    s = line.strip()
    return bool(s.startswith("|") and s.endswith("|") and len(s) > 1)


def detect_table_block(lines: List[str], start_idx: int) -> Optional[Tuple[int, List[str]]]:
    """Detect if lines[start_idx] begins a Markdown pipe table or plaintext tabular schedule.

    Returns (end_idx, table_lines) if detected, or None.
    """
    if start_idx >= len(lines):
        return None
    line = lines[start_idx]
    stripped = line.strip()

    # 1. Check for Markdown Pipe Table
    if is_pipe_table_row(line):
        table_lines = [line]
        j = start_idx + 1
        while j < len(lines) and is_pipe_table_row(lines[j]):
            table_lines.append(lines[j])
            j += 1
        if len(table_lines) >= 3 and re.match(r"^\s*\|[\s:\-\|]+\|\s*$", table_lines[1]):
            return (j, table_lines)

    # 2. Check for Plaintext Schedule (e.g. Offense Penalty)
    if re.match(
        r"^\s*(?:Offense|Violation|Item|Tier|Category|Class|Level|Step|Service|Permit|Fee)\s+(?:Penalty|Fine|Fee|Amount|Sanction|Charge|Rate|Cost|Deposit|Bond|Days)\s*$",
        stripped,
        re.IGNORECASE,
    ):
        sched_lines = [line]
        j = start_idx + 1
        entries_count = 0
        while j < len(lines):
            row = lines[j].strip()
            if not row:
                if j + 1 < len(lines):
                    next_row = lines[j + 1].strip()
                    if re.match(r"^([0-9]+(?:st|nd|rd|th)|[A-Za-z0-9\.\-]+)\s{2,}(\$[\s0-9\.,]+.*)$", next_row):
                        sched_lines.append(lines[j])
                        j += 1
                        continue
                break
            m = re.match(r"^([0-9]+(?:st|nd|rd|th)|[A-Za-z0-9\.\-]+)\s{2,}(\$[\s0-9\.,]+.*)$", row)
            if m:
                entries_count += 1
                sched_lines.append(lines[j])
                j += 1
            else:
                break
        if entries_count >= 2:
            return (j, sched_lines)

    return None


def serialize_tabular_data_with_metadata(text: str) -> Tuple[str, List[str], int, int, str]:
    """Detect Markdown pipe tables and plain text schedules.
    
    Returns (serialized_text, headers, rows, cols, chunk_type).
    """
    headers: List[str] = []
    num_rows: int = 0
    num_cols: int = 0
    detected_type: str = "clause"

    # Pre-check for badges if text was already serialized or sliced
    m_badge = re.search(r"> \*\*(?:Table|Schedule)(?: Context)?\*\*: \[(?:Table|Schedule): (\d+) rows? [×x] (\d+) columns?", text)
    if not m_badge:
        m_badge = re.search(r"> \*\*(?:Table|Schedule)\*\*: (\d+) rows? [×x] (\d+) columns?", text)
    if m_badge:
        num_rows = int(m_badge.group(1))
        num_cols = int(m_badge.group(2))
        detected_type = "table"

    if "<!-- TABLE_DATA:" in text:
        for line in text.splitlines():
            if is_pipe_table_row(line):
                raw_h = [c.strip() for c in line.strip().strip("|").split("|")]
                headers = [h for h in raw_h if h and not set(h) <= {":", "-"}]
                if headers:
                    detected_type = "table"
                    if not num_cols:
                        num_cols = len(headers)
                    break
        return text, headers, num_rows, num_cols, detected_type

    lines = text.splitlines()
    output_lines: List[str] = []
    i = 0
    while i < len(lines):
        tbl_info = detect_table_block(lines, i)
        if tbl_info:
            end_idx, table_lines = tbl_info

            # Check if Markdown Pipe Table
            if is_pipe_table_row(table_lines[0]):
                detected_type = "table"
                raw_headers = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
                headers = [c for c in raw_headers if c]
                data_rows = []
                for row_str in table_lines[2:]:
                    cells = [c.strip() for c in row_str.strip().strip("|").split("|")]
                    data_rows.append(cells)

                num_rows = len(data_rows)
                num_cols = len(raw_headers)
                dim_badge = f"> **Table**: {num_rows} rows × {num_cols} columns"

                table_data = None
                if len(raw_headers) == 2:
                    table_data = {r[0]: (r[1] if len(r) > 1 else "") for r in data_rows if r}
                    if not (output_lines and output_lines[-1].strip().startswith("> **Table")):
                        output_lines.append(dim_badge)
                    output_lines.extend(table_lines)
                    if table_data:
                        output_lines.append("\nKey-Value Schedule:")
                        for k, v in table_data.items():
                            if k and v:
                                output_lines.append(f"- **{k}**: {v}")
                        json_str = json.dumps(table_data, indent=2)
                        output_lines.append(f"<!-- TABLE_DATA:\n{json_str}\n-->")
                else:
                    # Multi-column grids (> 2 columns) remain pure, unsliced,
                    # grid-padded Markdown pipe tables without row-level card and JSON bloat.
                    if not (output_lines and output_lines[-1].strip().startswith("> **Table")):
                        output_lines.append(dim_badge)
                    output_lines.extend(table_lines)
                i = end_idx
                continue
            else:
                # Plain text schedule
                detected_type = "schedule"
                hdr_words = table_lines[0].strip().split()
                if len(hdr_words) >= 2:
                    headers = hdr_words
                else:
                    headers = ["Offense", "Penalty"]
                num_cols = len(headers)
                entries: Dict[str, str] = {}
                data_rows_count = 0
                for row in table_lines[1:]:
                    m = re.match(r"^([0-9]+(?:st|nd|rd|th)|[A-Za-z0-9\.\-]+)\s{2,}(\$[\s0-9\.,]+.*)$", row.strip())
                    if m:
                        entries[m.group(1).strip()] = m.group(2).strip()
                        data_rows_count += 1

                num_rows = data_rows_count
                dim_badge = f"> **Schedule**: {data_rows_count} rows × 2 columns"
                if not (output_lines and output_lines[-1].strip().startswith("> **Schedule") or (output_lines and output_lines[-1].strip().startswith("> **Table"))):
                    output_lines.append(dim_badge)
                output_lines.extend(table_lines)
                if entries:
                    output_lines.append("\nKey-Value Schedule:")
                    for k, v in entries.items():
                        output_lines.append(f"- **{k}**: {v}")

                    json_str = json.dumps(entries, indent=2)
                    output_lines.append(f"<!-- TABLE_DATA:\n{json_str}\n-->")
                i = end_idx
                continue

        output_lines.append(lines[i])
        i += 1

    res = "\n".join(output_lines)
    if text.endswith("\n"):
        res += "\n"

    if not headers:
        for line in lines:
            if is_pipe_table_row(line):
                raw_h = [c.strip() for c in line.strip().strip("|").split("|")]
                headers = [h for h in raw_h if h and not set(h) <= {":", "-"}]
                if headers:
                    detected_type = "table"
                    if not num_cols:
                        num_cols = len(headers)
                    break

    return res, headers, num_rows, num_cols, detected_type


def serialize_tabular_data(text: str) -> str:
    """Detect Markdown pipe tables and plain text schedules, appending machine-readable and linearized key-value blocks."""
    res, _, _, _, _ = serialize_tabular_data_with_metadata(text)
    return res


def slice_massive_table(table_lines: List[str], max_chars: int) -> List[str]:
    """Slice a massive table that exceeds max_chars by complete rows, duplicating the column header and dimension context."""
    badge_overhead = 120
    effective_max = max(max_chars - badge_overhead, 200)

    if is_pipe_table_row(table_lines[0]) and len(table_lines) >= 3:
        header_lines = table_lines[:2]
        header_text = "\n".join(header_lines) + "\n"
        data_rows = table_lines[2:]
        raw_headers = [c.strip() for c in header_lines[0].strip().strip("|").split("|")]
        num_cols = len(raw_headers)
        total_rows = len(data_rows)

        slices = []
        curr_slice_rows = []
        start_row_idx = 1
        curr_len = len(header_text)

        for idx, row in enumerate(data_rows, 1):
            row_len = len(row) + 1
            if curr_len + row_len > effective_max and curr_slice_rows:
                end_row_idx = idx - 1
                badge = f"> **Table Context**: [Table: {total_rows} rows × {num_cols} columns | Showing rows {start_row_idx}–{end_row_idx} of {total_rows}]\n\n"
                slices.append(badge + header_text + "\n".join(curr_slice_rows))
                curr_slice_rows = [row]
                curr_len = len(header_text) + row_len
                start_row_idx = idx
            else:
                curr_slice_rows.append(row)
                curr_len += row_len

        if curr_slice_rows:
            end_row_idx = total_rows
            badge = f"> **Table Context**: [Table: {total_rows} rows × {num_cols} columns | Showing rows {start_row_idx}–{end_row_idx} of {total_rows}]\n\n"
            slices.append(badge + header_text + "\n".join(curr_slice_rows))
        return slices
    else:
        header_line = table_lines[0]
        data_rows = table_lines[1:]
        total_rows = len(data_rows)
        slices = []
        curr_slice_rows = []
        start_row_idx = 1
        curr_len = len(header_line) + 1

        for idx, row in enumerate(data_rows, 1):
            row_len = len(row) + 1
            if curr_len + row_len > effective_max and curr_slice_rows:
                end_row_idx = idx - 1
                badge = f"> **Schedule Context**: [Schedule: {total_rows} rows × 2 columns | Showing entries {start_row_idx}–{end_row_idx} of {total_rows}]\n\n"
                slices.append(badge + header_line + "\n" + "\n".join(curr_slice_rows))
                curr_slice_rows = [row]
                curr_len = len(header_line) + 1 + row_len
                start_row_idx = idx
            else:
                curr_slice_rows.append(row)
                curr_len += row_len

        if curr_slice_rows:
            end_row_idx = total_rows
            badge = f"> **Schedule Context**: [Schedule: {total_rows} rows × 2 columns | Showing entries {start_row_idx}–{end_row_idx} of {total_rows}]\n\n"
            slices.append(badge + header_line + "\n" + "\n".join(curr_slice_rows))
        return slices


def chunk_document_text(
    document_title: str,
    raw_text: str,
    target_size: int = 1500,
    max_size: int = 3000,
) -> List[DocumentChunk]:
    """Parse document into semantic, clause-preserved chunks with hierarchical breadcrumbs."""
    lines = raw_text.splitlines()
    chunks: List[DocumentChunk] = []

    hierarchy: List[str] = [document_title]
    current_section_title: str = "Preamble / General"

    current_block_lines: List[str] = []
    in_code = False

    def build_breadcrumbs() -> str:
        return " > ".join(hierarchy)

    def finalize_chunk(content_str: str, chunk_type: str = "clause"):
        nonlocal chunks
        clean_content = content_str.strip()
        if not clean_content:
            return

        def add_chunk(raw_text: str, title: str):
            serialized, t_headers, t_rows, t_cols, t_type = serialize_tabular_data_with_metadata(raw_text)
            c_type = t_type if t_type in ("table", "schedule") else (
                "table" if ("<!-- TABLE_DATA:" in serialized or chunk_type in ("table", "schedule") or any(is_pipe_table_row(l) for l in raw_text.splitlines())) else chunk_type
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=len(chunks) + 1,
                    document_title=document_title,
                    section_title=title,
                    hierarchy=list(hierarchy),
                    breadcrumbs=build_breadcrumbs(),
                    content=serialized,
                    char_count=len(serialized),
                    token_estimate=int(len(serialized) / 4),
                    chunk_type=c_type,
                    table_headers=t_headers,
                    table_rows=t_rows,
                    table_cols=t_cols,
                )
            )

        if chunk_type == "table" or "Showing rows" in clean_content or "Showing entries" in clean_content:
            add_chunk(clean_content, current_section_title)
            return

        if len(clean_content) > max_size:
            # Table-aware paragraph splitting: never split inside atomic tables
            c_lines = clean_content.splitlines()
            sub_paragraphs: List[str] = []
            pj = 0
            curr_prose_lines: List[str] = []

            while pj < len(c_lines):
                tbl = detect_table_block(c_lines, pj)
                if tbl:
                    t_end, t_lines = tbl
                    if curr_prose_lines:
                        p_txt = "\n".join(curr_prose_lines).strip()
                        if p_txt:
                            sub_paragraphs.append(p_txt)
                        curr_prose_lines = []
                    t_txt = "\n".join(t_lines).strip()
                    if t_txt:
                        if len(t_txt) > max_size:
                            sub_paragraphs.extend(slice_massive_table(t_lines, max_size))
                        else:
                            sub_paragraphs.append(t_txt)
                    pj = t_end
                    continue

                line_j = c_lines[pj]
                if not line_j.strip():
                    if curr_prose_lines:
                        p_txt = "\n".join(curr_prose_lines).strip()
                        if p_txt:
                            sub_paragraphs.append(p_txt)
                        curr_prose_lines = []
                else:
                    curr_prose_lines.append(line_j)
                pj += 1

            if curr_prose_lines:
                p_txt = "\n".join(curr_prose_lines).strip()
                if p_txt:
                    sub_paragraphs.append(p_txt)

            buffer_paras = []
            curr_len = 0
            part_idx = 1

            for p in sub_paragraphs:
                p_len = len(p)
                if p_len > max_size:
                    for slice_start in range(0, p_len, max_size):
                        slice_p = p[slice_start : slice_start + max_size]
                        sub_title = f"{current_section_title} (Part {part_idx})"
                        add_chunk(slice_p, sub_title)
                        part_idx += 1
                    continue

                if curr_len + p_len > max_size and buffer_paras:
                    part_content = "\n\n".join(buffer_paras)
                    sub_title = f"{current_section_title} (Part {part_idx})"
                    add_chunk(part_content, sub_title)
                    buffer_paras = [p]
                    curr_len = p_len
                    part_idx += 1
                else:
                    buffer_paras.append(p)
                    curr_len += p_len + 2

            if buffer_paras:
                part_content = "\n\n".join(buffer_paras)
                sub_title = f"{current_section_title} (Part {part_idx})" if part_idx > 1 else current_section_title
                add_chunk(part_content, sub_title)
        else:
            add_chunk(clean_content, current_section_title)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            current_block_lines.append(line)
            i += 1
            continue

        if in_code:
            current_block_lines.append(line)
            i += 1
            continue

        # Prevent false structural breaks on wrapped lines within paragraphs
        is_continuation = False
        if i > 0 and not stripped.startswith("#"):
            prev_stripped = lines[i - 1].strip()
            if prev_stripped:
                if prev_stripped[-1] in (",", ";", "-", "–", "—") or (
                    prev_stripped[-1].isalnum()
                    and not prev_stripped.endswith((".", ":", "?", "!", '"""', "'''", "```"))
                ):
                    is_continuation = True

        heading = None if is_continuation else identify_heading(line)
        if heading:
            level, title = heading

            if current_block_lines:
                finalize_chunk("\n".join(current_block_lines))
                current_block_lines = []

            if level == 1:
                hierarchy = [document_title, title]
                current_section_title = title
            elif level == 2:
                if len(hierarchy) > 1:
                    hierarchy = hierarchy[:2] + [title]
                else:
                    hierarchy = [document_title, title]
                current_section_title = title
            elif level == 3:
                if len(hierarchy) >= 3:
                    hierarchy = hierarchy[:2] + [title]
                else:
                    hierarchy.append(title)
                current_section_title = title
            elif level >= 4:
                if len(hierarchy) >= 4:
                    hierarchy = hierarchy[:3] + [title]
                else:
                    hierarchy.append(title)
                current_section_title = title

            current_block_lines.append(line)
            i += 1
            continue

        # Atomic table & schedule ingestion: keep entire table together in the current block
        tbl_info = detect_table_block(lines, i)
        if tbl_info:
            end_idx, table_lines = tbl_info
            tbl_len = sum(len(l) for l in table_lines) + len(table_lines)
            curr_len = sum(len(l) for l in current_block_lines) + len(current_block_lines)

            # Pre-table flush: isolate distinct tables from each other, or flush if accumulated + table > max_size
            has_existing_table = any(is_pipe_table_row(l) for l in current_block_lines)
            if current_block_lines and (has_existing_table or curr_len + tbl_len > max_size):
                # Preserve introductory lead-in sentence/paragraph with the table
                k = len(current_block_lines) - 1
                while k >= 0 and not current_block_lines[k].strip():
                    k -= 1
                lead_end = k + 1
                while k >= 0 and current_block_lines[k].strip():
                    k -= 1
                candidate_lead_in = current_block_lines[k + 1 : lead_end]
                candidate_text = "\n".join(candidate_lead_in).strip()
                if (
                    not has_existing_table
                    and candidate_text
                    and len(candidate_text) <= 500
                    and (candidate_text.endswith(":") or k >= 0)
                ):
                    prose_lines = current_block_lines[: k + 1]
                    lead_in_lines = candidate_lead_in
                else:
                    prose_lines = current_block_lines
                    lead_in_lines = []

                if prose_lines:
                    finalize_chunk("\n".join(prose_lines), chunk_type="table" if has_existing_table else "clause")
                current_block_lines = list(lead_in_lines)

            if tbl_len > max_size:
                # Massive table: slice across chunks, attaching any lead-in to the first slice
                lead_prefix = "\n".join(current_block_lines).strip()
                current_block_lines = []
                for idx_s, s in enumerate(slice_massive_table(table_lines, max_size)):
                    chunk_text = f"{lead_prefix}\n\n{s}" if (idx_s == 0 and lead_prefix) else s
                    finalize_chunk(chunk_text, chunk_type="table")
                i = end_idx
                continue
            else:
                current_block_lines.extend(table_lines)
                i = end_idx
                # Finalize table chunk if target size reached
                if sum(len(l) for l in current_block_lines) >= target_size:
                    finalize_chunk("\n".join(current_block_lines), chunk_type="table")
                    current_block_lines = []
                continue

        current_block_lines.append(line)

        current_len = sum(len(l) for l in current_block_lines)
        if current_len >= target_size and not in_code:
            if not stripped and len(current_block_lines) > 2:
                finalize_chunk("\n".join(current_block_lines))
                current_block_lines = []

        i += 1

    if current_block_lines:
        finalize_chunk("\n".join(current_block_lines))

    return chunks


# =====================================================================
# Exporters: Enriched Markdown, Split Sections, JSONL
# =====================================================================

def export_enriched_markdown(chunks: List[DocumentChunk], out_file: Path):
    """Generate a single master markdown file formatted for AnythingLLM."""
    lines = []
    doc_title = chunks[0].document_title if chunks else "Document"
    lines.append(f"# {doc_title}\n")
    lines.append(f"> **Processed by**: Clause-Aware Legal & Tech-Spec Chunker\n")
    lines.append(f"> **Total Semantic Chunks**: {len(chunks)}\n\n---\n")

    for chunk in chunks:
        lines.append(f"## {chunk.section_title}")
        lines.append(f"> **Context Hierarchy**: {chunk.breadcrumbs}\n")
        lines.append(f"{chunk.content}\n")
        lines.append("\n---\n")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def export_split_sections(chunks: List[DocumentChunk], out_dir: Path):
    """Generate individual section markdown files in a subfolder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_title = chunks[0].document_title if chunks else "Document"

    for idx, chunk in enumerate(chunks, 1):
        clean_title = re.sub(r'[\\/*?:"<>|]', "", chunk.section_title).strip()
        clean_title = clean_title.replace(" ", "_")[:50]
        filename = f"{idx:02d}_{clean_title}.md"
        section_path = out_dir / filename

        content_parts = [
            f"# {chunk.section_title}\n",
            f"> **Context Hierarchy**: {chunk.breadcrumbs}\n",
            f"> **Document**: {doc_title}\n\n",
            chunk.content,
            "\n",
        ]
        section_path.write_text("\n".join(content_parts), encoding="utf-8")


def export_jsonl(chunks: List[DocumentChunk], out_file: Path):
    """Export machine-readable JSONL chunks for vector DBs and Python pipelines."""
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            data = asdict(chunk)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# =====================================================================
# Core Document & Batch Processors
# =====================================================================

def process_document(
    input_source: str,
    output_dir: Optional[str] = None,
    export_all: bool = False,
    split_sections: bool = False,
    output_jsonl: bool = False,
    target_size: int = 1500,
    max_size: int = 3000,
    custom_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single document from local file, web URL, or text stream."""
    is_url = input_source.startswith("http://") or input_source.startswith("https://")
    is_raw_text = "\n" in input_source and not Path(input_source).exists()

    if is_url:
        print(f"[EXTRACTING] Fetching web URL: {input_source}...")
        doc_title, raw_text = fetch_url_content(input_source)
        base_name = re.sub(r'[\\/*?:"<>|]', "", doc_title).strip().replace(" ", "_")[:40]
        out_base_dir = Path(output_dir) if output_dir else Path.cwd()
    elif is_raw_text:
        doc_title = custom_title or "Clipboard Document"
        raw_text = input_source
        base_name = "clipboard_snippet"
        out_base_dir = Path(output_dir) if output_dir else Path.cwd()
    else:
        src_path = Path(input_source).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        print(f"[EXTRACTING] Processing local file: {src_path.name}...")
        ext = src_path.suffix.lower()
        if ext == ".pdf":
            doc_title, raw_text = extract_pdf_content(src_path)
        elif ext == ".docx":
            doc_title, raw_text = extract_docx_content(src_path)
        elif ext in (".md", ".txt", ".markdown", ".rfc"):
            doc_title, raw_text = extract_text_file(src_path)
        else:
            doc_title, raw_text = extract_text_file(src_path)

        base_name = src_path.stem
        out_base_dir = Path(output_dir) if output_dir else src_path.parent

    print(f"  • Document Title : {doc_title}")
    print(f"  • Extracted Text : {len(raw_text):,} characters ({len(raw_text.splitlines()):,} lines)")

    print(f"[PARSING] Applying clause & spec boundary detection...")
    chunks = chunk_document_text(
        document_title=doc_title,
        raw_text=raw_text,
        target_size=target_size,
        max_size=max_size,
    )
    print(f"  • Generated      : \033[92m{len(chunks)} clause-preserved semantic chunks\033[0m")

    out_base_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    if export_all:
        do_enriched = True
        do_split = True
        do_jsonl = True
    else:
        do_split = split_sections
        do_jsonl = output_jsonl
        do_enriched = not (split_sections and not export_all and not output_jsonl) or not (do_split or do_jsonl)

    if do_enriched or export_all or (not do_split and not do_jsonl):
        enriched_path = out_base_dir / f"{base_name}_enriched.md"
        export_enriched_markdown(chunks, enriched_path)
        generated_files.append(("Master Enriched Markdown (AnythingLLM)", enriched_path))

    if do_split or export_all:
        sections_dir = out_base_dir / f"{base_name}_sections"
        export_split_sections(chunks, sections_dir)
        generated_files.append(("Individual Section Files", sections_dir))

    if do_jsonl or export_all:
        jsonl_path = out_base_dir / f"{base_name}_chunks.jsonl"
        export_jsonl(chunks, jsonl_path)
        generated_files.append(("Structured JSONL (Vector DB Ready)", jsonl_path))

    for label, path_obj in generated_files:
        print(f"  • \033[1;36m{label}\033[0m: {path_obj.name}")

    return {
        "title": doc_title,
        "base_name": base_name,
        "chunk_count": len(chunks),
        "files": [p[1] for p in generated_files],
    }


def process_batch_directory(
    dir_path: Path,
    output_dir: Optional[str] = None,
    export_all: bool = False,
    split_sections: bool = False,
    output_jsonl: bool = False,
    target_size: int = 1500,
    max_size: int = 3000,
):
    """Recursively discover and process all supported documents in a folder."""
    valid_exts = {".pdf", ".docx", ".md", ".txt", ".rfc"}
    files = [
        p for p in dir_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in valid_exts
        and not p.name.endswith("_enriched.md")
        and not "_sections" in str(p)
    ]

    if not files:
        print(f"\033[93m[WARN]: No supported documents (.pdf, .docx, .md, .txt) found in {dir_path}\033[0m")
        return

    print(f"\n\033[96m======================================================================\033[0m")
    print(f"\033[96m          BATCH PROCESSING {len(files)} DOCUMENTS IN {dir_path.name}  \033[0m")
    print(f"\033[96m======================================================================\033[0m\n")

    target_out_dir = Path(output_dir) if output_dir else dir_path / "enriched_batch"
    target_out_dir.mkdir(parents=True, exist_ok=True)

    master_chunks_raw: List[str] = []

    for idx, f in enumerate(files, 1):
        print(f"\n\033[1;33m[{idx}/{len(files)}] Document: {f.name}\033[0m")
        try:
            res = process_document(
                input_source=str(f),
                output_dir=str(target_out_dir),
                export_all=export_all,
                split_sections=split_sections,
                output_jsonl=output_jsonl,
                target_size=target_size,
                max_size=max_size,
            )
            if export_all or output_jsonl:
                j_file = target_out_dir / f"{res['base_name']}_chunks.jsonl"
                if j_file.is_file():
                    with open(j_file, "r", encoding="utf-8") as jf:
                        for line in jf:
                            if line.strip():
                                master_chunks_raw.append(line.strip())
        except Exception as e:
            print(f"  \033[91m[Error processing {f.name}]: {e}\033[0m")

    if master_chunks_raw:
        master_jsonl_path = target_out_dir / "all_documents_combined_chunks.jsonl"
        with open(master_jsonl_path, "w", encoding="utf-8") as mf:
            for c_str in master_chunks_raw:
                mf.write(c_str + "\n")

        print(f"\n\033[92m======================================================================\033[0m")
        print(f"\033[92m                  BATCH EXPORT COMPLETED SUCCESSFULLY                 \033[0m")
        print(f"\033[92m======================================================================\033[0m")
        print(f"  • Processed Documents : {len(files)}")
        print(f"  • Master Combined     : \033[1;36m{master_jsonl_path}\033[0m ({len(master_chunks_raw):,} chunks)")
        print(f"  • Output Directory    : {target_out_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clause-Aware Legal & Tech-Spec Document Chunker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chunk contract.pdf                        # Default: Master enriched .md for AnythingLLM
  chunk "https://example.com/terms" -a      # Web URL: Generate ALL formats (.md, sections, .jsonl)
  chunk ./contracts/ -a                     # Batch Mode: Process entire folder & build master dataset
  chunk -c -a                               # Clipboard: Process copied text or URL from clipboard
  chunk architecture_spec.docx -s           # Split into individual section files
        """,
    )
    parser.add_argument("source", nargs="?", default="", help="Path to file/folder, Web URL, or leave empty with -c")
    parser.add_argument("-c", "--clipboard", action="store_true", help="Ingest URL or contract text directly from clipboard")
    parser.add_argument("-o", "--output-dir", type=str, default="", help="Output directory (defaults to file location)")
    parser.add_argument("-a", "--all", action="store_true", help="Generate all formats: enriched .md, sections/, and .jsonl")
    parser.add_argument("-s", "--split", action="store_true", help="Generate individual section files in <name>_sections/")
    parser.add_argument("-j", "--jsonl", action="store_true", help="Generate structured JSONL dataset in <name>_chunks.jsonl")
    parser.add_argument("--target-size", type=int, default=1500, help="Target chunk size in characters (default: 1500)")
    parser.add_argument("--max-size", type=int, default=3000, help="Maximum chunk size before paragraph split (default: 3000)")

    args = parser.parse_args()

    input_source = args.source.strip()
    if args.clipboard or input_source == "-c":
        clip = get_clipboard_text()
        if not clip:
            print("\033[91m[Error]: Clipboard is empty.\033[0m", file=sys.stderr)
            sys.exit(1)
        input_source = clip
        print("\033[92m[Loaded content from clipboard]\033[0m")

    if not input_source:
        parser.print_help()
        sys.exit(0)

    try:
        source_path = Path(input_source)
        if source_path.is_dir():
            process_batch_directory(
                dir_path=source_path,
                output_dir=args.output_dir or None,
                export_all=args.all,
                split_sections=args.split,
                output_jsonl=args.jsonl,
                target_size=args.target_size,
                max_size=args.max_size,
            )
        else:
            process_document(
                input_source=input_source,
                output_dir=args.output_dir or None,
                export_all=args.all,
                split_sections=args.split,
                output_jsonl=args.jsonl,
                target_size=args.target_size,
                max_size=args.max_size,
            )
    except Exception as e:
        print(f"\n\033[91m[Error]: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
