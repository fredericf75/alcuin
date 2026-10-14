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
from dataclasses import asdict, dataclass
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
    chunk_type: str = "clause"  # clause, table, code, preamble


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


def extract_pdf_content(pdf_path: Path) -> Tuple[str, str]:
    """Extract structured text and tables from PDF, with auto-OCR for scanned pages."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            raise RuntimeError("pymupdf is required for PDF extraction. Run: pip install pymupdf")

    tess_exe = find_tesseract_binary()
    if tess_exe:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tess_exe
        except Exception:
            pass

    doc = fitz.open(str(pdf_path))
    doc_title = doc.metadata.get("title") if doc.metadata else ""
    if not doc_title or len(doc_title.strip()) < 3:
        doc_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    pages_text = []
    scanned_pages = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()

        # Detect scanned page (less than 40 chars of selectable text with images/drawings)
        if len(text) < 40 and (len(page.get_images()) > 0 or len(page.get_drawings()) > 0):
            scanned_pages.append(page_idx + 1)
            if tess_exe:
                try:
                    import pytesseract
                    from PIL import Image
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        print(f"  \033[92m[OCR]\033[0m Page {page_idx + 1}: Scanned image recognized ({len(ocr_text):,} characters)")
                        text = ocr_text
                    else:
                        print(f"  \033[90m[OCR]\033[0m Page {page_idx + 1}: No text detected via OCR")
                except Exception as e:
                    print(f"  \033[91m[OCR Error]\033[0m Page {page_idx + 1}: {e}")

        if text:
            pages_text.append(text)

    if scanned_pages and not tess_exe:
        print(f"\n  \033[93m[WARN]: Scanned PDF detected!\033[0m Pages {scanned_pages} appear to be image-based scans.")
        print(f"  \033[90m(Install Tesseract to enable automatic OCR: 'winget install UB-Mannheim.TesseractOCR')\033[0m\n")

    doc.close()
    return doc_title, "\n\n".join(pages_text)


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

    m_art = RE_ARTICLE.match(stripped)
    if m_art:
        art_num = m_art.group(1).strip()
        art_title = m_art.group(2).strip()
        title = f"Article {art_num}" + (f": {art_title}" if art_title else "")
        return 1, title

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
        return 3, title

    return None


# =====================================================================
# Clause-Aware Chunking Engine
# =====================================================================

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
    in_table = False
    in_code = False

    def build_breadcrumbs() -> str:
        return " > ".join(hierarchy)

    def finalize_chunk(content_str: str, chunk_type: str = "clause"):
        nonlocal chunks
        clean_content = content_str.strip()
        if not clean_content:
            return

        if len(clean_content) > max_size:
            splitter = "\n\n" if "\n\n" in clean_content else "\n"
            sub_paragraphs = clean_content.split(splitter)
            buffer_paras = []
            curr_len = 0
            part_idx = 1

            for p in sub_paragraphs:
                p_len = len(p)
                if p_len > max_size:
                    for slice_start in range(0, p_len, max_size):
                        slice_p = p[slice_start : slice_start + max_size]
                        sub_title = f"{current_section_title} (Part {part_idx})"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=len(chunks) + 1,
                                document_title=document_title,
                                section_title=sub_title,
                                hierarchy=list(hierarchy),
                                breadcrumbs=build_breadcrumbs(),
                                content=slice_p,
                                char_count=len(slice_p),
                                token_estimate=int(len(slice_p) / 4),
                                chunk_type=chunk_type,
                            )
                        )
                        part_idx += 1
                    continue

                if curr_len + p_len > max_size and buffer_paras:
                    part_content = splitter.join(buffer_paras)
                    sub_title = f"{current_section_title} (Part {part_idx})"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=len(chunks) + 1,
                            document_title=document_title,
                            section_title=sub_title,
                            hierarchy=list(hierarchy),
                            breadcrumbs=build_breadcrumbs(),
                            content=part_content,
                            char_count=len(part_content),
                            token_estimate=int(len(part_content) / 4),
                            chunk_type=chunk_type,
                        )
                    )
                    buffer_paras = [p]
                    curr_len = p_len
                    part_idx += 1
                else:
                    buffer_paras.append(p)
                    curr_len += p_len + len(splitter)

            if buffer_paras:
                part_content = splitter.join(buffer_paras)
                sub_title = f"{current_section_title} (Part {part_idx})" if part_idx > 1 else current_section_title
                chunks.append(
                    DocumentChunk(
                        chunk_id=len(chunks) + 1,
                        document_title=document_title,
                        section_title=sub_title,
                        hierarchy=list(hierarchy),
                        breadcrumbs=build_breadcrumbs(),
                        content=part_content,
                        char_count=len(part_content),
                        token_estimate=int(len(part_content) / 4),
                        chunk_type=chunk_type,
                    )
                )
        else:
            chunks.append(
                DocumentChunk(
                    chunk_id=len(chunks) + 1,
                    document_title=document_title,
                    section_title=current_section_title,
                    hierarchy=list(hierarchy),
                    breadcrumbs=build_breadcrumbs(),
                    content=clean_content,
                    char_count=len(clean_content),
                    token_estimate=int(len(clean_content) / 4),
                    chunk_type=chunk_type,
                )
            )

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

        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            current_block_lines.append(line)
            i += 1
            continue
        elif in_table:
            in_table = False

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
            elif level >= 3:
                if len(hierarchy) >= 3:
                    hierarchy = hierarchy[:3] + [title]
                elif len(hierarchy) == 2:
                    hierarchy = hierarchy[:2] + [title]
                else:
                    hierarchy.append(title)
                current_section_title = title

            current_block_lines.append(line)
            i += 1
            continue

        current_block_lines.append(line)

        current_len = sum(len(l) for l in current_block_lines)
        if current_len >= target_size and not in_table and not in_code:
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
        elif ext in (".docx", ".doc"):
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
    valid_exts = {".pdf", ".docx", ".doc", ".md", ".txt", ".rfc"}
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
        print(f"  • Master Combined     : \033[1;36{master_jsonl_path}\033[0m ({len(master_chunks_raw):,} chunks)")
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
