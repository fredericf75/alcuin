"""
alcuin.storage.ingest - Multimodal Inbox Intake and Conversion Engine
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from alcuin.chunker.chunker import (
    extract_docx_content,
    extract_pdf_content,
    extract_text_file,
    find_tesseract_binary,
)
from alcuin.storage.engine import VaultStorage, compute_file_sha256
from alcuin.storage.vault import Vault

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt"}
SUPPORTED_INBOX_EXTENSIONS = IMAGE_EXTENSIONS | DOC_EXTENSIONS


@dataclass
class IngestRecord:
    sha256: str
    original_name: str
    vault_note_path: str
    archive_path: str
    ingested_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IngestRecord:
        return cls(**data)


@dataclass
class IngestResult:
    ingested_notes: List[str] = field(default_factory=list)
    skipped_duplicates: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingested_notes": self.ingested_notes,
            "skipped_duplicates": self.skipped_duplicates,
            "errors": self.errors,
            "ingested_count": len(self.ingested_notes),
            "skipped_count": len(self.skipped_duplicates),
            "error_count": len(self.errors),
        }


class InboxIngestionEngine:
    """Manages intake conversion from inbox/ to vault/ notes and archive/."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self.inbox_dir = vault.root_path / vault.config.inbox_dir
        self.archive_dir = vault.root_path / vault.config.archive_dir
        self.history_file = vault.alcuin_dir / "ingest_history.json"

    def validate_folders(self) -> Tuple[bool, str]:
        """Check if inbox and archive folders exist and are properly configured."""
        if not self.inbox_dir.exists() or not self.archive_dir.exists():
            return (
                False,
                f"Vault intake folders ('{self.vault.config.inbox_dir}' or '{self.vault.config.archive_dir}') are missing. Run 'alcuin lint --fix' to upgrade this vault.",
            )
        return True, ""

    def load_history(self) -> Dict[str, IngestRecord]:
        if not self.history_file.exists():
            return {}
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: IngestRecord.from_dict(v) for k, v in data.get("records", {}).items()}
        except Exception:
            return {}

    def save_history(self, records: Dict[str, IngestRecord]) -> None:
        self.vault.alcuin_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": {k: v.to_dict() for k, v in records.items()},
        }
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def find_local_vision_model(self) -> Optional[str]:
        """Discover available vision-capable models in local Ollama instance."""
        endpoint = self.vault.config.model_endpoint or "http://localhost:11434"
        if not ("localhost" in endpoint or "127.0.0.1" in endpoint):
            return None
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{endpoint.rstrip('/')}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    for m in models:
                        caps = m.get("details", {}).get("capabilities", []) or m.get("capabilities", [])
                        if "vision" in caps:
                            return m.get("name") or m.get("model")
        except Exception:
            pass
        return None

    def extract_image_with_tesseract(self, image_path: Path) -> Optional[str]:
        """Attempt fast local OCR extraction via Tesseract binary."""
        tess = find_tesseract_binary()
        if not tess:
            return None
        try:
            res = subprocess.run(
                [tess, str(image_path), "stdout", "-l", "eng"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def extract_image_with_vlm(self, image_path: Path, model_name: str) -> Optional[str]:
        """Extract text and layout details using a local Ollama vision model."""
        endpoint = self.vault.config.model_endpoint or "http://localhost:11434"
        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                "Transcribe all legible text from this image verbatim. "
                "Include dates, numbers, totals, names, addresses, and tables. "
                "Provide a brief description of the document format (e.g., receipt, letter, handwritten note)."
            )

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_img],
                    }
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            }

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{endpoint.rstrip('/')}/api/chat", json=payload)
                if resp.status_code == 200:
                    content = resp.json().get("message", {}).get("content", "")
                    if content and content.strip():
                        return content.strip()
        except Exception:
            pass
        return None

    def extract_content(self, file_path: Path) -> Tuple[str, str]:
        """
        Extract title and markdown text from an incoming document or image.
        Returns: (doc_title, markdown_content)
        """
        ext = file_path.suffix.lower()
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

        # 1. Digital PDF
        if ext == ".pdf":
            try:
                pdf_title, pdf_text = extract_pdf_content(file_path)
                return pdf_title or title, pdf_text
            except Exception as e:
                return title, f"PDF extraction error: {e}"

        # 2. Word documents
        elif ext in (".docx", ".doc"):
            try:
                docx_title, docx_text = extract_docx_content(file_path)
                return docx_title or title, docx_text
            except Exception as e:
                return title, f"Word extraction error: {e}"

        # 3. Plain text / Markdown
        elif ext in (".md", ".txt"):
            _, text = extract_text_file(file_path)
            return title, text

        # 4. Images (PNG, JPG, WebP, etc.)
        elif ext in IMAGE_EXTENSIONS:
            # Priority A: Tesseract OCR (fastest local path)
            ocr_text = self.extract_image_with_tesseract(file_path)
            if ocr_text and len(ocr_text) > 30:
                return title, f"## Transcribed Text (OCR)\n\n{ocr_text}"

            # Priority B: Local Ollama Vision Model
            vlm_model = self.find_local_vision_model()
            if vlm_model:
                vlm_text = self.extract_image_with_vlm(file_path, vlm_model)
                if vlm_text:
                    return title, f"## Visual Document Transcription ({vlm_model})\n\n{vlm_text}"

            # If Tesseract returned short text, keep it
            if ocr_text:
                return title, f"## Transcribed Text (OCR)\n\n{ocr_text}"

            # Priority C: Offline Metadata Stub
            return title, (
                f"*Image file ingested: `{file_path.name}`.*\n\n"
                "*No local OCR (Tesseract) or Vision LLM was reachable during ingestion. "
                "You may edit this note by hand to add transcriptions.*"
            )

        return title, f"Unsupported file format: {ext}"

    def ingest_file(
        self,
        file_path: Path,
        dry_run: bool = False,
        force: bool = False,
    ) -> Tuple[Optional[Path], str]:
        """
        Ingest a single raw file into the vault:
        - Deduplicates via SHA-256
        - Converts to .md note
        - Moves raw file to archive/
        Returns: (generated_note_path or None, status_message)
        """
        if not file_path.is_file():
            return None, f"File does not exist: {file_path}"

        sha = compute_file_sha256(file_path)
        history = self.load_history()

        if not force and sha in history:
            prev = history[sha]
            return (
                None,
                f"Skipped: Already ingested as '{prev.vault_note_path}' on {prev.ingested_at[:10]} (SHA-256 match)",
            )

        title, extracted_body = self.extract_content(file_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        date_short = now_iso[:10]

        # Determine target note path in vault root
        safe_stem = re.sub(r'[\\/*?:"<>|]', "_", file_path.stem)
        note_name = f"{safe_stem}.md"
        target_note = self.vault.root_path / note_name

        # Prevent overwriting an existing different note
        counter = 1
        while target_note.exists() and not force:
            note_name = f"{safe_stem}_{counter}.md"
            target_note = self.vault.root_path / note_name
            counter += 1

        # Determine target archive path
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        archive_name = file_path.name
        target_archive = self.archive_dir / archive_name
        a_counter = 1
        while target_archive.exists():
            target_archive = self.archive_dir / f"{file_path.stem}_{a_counter}{file_path.suffix}"
            a_counter += 1

        rel_archive = str(target_archive.relative_to(self.vault.root_path)).replace("\\", "/")

        note_content = (
            f"# {title}\n\n"
            f"> **Source**: `{rel_archive}` | **Ingested**: {date_short}\n\n"
            f"{extracted_body.strip()}\n"
        )

        if dry_run:
            return (
                target_note,
                f"[Dry Run] Would convert '{file_path.name}' -> '{note_name}' and archive to '{rel_archive}'",
            )

        # 1. Write Markdown note to vault
        target_note.write_text(note_content, encoding="utf-8")

        # 2. Move original raw file to archive
        shutil.move(str(file_path), str(target_archive))

        # 3. Update history
        rel_note = str(target_note.relative_to(self.vault.root_path)).replace("\\", "/")
        history[sha] = IngestRecord(
            sha256=sha,
            original_name=file_path.name,
            vault_note_path=rel_note,
            archive_path=rel_archive,
            ingested_at=now_iso,
        )
        self.save_history(history)

        return target_note, f"Ingested '{file_path.name}' -> '{rel_note}' (Archived to '{rel_archive}')"

    def ingest_inbox(
        self,
        dry_run: bool = False,
        force: bool = False,
    ) -> IngestResult:
        """Process all eligible files in the vault inbox directory."""
        result = IngestResult()

        valid, msg = self.validate_folders()
        if not valid:
            result.errors.append(msg)
            return result

        inbox_files = [
            p
            for p in self.inbox_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_INBOX_EXTENSIONS
        ]

        if not inbox_files:
            return result

        for f in sorted(inbox_files, key=lambda x: x.name.lower()):
            note_path, status = self.ingest_file(f, dry_run=dry_run, force=force)
            if note_path:
                result.ingested_notes.append(status)
            elif "Skipped:" in status:
                result.skipped_duplicates.append(status)
            else:
                result.errors.append(status)

        # Trigger incremental indexing if new notes were created
        if result.ingested_notes and not dry_run:
            try:
                storage = VaultStorage(self.vault)
                storage.index(force=False)
            except Exception as e:
                result.errors.append(f"Auto-index failed: {e}")

        return result
