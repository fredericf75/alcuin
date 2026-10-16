"""
alcuin.storage.engine - In-situ LanceDB Hybrid Storage & Incremental Ingestion Engine
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import lancedb
from lancedb.index import FTS
import pyarrow as pa

from alcuin.chunker.chunker import (
    DocumentChunk,
    chunk_document_text,
    extract_docx_content,
    extract_pdf_content,
    extract_text_file,
)
from alcuin.storage.vault import Vault, VaultConfig

WIKILINK_PATTERN = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")
CHUNKS_TABLE_NAME = "chunks"
EMBEDDING_DIM = 768

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("document_title", pa.string()),
        pa.field("section_title", pa.string()),
        pa.field("breadcrumbs", pa.string()),
        pa.field("content", pa.string()),
        pa.field("wikilinks", pa.list_(pa.string())),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        pa.field("updated_at", pa.string()),
    ]
)


@dataclass
class SearchResult:
    id: str
    file_path: str
    document_title: str
    section_title: str
    breadcrumbs: str
    content: str
    wikilinks: List[str]
    score: float


@dataclass
class IndexReport:
    vault_name: str
    added_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    unchanged_files: List[str] = field(default_factory=list)
    total_chunks_indexed: int = 0
    duration_seconds: float = 0.0

    @property
    def has_changes(self) -> bool:
        return bool(self.added_files or self.modified_files or self.deleted_files)


class EmbeddingClient:
    """Provides vector embeddings using OpenAI-compatible /v1/embeddings, Ollama, or deterministic fallback."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.endpoint = endpoint.rstrip("/") if endpoint else "http://localhost:11434"
        self.model = model or "nomic-embed-text"
        self._available: Optional[bool] = None

    def _is_openai(self) -> bool:
        ep = self.endpoint.lower()
        return ep.endswith("/v1") or ep.endswith("/embeddings") or "/v1/" in ep or ":8000" in ep

    def get_embedding(self, text: str) -> List[float]:
        res = self.get_embeddings_batch([text], batch_size=1)
        if res:
            return res[0]
        return self._fallback_embedding(text)

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        if self._available is False:
            return [self._fallback_embedding(t) for t in texts]

        results: List[List[float]] = []
        is_openai = self._is_openai()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embs = None
            try:
                with httpx.Client(timeout=30.0) as client:
                    if is_openai:
                        base = self.endpoint
                        if base.endswith("/embeddings"):
                            url = base
                        elif base.endswith("/v1"):
                            url = f"{base}/embeddings"
                        else:
                            url = f"{base}/v1/embeddings"
                        resp = client.post(
                            url,
                            json={"model": self.model, "input": batch},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            items = data.get("data", [])
                            if items:
                                items_sorted = sorted(items, key=lambda x: x.get("index", 0))
                                batch_embs = [item["embedding"] for item in items_sorted if "embedding" in item]
                    else:
                        resp = client.post(
                            f"{self.endpoint}/api/embed",
                            json={"model": self.model, "input": batch},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            embs = data.get("embeddings")
                            if embs and len(embs) == len(batch):
                                batch_embs = [[float(x) for x in emb] for emb in embs]
                if batch_embs and len(batch_embs) == len(batch):
                    self._available = True
                else:
                    batch_embs = None
            except Exception:
                self._available = False
                batch_embs = None

            if batch_embs is not None:
                results.extend(batch_embs)
            else:
                self._available = False
                results.extend([self._fallback_embedding(t) for t in batch])
        return results

    @staticmethod
    def _fallback_embedding(text: str) -> List[float]:
        """Generates a deterministic normalized embedding vector for tests/offline mode."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(EMBEDDING_DIM):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 255.0) - 0.5)
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


def extract_wikilinks(text: str) -> List[str]:
    """Extract all [[wikilink]] targets, document citations, and standards references."""
    links: Set[str] = set()
    # 1. Standard [[wikilinks]]
    for match in WIKILINK_PATTERN.finditer(text):
        target = match.group(1).strip()
        if target:
            links.add(target)
    # 2. Statutory / Document section citations (Sec. X-Y, § X-Y, Chapter X, etc.)
    for match in re.finditer(
        r"\b(?:Section|Sec\.|§|Article|Chapter|Clause)\s*([0-9]+(?:[-.][0-9]+)*[a-z]?)\b",
        text,
        re.IGNORECASE,
    ):
        raw = match.group(0).strip()
        if raw and len(raw) <= 40:
            links.add(raw)
    # 3. Standards / Technical identifiers (RFC, PEP, Issue, PR)
    for match in re.finditer(r"\b(?:RFC|PEP|Issue|PR)\s*#?([0-9]+)\b", text, re.IGNORECASE):
        links.add(match.group(0).strip())

    return sorted(list(links))


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def chunk_file_in_memory(file_path: Path) -> Tuple[str, str, List[DocumentChunk]]:
    """Extract and chunk a document file without creating disk artifacts."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        doc_title, raw_text = extract_pdf_content(file_path)
    elif ext in (".docx", ".doc"):
        doc_title, raw_text = extract_docx_content(file_path)
    else:
        doc_title, raw_text = extract_text_file(file_path)

    chunks = chunk_document_text(
        document_title=doc_title,
        raw_text=raw_text,
        target_size=800,
        max_size=1800,
    )
    return doc_title, raw_text, chunks


class VaultStorage:
    """Manages LanceDB tables, incremental indexing, and hybrid retrieval for a Vault."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self.db_dir = vault.lancedb_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_dir))
        self.embedder = EmbeddingClient(
            endpoint=vault.config.embedding_endpoint or vault.config.model_endpoint,
            model=vault.config.embedding_model,
        )
        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        try:
            return self.db.open_table(CHUNKS_TABLE_NAME)
        except Exception:
            return self.db.create_table(CHUNKS_TABLE_NAME, schema=SCHEMA, mode="overwrite")

    def _should_exclude(self, rel_path_str: str) -> bool:
        # Normalize slashes for glob matching
        norm_path = rel_path_str.replace("\\", "/")
        if norm_path.startswith(".alcuin/"):
            return True
        for pattern in self.vault.config.exclude_patterns:
            norm_pattern = pattern.replace("\\", "/")
            if fnmatch.fnmatch(norm_path, norm_pattern) or fnmatch.fnmatch(
                Path(norm_path).name, norm_pattern
            ):
                return True
        return False

    def scan_files(self) -> Dict[str, Path]:
        """Scan vault directory for eligible files returning relative path -> full path."""
        results: Dict[str, Path] = {}
        extensions = set(ext.lower() for ext in self.vault.config.file_extensions)

        for root, dirs, files in os.walk(self.vault.root_path):
            # Prune .alcuin and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != ".alcuin"]
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in extensions:
                    rel_path = str(p.relative_to(self.vault.root_path))
                    if not self._should_exclude(rel_path):
                        results[rel_path] = p

        return results

    def load_manifest(self) -> Dict[str, Any]:
        if self.vault.manifest_path.exists():
            try:
                with open(self.vault.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data.setdefault("files", {})
                        data.setdefault("wikilinks", {})
                        return data
            except Exception:
                pass
        return {"files": {}, "wikilinks": {}}

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        with open(self.vault.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def index(self, force: bool = False) -> IndexReport:
        """Run incremental ingestion of files into LanceDB and manifest."""
        import time

        start_time = time.time()
        report = IndexReport(vault_name=self.vault.name)

        manifest = self.load_manifest()
        known_files = manifest["files"]
        disk_files = self.scan_files()

        # Detect deleted files
        for rel_path in list(known_files.keys()):
            if rel_path not in disk_files:
                report.deleted_files.append(rel_path)
                del known_files[rel_path]
                # Delete rows from LanceDB
                try:
                    # Escape single quotes in path
                    safe_path = rel_path.replace("'", "''")
                    self.table.delete(f"file_path = '{safe_path}'")
                except Exception:
                    pass

        files_to_index: List[Tuple[str, Path, str]] = []

        # Detect added and modified files
        for rel_path, full_path in disk_files.items():
            current_sha = compute_file_sha256(full_path)
            prev_info = known_files.get(rel_path)

            if force or not prev_info:
                report.added_files.append(rel_path)
                files_to_index.append((rel_path, full_path, current_sha))
                if prev_info:
                    try:
                        safe_path = rel_path.replace("'", "''")
                        self.table.delete(f"file_path = '{safe_path}'")
                    except Exception:
                        pass
            elif prev_info.get("sha256") != current_sha:
                report.modified_files.append(rel_path)
                files_to_index.append((rel_path, full_path, current_sha))
                # Delete old rows before re-inserting
                try:
                    safe_path = rel_path.replace("'", "''")
                    self.table.delete(f"file_path = '{safe_path}'")
                except Exception:
                    pass
            else:
                report.unchanged_files.append(rel_path)

        all_new_records: List[Dict[str, Any]] = []
        wikilinks_map: Dict[str, List[str]] = manifest.get("wikilinks", {})

        now_str = datetime.now(timezone.utc).isoformat()

        for rel_path, full_path, sha in files_to_index:
            try:
                doc_title, raw_text, chunks = chunk_file_in_memory(full_path)
                file_links = extract_wikilinks(raw_text)
                wikilinks_map[rel_path] = file_links

                # Update manifest entry
                known_files[rel_path] = {
                    "sha256": sha,
                    "mtime": full_path.stat().st_mtime,
                    "chunk_count": len(chunks),
                    "document_title": doc_title,
                    "wikilinks": file_links,
                    "updated_at": now_str,
                }

                # Batch embed and prepare records
                chunk_texts = [f"{c.breadcrumbs}\n{c.content}" for c in chunks]
                embeddings = self.embedder.get_embeddings_batch(chunk_texts)

                for chunk, vec in zip(chunks, embeddings):
                    chunk_links = extract_wikilinks(chunk.content)
                    enriched_content = f"> **Context**: {chunk.breadcrumbs}\n\n{chunk.content}"
                    all_new_records.append(
                        {
                            "id": f"{rel_path}#{chunk.chunk_id}",
                            "file_path": rel_path,
                            "document_title": chunk.document_title,
                            "section_title": chunk.section_title,
                            "breadcrumbs": chunk.breadcrumbs,
                            "content": enriched_content,
                            "wikilinks": chunk_links,
                            "vector": vec,
                            "updated_at": now_str,
                        }
                    )
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")

        if all_new_records:
            self.table.add(all_new_records)
            report.total_chunks_indexed = len(all_new_records)
            # Rebuild FTS index for fast hybrid keyword matching
            try:
                self.table.create_index("content", config=FTS(), replace=True)
            except Exception:
                pass

        manifest["files"] = known_files
        manifest["wikilinks"] = wikilinks_map
        self.save_manifest(manifest)

        report.duration_seconds = round(time.time() - start_time, 2)
        return report

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform hybrid search over LanceDB chunks."""
        if self.table.count_rows() == 0:
            return []

        query_vec = self.embedder.get_embedding(query)
        try:
            # Try native hybrid search
            res = (
                self.table.search(query_type="hybrid")
                .vector(query_vec)
                .text(query)
                .limit(top_k)
                .to_list()
            )
        except Exception:
            # Fallback to vector search
            res = self.table.search(query_vec).limit(top_k).to_list()

        results: List[SearchResult] = []
        for r in res:
            score = float(r.get("_score", r.get("_distance", 1.0)))
            results.append(
                SearchResult(
                    id=r["id"],
                    file_path=r["file_path"],
                    document_title=r["document_title"],
                    section_title=r["section_title"],
                    breadcrumbs=r["breadcrumbs"],
                    content=r["content"],
                    wikilinks=list(r.get("wikilinks", [])),
                    score=score,
                )
            )
        return results

    def get_linked_pages(self, page_target: str) -> List[str]:
        """Find files that mention [[page_target]] or pages linked from it."""
        manifest = self.load_manifest()
        wikilinks = manifest.get("wikilinks", {})
        found: Set[str] = set()

        norm_target = page_target.strip().lower()

        # Check who links to target
        for src_path, links in wikilinks.items():
            for link in links:
                if link.lower() == norm_target:
                    found.add(src_path)

        # Check outgoing links if page_target is a known file
        for src_path, links in wikilinks.items():
            if Path(src_path).stem.lower() == norm_target or src_path.lower() == norm_target:
                found.update(links)

        return sorted(list(found))

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[SearchResult]:
        """Retrieve multiple chunks directly by chunk ID."""
        if not chunk_ids:
            return []
        try:
            # Query LanceDB by ID filter
            clauses = []
            for cid in chunk_ids[:200]:
                clean_id = cid.replace("'", "''")
                clauses.append(f"id = '{clean_id}'")
            where_sql = " or ".join(clauses)
            arrow_tbl = self.table.search().where(where_sql).to_arrow()
            results = []
            for i in range(len(arrow_tbl)):
                results.append(
                    SearchResult(
                        id=arrow_tbl["id"][i].as_py(),
                        file_path=arrow_tbl["file_path"][i].as_py(),
                        document_title=arrow_tbl["document_title"][i].as_py(),
                        section_title=arrow_tbl["section_title"][i].as_py(),
                        breadcrumbs=arrow_tbl["breadcrumbs"][i].as_py(),
                        content=arrow_tbl["content"][i].as_py(),
                        wikilinks=list(arrow_tbl["wikilinks"][i].as_py() or []),
                        score=1.0,
                    )
                )
            return results
        except Exception:
            return []
