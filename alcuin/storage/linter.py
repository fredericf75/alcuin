"""
alcuin.storage.linter - Vault Integrity Diagnostic and Remediation Engine
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from alcuin.storage.engine import VaultStorage, compute_file_sha256
from alcuin.storage.vault import Vault

WIKILINK_EXPLICIT_PATTERN = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LintCategory(str, Enum):
    DEAD_LINK = "dead-link"
    SYNC_DRIFT = "sync-drift"
    STORAGE_DRIFT = "storage-drift"
    DOCUMENT_HYGIENE = "document-hygiene"
    KNOWLEDGE_GAP = "knowledge-gap"


@dataclass
class LintIssue:
    category: LintCategory
    severity: LintSeverity
    file_path: str
    message: str
    line_number: Optional[int] = None
    fixable: bool = False
    fix_description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "message": self.message,
            "fixable": self.fixable,
            "fix_description": self.fix_description,
            "details": self.details,
        }


@dataclass
class LintReport:
    vault_name: str
    vault_path: str
    issues: List[LintIssue] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == LintSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == LintSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == LintSeverity.INFO)

    @property
    def fixable_count(self) -> int:
        return sum(1 for i in self.issues if i.fixable)

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0 and self.warning_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vault_name": self.vault_name,
            "vault_path": self.vault_path,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "fixable_count": self.fixable_count,
            "is_clean": self.is_clean,
            "duration_seconds": self.duration_seconds,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class FixResult:
    repaired: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repaired": self.repaired,
            "skipped": self.skipped,
            "errors": self.errors,
            "repaired_count": len(self.repaired),
            "error_count": len(self.errors),
        }


class VaultLinter:
    """Audits vault consistency, link validity, and vector storage parity with remediation."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self.storage = VaultStorage(vault)

    def _collect_known_targets(self, disk_files: Dict[str, Path], manifest: Dict[str, Any]) -> Dict[str, Set[str]]:
        """
        Build an index of valid targets for wikilink resolution:
        - Exact relative paths (with and without extension)
        - File stems (case-insensitive)
        - Document titles from manifest and headers
        """
        exact_paths: Set[str] = set()
        stems_lower: Set[str] = set()
        titles_lower: Set[str] = set()

        # All disk files (including media/images)
        for root, dirs, files in os.walk(self.vault.root_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != ".alcuin"]
            for f in files:
                p = Path(root) / f
                try:
                    rel = str(p.relative_to(self.vault.root_path)).replace("\\", "/")
                    exact_paths.add(rel)
                    exact_paths.add(rel.lower())
                    stems_lower.add(p.stem.lower())
                    stems_lower.add(p.name.lower())
                except Exception:
                    pass

        # Titles from manifest
        files_manifest = manifest.get("files", {})
        for rel_path, meta in files_manifest.items():
            doc_title = meta.get("document_title")
            if doc_title and isinstance(doc_title, str) and doc_title.strip():
                titles_lower.add(doc_title.strip().lower())

        return {
            "exact_paths": exact_paths,
            "stems_lower": stems_lower,
            "titles_lower": titles_lower,
        }

    def _resolve_wikilink_target(
        self,
        raw_target: str,
        source_rel_path: str,
        known_targets: Dict[str, Set[str]],
    ) -> bool:
        """Check if a wikilink target resolves to any existing document or anchor."""
        clean = raw_target.strip()
        # Internal section link in same document (e.g. [[#Summary]])
        if clean.startswith("#"):
            return True

        # Strip heading anchor (e.g. Target#Heading -> Target)
        if "#" in clean:
            clean = clean.split("#", 1)[0].strip()

        if not clean:
            return True

        # Strip .md if explicitly included
        if clean.lower().endswith(".md"):
            clean_stem = clean[:-3]
        else:
            clean_stem = clean

        clean_lower = clean.lower()
        clean_stem_lower = clean_stem.lower()

        # 1. Exact path or stem matches
        if clean in known_targets["exact_paths"] or clean_lower in known_targets["exact_paths"]:
            return True
        if clean_stem_lower in known_targets["stems_lower"]:
            return True
        if clean_lower in known_targets["titles_lower"]:
            return True

        # 2. Relative to source file directory
        source_dir = Path(source_rel_path).parent
        rel_candidate = (source_dir / clean).as_posix()
        if rel_candidate in known_targets["exact_paths"] or rel_candidate.lower() in known_targets["exact_paths"]:
            return True

        rel_md_candidate = (source_dir / f"{clean}.md").as_posix()
        if rel_md_candidate in known_targets["exact_paths"] or rel_md_candidate.lower() in known_targets["exact_paths"]:
            return True

        # 3. Direct filesystem check fallback
        if (self.vault.root_path / clean).exists() or (self.vault.root_path / f"{clean}.md").exists():
            return True

        return False

    def lint(self) -> LintReport:
        """Run all diagnostic audits on the vault."""
        start_time = time.time()
        issues: List[LintIssue] = []

        manifest = self.storage.load_manifest()
        known_manifest_files: Dict[str, Any] = manifest.get("files", {})
        disk_files = self.storage.scan_files()
        known_targets = self._collect_known_targets(disk_files, manifest)

        # -------------------------------------------------------------
        # 1. Sync Drift Checks
        # -------------------------------------------------------------
        # Unindexed files on disk
        for rel_path, full_path in disk_files.items():
            if rel_path not in known_manifest_files:
                issues.append(
                    LintIssue(
                        category=LintCategory.SYNC_DRIFT,
                        severity=LintSeverity.WARNING,
                        file_path=rel_path,
                        message=f"File exists on disk but is not indexed in manifest.",
                        fixable=True,
                        fix_description=f"Index '{rel_path}' into LanceDB",
                        details={"type": "unindexed", "full_path": str(full_path)},
                    )
                )
            else:
                # Check for pending modifications
                curr_sha = compute_file_sha256(full_path)
                prev_sha = known_manifest_files[rel_path].get("sha256")
                if curr_sha != prev_sha:
                    issues.append(
                        LintIssue(
                            category=LintCategory.SYNC_DRIFT,
                            severity=LintSeverity.WARNING,
                            file_path=rel_path,
                            message=f"File has been modified on disk since last indexing.",
                            fixable=True,
                            fix_description=f"Re-index modified '{rel_path}' into LanceDB",
                            details={"type": "modified", "full_path": str(full_path)},
                        )
                    )

        # Excluded files currently lingering in manifest
        for rel_path in list(known_manifest_files.keys()):
            if self.storage._should_exclude(rel_path):
                issues.append(
                    LintIssue(
                        category=LintCategory.SYNC_DRIFT,
                        severity=LintSeverity.INFO,
                        file_path=rel_path,
                        message=f"File in manifest matches active exclusion pattern.",
                        fixable=True,
                        fix_description=f"Remove excluded file '{rel_path}' from manifest and LanceDB",
                        details={"type": "excluded_tracked"},
                    )
                )

        # Check intake folders (inbox / archive)
        inbox_path = self.vault.root_path / self.vault.config.inbox_dir
        archive_path = self.vault.root_path / self.vault.config.archive_dir
        missing_folders = []
        if not inbox_path.exists():
            missing_folders.append(self.vault.config.inbox_dir)
        if not archive_path.exists():
            missing_folders.append(self.vault.config.archive_dir)

        if missing_folders:
            issues.append(
                LintIssue(
                    category=LintCategory.SYNC_DRIFT,
                    severity=LintSeverity.WARNING,
                    file_path=self.vault.config.inbox_dir,
                    message=f"Vault intake folder(s) missing: {', '.join(missing_folders)}. Run 'alcuin lint --fix' to scaffold.",
                    fixable=True,
                    fix_description="Scaffold missing inbox/archive folders and update config.toml",
                    details={"type": "missing_intake_folders", "folders": missing_folders},
                )
            )

        # -------------------------------------------------------------
        # 2. Document Hygiene & Frontmatter Checks
        # -------------------------------------------------------------
        doc_titles: Dict[str, List[str]] = {}

        for rel_path, full_path in disk_files.items():
            try:
                st = full_path.stat()
                if st.st_size == 0:
                    issues.append(
                        LintIssue(
                            category=LintCategory.DOCUMENT_HYGIENE,
                            severity=LintSeverity.WARNING,
                            file_path=rel_path,
                            message="File is empty (0 bytes).",
                            fixable=True,
                            fix_description=f"Inject default document header into '{rel_path}'",
                            details={"type": "empty_file"},
                        )
                    )
                    continue

                if full_path.suffix.lower() in (".md", ".txt"):
                    try:
                        raw_content = full_path.read_text(encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        raw_content = full_path.read_text(encoding="latin-1", errors="replace")

                    if not raw_content.strip():
                        issues.append(
                            LintIssue(
                                category=LintCategory.DOCUMENT_HYGIENE,
                                severity=LintSeverity.WARNING,
                                file_path=rel_path,
                                message="File contains only whitespace.",
                                fixable=True,
                                fix_description=f"Inject default document header into '{rel_path}'",
                                details={"type": "whitespace_only"},
                            )
                        )
                        continue

                    lines = raw_content.splitlines()

                    # Check Frontmatter delimiter closure
                    if lines and lines[0].strip() == "---":
                        closed = False
                        for idx, l in enumerate(lines[1:], start=2):
                            if l.strip() == "---":
                                closed = True
                                break
                        if not closed:
                            issues.append(
                                LintIssue(
                                    category=LintCategory.DOCUMENT_HYGIENE,
                                    severity=LintSeverity.ERROR,
                                    file_path=rel_path,
                                    line_number=1,
                                    message="Unclosed YAML frontmatter delimiter ('---').",
                                    fixable=True,
                                    fix_description=f"Close frontmatter delimiter in '{rel_path}'",
                                    details={"type": "unclosed_frontmatter"},
                                )
                            )

                    # Check for top-level heading
                    has_heading = any(l.strip().startswith("# ") for l in lines)
                    if not has_heading and full_path.suffix.lower() == ".md":
                        issues.append(
                            LintIssue(
                                category=LintCategory.DOCUMENT_HYGIENE,
                                severity=LintSeverity.INFO,
                                file_path=rel_path,
                                message="Markdown file lacks top-level '# Title' heading.",
                                fixable=True,
                                fix_description=f"Prepend '# {full_path.stem.title()}' header to '{rel_path}'",
                                details={"type": "missing_heading"},
                            )
                        )

                    # Collect titles for duplicate check
                    title = known_manifest_files.get(rel_path, {}).get("document_title")
                    if not title:
                        for l in lines:
                            if l.strip().startswith("# "):
                                title = l.strip()[2:].strip()
                                break
                    if title and title.strip():
                        doc_titles.setdefault(title.strip(), []).append(rel_path)

            except Exception as e:
                issues.append(
                    LintIssue(
                        category=LintCategory.DOCUMENT_HYGIENE,
                        severity=LintSeverity.ERROR,
                        file_path=rel_path,
                        message=f"Failed to read file for linting: {e}",
                        fixable=False,
                    )
                )

        # Duplicate title check
        for title, paths in doc_titles.items():
            if len(paths) > 1:
                for p in paths:
                    issues.append(
                        LintIssue(
                            category=LintCategory.DOCUMENT_HYGIENE,
                            severity=LintSeverity.WARNING,
                            file_path=p,
                            message=f"Duplicate document title '{title}' shared with: {', '.join([x for x in paths if x != p])}",
                            fixable=False,
                            details={"type": "duplicate_title", "title": title, "duplicates": paths},
                        )
                    )

        # -------------------------------------------------------------
        # 3. Dead Wikilinks Checks
        # -------------------------------------------------------------
        for rel_path, full_path in disk_files.items():
            if full_path.suffix.lower() not in (".md", ".txt"):
                continue
            try:
                try:
                    content = full_path.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError:
                    content = full_path.read_text(encoding="latin-1", errors="replace")

                for line_idx, line in enumerate(content.splitlines(), start=1):
                    for match in WIKILINK_EXPLICIT_PATTERN.finditer(line):
                        raw_target = match.group(1).strip()
                        if not raw_target:
                            continue
                        if not self._resolve_wikilink_target(raw_target, rel_path, known_targets):
                            # Target is broken/dead
                            issues.append(
                                LintIssue(
                                    category=LintCategory.DEAD_LINK,
                                    severity=LintSeverity.WARNING,
                                    file_path=rel_path,
                                    line_number=line_idx,
                                    message=f"Dead wikilink '[[{raw_target}]]' points to non-existent target.",
                                    fixable=True,
                                    fix_description=f"Create stub note '{raw_target}.md'",
                                    details={"type": "dead_wikilink", "target": raw_target, "line": line_idx},
                                )
                            )
            except Exception:
                pass

        # -------------------------------------------------------------
        # 4. Storage & LanceDB Parity Checks
        # -------------------------------------------------------------
        try:
            table = self.storage._get_or_create_table()
            self.storage.table = table
            row_count = table.count_rows()

            if row_count > 0:
                arrow_tbl = table.to_arrow()
                db_paths = arrow_tbl["file_path"].to_pylist()

                db_counts: Dict[str, int] = {}
                for p in db_paths:
                    db_counts[p] = db_counts.get(p, 0) + 1

                # Check for orphan paths in LanceDB
                for db_file, count in db_counts.items():
                    if db_file not in disk_files and db_file not in known_manifest_files:
                        issues.append(
                            LintIssue(
                                category=LintCategory.STORAGE_DRIFT,
                                severity=LintSeverity.ERROR,
                                file_path=db_file,
                                message=f"Orphan data in LanceDB ({count} chunks) for deleted file.",
                                fixable=True,
                                fix_description=f"Purge {count} orphan chunk(s) for '{db_file}' from LanceDB",
                                details={"type": "orphan_chunks", "count": count},
                            )
                        )

                # Check chunk count mismatch vs manifest
                for rel_path, meta in known_manifest_files.items():
                    if rel_path in disk_files:
                        expected_chunks = meta.get("chunk_count", 0)
                        actual_chunks = db_counts.get(rel_path, 0)
                        if expected_chunks != actual_chunks:
                            issues.append(
                                LintIssue(
                                    category=LintCategory.STORAGE_DRIFT,
                                    severity=LintSeverity.WARNING,
                                    file_path=rel_path,
                                    message=f"Chunk count mismatch: manifest expects {expected_chunks}, LanceDB has {actual_chunks}.",
                                    fixable=True,
                                    fix_description=f"Re-index '{rel_path}' to synchronize chunk counts",
                                    details={
                                        "type": "chunk_mismatch",
                                        "expected": expected_chunks,
                                        "actual": actual_chunks,
                                    },
                                )
                            )
            elif known_manifest_files:
                issues.append(
                    LintIssue(
                        category=LintCategory.STORAGE_DRIFT,
                        severity=LintSeverity.ERROR,
                        file_path=".alcuin/lancedb",
                        message=f"LanceDB chunks table is empty while manifest contains {len(known_manifest_files)} indexed files.",
                        fixable=True,
                        fix_description="Re-index all vault documents",
                        details={"type": "empty_database"},
                    )
                )

        except Exception as e:
            issues.append(
                LintIssue(
                    category=LintCategory.STORAGE_DRIFT,
                    severity=LintSeverity.ERROR,
                    file_path=".alcuin/lancedb",
                    message=f"Failed to inspect LanceDB table: {e}",
                    fixable=False,
                )
            )
        # -------------------------------------------------------------
        # 5. Knowledge Gap Checks
        # -------------------------------------------------------------
        try:
            from alcuin.storage.gaps import KnowledgeGapTracker

            gap_tracker = KnowledgeGapTracker(self.vault)
            unresolved_gaps = gap_tracker.list_gaps(include_resolved=False)
            for gap in unresolved_gaps:
                sev = LintSeverity.WARNING if gap.ask_count >= 3 else LintSeverity.INFO
                issues.append(
                    LintIssue(
                        category=LintCategory.KNOWLEDGE_GAP,
                        severity=sev,
                        file_path=".alcuin/knowledge_gaps.json",
                        message=f"Unresolved question gap: '{gap.query}' (asked {gap.ask_count} time(s)).",
                        fixable=not gap.stub_created,
                        fix_description=f"Create stub note '{gap.suggested_title}.md' for missing knowledge topic",
                        details={
                            "type": "knowledge_gap",
                            "gap_id": gap.id,
                            "query": gap.query,
                            "title": gap.suggested_title,
                            "ask_count": gap.ask_count,
                            "stub_created": gap.stub_created,
                        },
                    )
                )
        except Exception:
            pass

        duration = round(time.time() - start_time, 2)
        return LintReport(
            vault_name=self.vault.name,
            vault_path=str(self.vault.root_path),
            issues=issues,
            duration_seconds=duration,
        )

    def fix(self, create_stubs: bool = False, dry_run: bool = False) -> FixResult:
        """Remediate fixable issues identified during audit."""
        report = self.lint()
        result = FixResult()

        files_to_reindex: Set[str] = set()
        stubs_to_create: Dict[str, str] = {}  # target -> source_file
        gap_stubs_to_create: List[Dict[str, Any]] = []  # knowledge gap details
        orphans_to_purge: Set[str] = set()
        excluded_to_remove: Set[str] = set()
        headers_to_inject: List[Tuple[Path, str]] = []  # (path, title)
        frontmatter_to_close: List[Path] = []

        scaffold_intake = False

        for issue in report.issues:
            if not issue.fixable:
                result.skipped.append(f"{issue.file_path}: {issue.message} (Not auto-fixable)")
                continue

            itype = issue.details.get("type", "")

            if itype == "missing_intake_folders":
                scaffold_intake = True

            elif itype in ("unindexed", "modified", "chunk_mismatch"):
                files_to_reindex.add(issue.file_path)

            elif itype == "empty_database":
                files_to_reindex.update(self.storage.scan_files().keys())

            elif itype == "orphan_chunks":
                orphans_to_purge.add(issue.file_path)

            elif itype == "excluded_tracked":
                excluded_to_remove.add(issue.file_path)

            elif itype == "dead_wikilink":
                if create_stubs:
                    target = issue.details.get("target")
                    if target:
                        # Clean target of anchors
                        clean_target = target.split("#")[0].strip()
                        if clean_target and clean_target not in stubs_to_create:
                            stubs_to_create[clean_target] = issue.file_path
                else:
                    result.skipped.append(
                        f"Dead wikilink '[[{issue.details.get('target')}]]' in {issue.file_path} (Pass --create-stubs to generate placeholder notes)"
                    )

            elif itype == "knowledge_gap":
                if create_stubs:
                    gap_stubs_to_create.append(issue.details)
                else:
                    result.skipped.append(
                        f"Unresolved question gap '{issue.details.get('query')}' (Pass --create-stubs to generate placeholder notes)"
                    )

            elif itype in ("empty_file", "whitespace_only", "missing_heading"):
                full_p = self.vault.root_path / issue.file_path
                title = full_p.stem.replace("_", " ").replace("-", " ").title()
                headers_to_inject.append((full_p, title))

            elif itype == "unclosed_frontmatter":
                full_p = self.vault.root_path / issue.file_path
                frontmatter_to_close.append(full_p)

        # 1. Preview / Dry-run actions
        if dry_run:
            if scaffold_intake:
                result.repaired.append(
                    f"[Dry Run] Scaffold intake folders '{self.vault.config.inbox_dir}' and '{self.vault.config.archive_dir}' and update config.toml"
                )
            for p, title in headers_to_inject:
                result.repaired.append(f"[Dry Run] Inject '# {title}' header into '{p.name}'")
            for p in frontmatter_to_close:
                result.repaired.append(f"[Dry Run] Close YAML frontmatter in '{p.name}'")
            for target, src in stubs_to_create.items():
                result.repaired.append(f"[Dry Run] Create stub note '{target}.md' from link in '{src}'")
            for g_info in gap_stubs_to_create:
                t = g_info.get("title", "Gap")
                q = g_info.get("query", "")
                result.repaired.append(f"[Dry Run] Create stub note '{t}.md' for knowledge gap '{q}'")
            for f in files_to_reindex:
                result.repaired.append(f"[Dry Run] Re-index '{f}' into LanceDB")
            for f in orphans_to_purge:
                result.repaired.append(f"[Dry Run] Purge orphan records for '{f}' from LanceDB")
            for f in excluded_to_remove:
                result.repaired.append(f"[Dry Run] Remove excluded file '{f}' from manifest and LanceDB")
            return result

        # 1b. Execute Intake Folder Scaffolding
        if scaffold_intake:
            try:
                (self.vault.root_path / self.vault.config.inbox_dir).mkdir(parents=True, exist_ok=True)
                (self.vault.root_path / self.vault.config.archive_dir).mkdir(parents=True, exist_ok=True)
                with open(self.vault.config_path, "w", encoding="utf-8") as f:
                    f.write(self.vault.config.to_toml())
                result.repaired.append(
                    f"Scaffolded intake folders '{self.vault.config.inbox_dir}' and '{self.vault.config.archive_dir}' and updated config.toml"
                )
            except Exception as e:
                result.errors.append(f"Failed to scaffold intake folders: {e}")

        # 2. Execute Document Hygiene Fixes
        for p, title in headers_to_inject:
            try:
                rel = str(p.relative_to(self.vault.root_path))
                if not p.exists() or p.stat().st_size == 0:
                    p.write_text(f"# {title}\n\n", encoding="utf-8")
                else:
                    curr = p.read_text(encoding="utf-8", errors="replace")
                    if not curr.strip():
                        p.write_text(f"# {title}\n\n", encoding="utf-8")
                    else:
                        p.write_text(f"# {title}\n\n{curr}", encoding="utf-8")
                result.repaired.append(f"Injected header '# {title}' into '{rel}'")
                files_to_reindex.add(rel)
            except Exception as e:
                result.errors.append(f"Failed to inject header into {p}: {e}")

        for p in frontmatter_to_close:
            try:
                rel = str(p.relative_to(self.vault.root_path))
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                # Find best place to close frontmatter or append
                lines.insert(len(lines), "---")
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result.repaired.append(f"Closed YAML frontmatter delimiter in '{rel}'")
                files_to_reindex.add(rel)
            except Exception as e:
                result.errors.append(f"Failed to close frontmatter in {p}: {e}")

        # 3. Execute Stub Creation
        newly_created_stubs = []
        for target, src in stubs_to_create.items():
            try:
                # Sanitize target for filename
                safe_stem = re.sub(r'[\\/*?:"<>|]', "_", target)
                stub_file = self.vault.root_path / f"{safe_stem}.md"
                if not stub_file.exists():
                    stub_content = f"# {target}\n\n*Stub created by Alcuin lint from backlink in [[{src}]].*\n"
                    stub_file.write_text(stub_content, encoding="utf-8")
                    rel_stub = str(stub_file.relative_to(self.vault.root_path))
                    result.repaired.append(f"Created stub note '{rel_stub}' for backlink in '{src}'")
                    newly_created_stubs.append(rel_stub)
                    files_to_reindex.add(rel_stub)
            except Exception as e:
                result.errors.append(f"Failed to create stub note for '{target}': {e}")

        # 3b. Execute Knowledge Gap Stub Creation
        if gap_stubs_to_create:
            try:
                from alcuin.storage.gaps import KnowledgeGapTracker

                gap_tracker = KnowledgeGapTracker(self.vault)
                for g_info in gap_stubs_to_create:
                    title = g_info.get("title", "Gap")
                    query = g_info.get("query", "")
                    gid = g_info.get("gap_id", "")
                    safe_stem = re.sub(r'[\\/*?:"<>|]', "_", title)
                    stub_file = self.vault.root_path / f"{safe_stem}.md"
                    try:
                        if not stub_file.exists():
                            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            stub_content = (
                                f"# {title}\n\n"
                                f"*Knowledge gap recorded by Alcuin on {date_str}.*\n"
                                f"*User Query: \"{query}\"*\n"
                            )
                            stub_file.write_text(stub_content, encoding="utf-8")
                            rel_stub = str(stub_file.relative_to(self.vault.root_path))
                            result.repaired.append(f"Created stub note '{rel_stub}' for knowledge gap '{query}'")
                            newly_created_stubs.append(rel_stub)
                            files_to_reindex.add(rel_stub)
                        gap_tracker.mark_stub_created(gid)
                    except Exception as e:
                        result.errors.append(f"Failed to create stub note for knowledge gap '{query}': {e}")
            except Exception as e:
                result.errors.append(f"Failed to process knowledge gap stubs: {e}")

        # 4. Execute Storage Drift / Orphan Purge
        if orphans_to_purge or excluded_to_remove:
            manifest = self.storage.load_manifest()
            manifest_files = manifest.get("files", {})
            for orphan in orphans_to_purge:
                try:
                    safe_path = orphan.replace("'", "''")
                    self.storage.table.delete(f"file_path = '{safe_path}'")
                    result.repaired.append(f"Purged orphan LanceDB records for '{orphan}'")
                except Exception as e:
                    result.errors.append(f"Failed to purge LanceDB records for '{orphan}': {e}")

            for excl in excluded_to_remove:
                try:
                    safe_path = excl.replace("'", "''")
                    self.storage.table.delete(f"file_path = '{safe_path}'")
                    if excl in manifest_files:
                        del manifest_files[excl]
                    result.repaired.append(f"Removed excluded file '{excl}' from manifest and LanceDB")
                except Exception as e:
                    result.errors.append(f"Failed to remove excluded file '{excl}': {e}")

            manifest["files"] = manifest_files
            self.storage.save_manifest(manifest)

        # 5. Trigger Incremental Indexing for Pending / Reindexed Files
        if files_to_reindex:
            try:
                report = self.storage.index(force=False)
                result.repaired.append(
                    f"Synchronized LanceDB index ({len(report.added_files)} added, {len(report.modified_files)} modified, {report.total_chunks_indexed} chunks upserted)"
                )
            except Exception as e:
                result.errors.append(f"Indexing sync failed: {e}")

        return result
