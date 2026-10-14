"""
alcuin.storage.gaps - Knowledge Gap Tracking and Unanswered Query Logging
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from alcuin.storage.vault import Vault

QUESTION_PREFIX_PATTERN = re.compile(
    r"^(?:what\s+(?:is|are|was|were|do|does)|who\s+(?:is|are|was|were)|where\s+(?:is|are|was|were)|when\s+(?:is|was)|how\s+(?:much|many|is|are)|can\s+you\s+tell\s+me|tell\s+me\s+about|do\s+we\s+have)\s+",
    re.IGNORECASE,
)


def normalize_query_key(query: str) -> str:
    """Normalize query text for deduplication: lowercase, strip punctuation and extra spaces."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    return " ".join(cleaned.split())


def suggest_title_from_query(query: str) -> str:
    """Derive a clean document title from a user query."""
    # Strip question prefix if present
    stripped = QUESTION_PREFIX_PATTERN.sub("", query.strip())
    # Strip trailing punctuation
    stripped = re.sub(r"[?.,!]+$", "", stripped).strip()
    if not stripped:
        stripped = query.strip()
    # Replace characters that cannot be used in filenames
    clean_title = re.sub(r'[\\/*?:"<>|]', "", stripped)
    # Convert to title case
    words = clean_title.split()
    return " ".join(w.capitalize() for w in words) if words else "Untitled Knowledge Gap"


@dataclass
class KnowledgeGap:
    id: str
    query: str
    normalized_key: str
    suggested_title: str
    status: str = "unresolved"  # "unresolved", "resolved", "dismissed"
    first_asked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_asked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ask_count: int = 1
    last_confidence: float = 0.0
    resolved_at: Optional[str] = None
    stub_created: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeGap:
        return cls(**data)


class KnowledgeGapTracker:
    """Manages persistent tracking, deduplication, and resolution of knowledge gaps."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self.gap_file = vault.alcuin_dir / "knowledge_gaps.json"

    def load_gaps(self) -> Dict[str, KnowledgeGap]:
        if not self.gap_file.exists():
            return {}
        try:
            with open(self.gap_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_gaps = data.get("gaps", {})
                return {k: KnowledgeGap.from_dict(v) for k, v in raw_gaps.items()}
        except Exception:
            return {}

    def save_gaps(self, gaps: Dict[str, KnowledgeGap]) -> None:
        self.vault.alcuin_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "gaps": {k: v.to_dict() for k, v in gaps.items()},
        }
        with open(self.gap_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def record_query(
        self,
        query: str,
        confidence: float,
        evidence_count: int,
        answer: str = "",
    ) -> Optional[KnowledgeGap]:
        """
        Record a query result:
        - If unanswered (evidence == 0 or confidence < 0.3 or non-finding text), record as an unresolved gap.
        - If answered with high confidence (>= 0.7), mark any matching existing gap as resolved.
        """
        clean_q = query.strip()
        if len(clean_q) < 4:
            return None

        norm_key = normalize_query_key(clean_q)
        if not norm_key:
            return None

        gaps = self.load_gaps()
        now_str = datetime.now(timezone.utc).isoformat()

        # Heuristic for non-finding answers
        ans_lower = answer.lower()
        is_non_finding = (
            evidence_count == 0
            or confidence < 0.3
            or "no relevant documents" in ans_lower
            or "information is not found" in ans_lower
            or "not contained in the vault" in ans_lower
            or "not found in the provided evidence" in ans_lower
        )

        if is_non_finding:
            # Check if this normalized query already exists
            gap_id = None
            for gid, gap in gaps.items():
                if gap.normalized_key == norm_key:
                    gap_id = gid
                    break

            if gap_id:
                gap = gaps[gap_id]
                gap.ask_count += 1
                gap.last_asked_at = now_str
                gap.last_confidence = confidence
                # Reopen if previously dismissed or resolved
                gap.status = "unresolved"
            else:
                # Create a readable ID
                raw_id = re.sub(r"[^\w]+", "_", norm_key[:40]).strip("_") or "gap"
                gap_id = f"{raw_id}_{len(gaps) + 1}"
                gap = KnowledgeGap(
                    id=gap_id,
                    query=clean_q,
                    normalized_key=norm_key,
                    suggested_title=suggest_title_from_query(clean_q),
                    status="unresolved",
                    first_asked_at=now_str,
                    last_asked_at=now_str,
                    ask_count=1,
                    last_confidence=confidence,
                )
                gaps[gap_id] = gap

            self.save_gaps(gaps)
            return gap

        # If answer was found with high confidence, auto-resolve any matching unresolved gap
        if confidence >= 0.7:
            resolved_any = False
            for gid, gap in gaps.items():
                if gap.normalized_key == norm_key and gap.status == "unresolved":
                    gap.status = "resolved"
                    gap.resolved_at = now_str
                    resolved_any = True
            if resolved_any:
                self.save_gaps(gaps)

        return None

    def list_gaps(self, include_resolved: bool = False) -> List[KnowledgeGap]:
        gaps = self.load_gaps()
        results = []
        for g in gaps.values():
            if include_resolved or g.status == "unresolved":
                results.append(g)
        # Sort by ask_count descending, then last_asked_at descending
        results.sort(key=lambda x: (x.ask_count, x.last_asked_at), reverse=True)
        return results

    def resolve_gap(self, target: str) -> bool:
        """Resolve a gap by ID or query substring."""
        gaps = self.load_gaps()
        target_norm = normalize_query_key(target)
        now_str = datetime.now(timezone.utc).isoformat()

        found = False
        for gid, gap in gaps.items():
            if gid == target or target_norm in gap.normalized_key or gap.normalized_key in target_norm:
                gap.status = "resolved"
                gap.resolved_at = now_str
                found = True

        if found:
            self.save_gaps(gaps)
        return found

    def mark_stub_created(self, gap_id: str) -> bool:
        gaps = self.load_gaps()
        if gap_id in gaps:
            gaps[gap_id].stub_created = True
            self.save_gaps(gaps)
            return True
        return False

    def clear_gaps(self, only_resolved: bool = True) -> int:
        gaps = self.load_gaps()
        before_count = len(gaps)
        if only_resolved:
            gaps = {k: v for k, v in gaps.items() if v.status == "unresolved"}
        else:
            gaps = {}
        self.save_gaps(gaps)
        return before_count - len(gaps)
