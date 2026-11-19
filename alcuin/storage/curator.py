"""Vault curation engine for synthesizing canonical entity profiles (SPEC-041)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from alcuin.agent.graph import LLMClient
from alcuin.storage.engine import VaultStorage
from alcuin.storage.vault import Vault

logger = logging.getLogger(__name__)

_CURATED_DIR_NAME = "_curated"
_PEOPLE_SUBDIR = "people"
_ORG_SUBDIR = "organizations"

_FACT_SIGNAL_RE = re.compile(
    r"(?:"
    r"\b(?:dob|birth(?:day|date)?|born|ssn|policy|account|case\s*(?:no|number|#)|phone|email|address)\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r")",
    re.IGNORECASE,
)

_HIGH_AUTHORITY_RE = re.compile(r"(?:order|decree|judgment|certificate|summary|filing|petition|contract)", re.IGNORECASE)
_LOW_AUTHORITY_RE = re.compile(r"(?:transcript|email|correspondence|chat|text|sms|message)", re.IGNORECASE)
_ILLEGAL_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_ORG_SUFFIX_RE = re.compile(
    r"\b(?:LLC|Inc\.?|Corp\.?|Ltd\.?|PA|PLLC|LLP|Bank|County|Department|Office|Service|School|University|Hospital|Court|Solutions|Properties|Ventures|Foundation|Association|Group|Firm)\b",
    re.IGNORECASE,
)
_PRIMARY_FIELDS = {"Date of Birth", "Email", "Phone"}


@dataclass
class CurationReport:
    """Summary of a vault curation run."""

    chunks_scanned: int = 0
    chunks_extracted: int = 0
    chunks_cached: int = 0
    chunks_skipped: int = 0
    entities_written: List[str] = field(default_factory=list)
    conflicts_found: int = 0
    errors: List[str] = field(default_factory=list)


def _chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def _source_authority(file_path: str, chunk_type: str = "") -> int:
    """Rank source authority: formal records (3) > tables/docs (2) > emails/transcripts (1)."""
    if _HIGH_AUTHORITY_RE.search(file_path):
        return 3
    if _LOW_AUTHORITY_RE.search(file_path):
        return 1
    if chunk_type == "table" or "TABLE_DATA" in file_path:
        return 2
    return 2


def _sanitize_entity_filename(entity: str) -> str:
    cleaned = _ILLEGAL_NAME_RE.sub("_", entity).strip().rstrip(".")
    return cleaned or "Unknown_Entity"


def _is_fact_bearing_chunk(chunk: Dict[str, Any]) -> bool:
    """Fast heuristic filter to skip boilerplate chunks when --all is not set."""
    if (chunk.get("chunk_type") or "") == "table":
        return True
    content = chunk.get("content") or ""
    if "TABLE_DATA" in content or "Key-Value Schedule" in content:
        return True
    return bool(_FACT_SIGNAL_RE.search(content))


def _normalize_field_name(fld: str) -> Optional[str]:
    """Map raw field labels to a strict canonical whitelist, discarding arbitrary/malformed keys."""
    low = re.sub(r"\s+", " ", fld.strip().lower())
    if not low or "|" in low or len(low) > 40:
        return None
    if low in {"dob", "birth date", "birthdate", "date of birth", "birthday"}:
        return "Date of Birth"
    if low in {"role", "relationship", "title", "role / relationship"}:
        return "Role / Relationship"
    if low in {"email", "email address", "e-mail"}:
        return "Email"
    if low in {"phone", "phone number", "telephone", "cell", "mobile"}:
        return "Phone"
    if low in {"address", "mailing address", "home address", "street address"}:
        return "Address"
    if low in {"case number", "case no", "case #", "docket", "docket number"}:
        return "Case Number"
    if low in {"account", "account number", "policy", "policy number"}:
        return "Account"
    if low in {"id", "id number", "license", "license number", "ssn"}:
        return "ID / License"
    if low in {"key date", "effective date", "date"}:
        return "Key Date"
    return None


def _normalize_date_value(val: str) -> Tuple[str, str]:
    """Return (canonical_key, display_value), normalizing common date formats to ISO YYYY-MM-DD."""
    s = re.sub(r"\s+", " ", val).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            iso = datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            return iso, iso
        except ValueError:
            continue
    return s.lower(), s


def _classify_entity_type(name: str, explicit_type: str = "") -> str:
    t = explicit_type.strip().lower()
    if t in {"person", "organization"}:
        if t == "person" and _ORG_SUFFIX_RE.search(name):
            return "organization"
        return t
    if _ORG_SUFFIX_RE.search(name):
        return "organization"
    return "person"


def _canonical_entity_key_and_name(raw_name: str, ent_type: str) -> Tuple[str, str]:
    """Normalize 'Last, First [Middle]' and strip middle initials for person entities."""
    name = re.sub(r"\s+", " ", raw_name).strip()
    if ent_type == "organization":
        return f"org:{name.lower()}", name

    # Convert "Last, First Middle" -> "First Middle Last"
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1] and len(parts[0].split()) <= 2:
            name = f"{parts[1]} {parts[0]}"

    if name.isupper():
        name = name.title()

    tokens = [t for t in name.split() if t]
    # Drop middle initials/names when forming merge key for "First [Middle...] Last"
    if len(tokens) >= 3:
        key_name = f"{tokens[0]} {tokens[-1]}".lower()
        display_tokens = [t for t in tokens if len(t.rstrip(".")) > 1]
        display = " ".join(display_tokens) if len(display_tokens) >= 2 else name
    else:
        key_name = name.lower()
        display = name

    return f"person:{key_name}", display


def _parse_facts_json(raw_text: str) -> List[Dict[str, str]]:
    """Parse JSON array of {entity, type?, field, value} objects from LLM output."""
    text = (raw_text or "").strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity", "")).strip()
        fld = str(item.get("field", "")).strip()
        val = str(item.get("value", "")).strip()
        ent_type = str(item.get("type", "")).strip()
        if not entity or not fld or not val:
            continue
        if val.startswith(("{", "[")) or len(val) > 200:
            continue
        if val.lower() in {"unknown", "n/a", "none", "not found", "not specified"}:
            continue
        cleaned.append({"entity": entity, "type": ent_type, "field": fld, "value": val})
    return cleaned


class VaultCurator:
    """Synthesizes canonical Markdown entity profiles from indexed vault chunks."""

    def __init__(self, vault: Vault, llm: Optional[LLMClient] = None):
        self.vault = vault
        self.storage = VaultStorage(vault)
        self.llm = llm or LLMClient(
            model=vault.config.model_name or "default",
            endpoint=vault.config.model_endpoint,
            temperature=0.0,
            context_window=vault.config.context_window,
            think=False,
        )

    def _load_raw_chunks(self) -> List[Dict[str, Any]]:
        """Read all non-_curated chunks directly from the vault's LanceDB table."""
        table = self.storage._get_or_create_table()
        rows = table.to_arrow().to_pylist()
        return [
            r
            for r in rows
            if not str(r.get("file_path", "")).replace("\\", "/").startswith(f"{_CURATED_DIR_NAME}/")
        ]

    def _extract_chunk_facts(self, chunk: Dict[str, Any]) -> List[Dict[str, str]]:
        """Ask the LLM to extract strictly entity-bound facts from a single chunk."""
        sys_prompt = (
            "You are a precision knowledge curator.\n"
            "Extract ONLY concrete, explicitly stated facts about named people or organizations from the excerpt.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a JSON array of objects: "
            '[{"entity": "Full Name", "type": "person|organization", "field": "Date of Birth|Role|Email|Phone|Address|Case Number|Account|ID|Key Date", "value": "..."}].\n'
            "2. Use ONLY those exact field names. Use ISO YYYY-MM-DD for dates when possible.\n"
            "3. STRICT ENTITY BINDING: Only bind a date, DOB, email, phone, or ID to a person if the text explicitly links them. "
            "Never assign one person's DOB or attribute to another person in the same table or paragraph.\n"
            "4. If no concrete entity facts are present, output [].\n"
            "5. Output ONLY valid JSON, no markdown prose."
        )
        src = chunk.get("file_path", "unknown")
        sec = chunk.get("section_title") or ""
        content = (chunk.get("content") or "")[:3500]
        user_prompt = f"[Source: {src} | Section: {sec}]\n{content}"
        raw_resp = self.llm.chat(sys_prompt, user_prompt)
        return _parse_facts_json(raw_resp)

    def _consolidate_profiles(
        self,
        chunk_records: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, str], int]:
        """Merge extracted facts into canonical Markdown profile cards keyed by relative path."""
        entities: Dict[str, Dict[str, Any]] = {}

        for rec in chunk_records.values():
            src = rec.get("file_path", "unknown")
            auth = int(rec.get("authority", 2))
            for fact in rec.get("facts", []):
                norm_fld = _normalize_field_name(fact.get("field", ""))
                if not norm_fld:
                    continue
                val_raw = str(fact.get("value", "")).strip()
                if not val_raw or val_raw.startswith(("{", "[")) or len(val_raw) > 200:
                    continue

                ent_type = _classify_entity_type(fact["entity"], fact.get("type", ""))
                ent_key, display_name = _canonical_entity_key_and_name(fact["entity"], ent_type)

                if ent_key not in entities:
                    entities[ent_key] = {
                        "display_name": display_name,
                        "type": ent_type,
                        "fields": {},
                    }
                elif len(display_name) > len(entities[ent_key]["display_name"]):
                    entities[ent_key]["display_name"] = display_name

                entities[ent_key]["fields"].setdefault(norm_fld, []).append(
                    (auth, val_raw, src)
                )

        profiles: Dict[str, str] = {}
        total_conflicts = 0

        for ent_data in entities.values():
            display = ent_data["display_name"]
            ent_type = ent_data["type"]
            fields_map: Dict[str, List[Tuple[int, str, str]]] = ent_data["fields"]

            # Prune low-signal / one-off entities lacking primary identifiers and multi-field depth
            if len(fields_map) < 2 and not any(k in _PRIMARY_FIELDS for k in fields_map):
                continue

            lines = [
                f"# {display} — Canonical Profile",
                "",
                "> Curated automatically by Alcuin (`SPEC-041`). Formal records take precedence over informal mentions.",
                "",
                "## Canonical Facts",
            ]
            conflicts: List[str] = []

            for fld_name in sorted(fields_map.keys()):
                mentions = fields_map[fld_name]
                by_val: Dict[str, Tuple[int, str, List[str]]] = {}
                is_date_field = fld_name in {"Date of Birth", "Key Date"}

                for auth, val, src in mentions:
                    if is_date_field:
                        vkey, disp_val = _normalize_date_value(val)
                    else:
                        vkey = re.sub(r"\s+", " ", val).strip().lower()
                        disp_val = val
                    if vkey not in by_val:
                        by_val[vkey] = (auth, disp_val, [src])
                    else:
                        best_auth, best_val, srcs = by_val[vkey]
                        if auth > best_auth:
                            best_auth = auth
                        if src not in srcs:
                            srcs.append(src)
                        by_val[vkey] = (best_auth, best_val, srcs)

                ranked = sorted(
                    by_val.values(),
                    key=lambda item: (item[0], len(item[2])),
                    reverse=True,
                )
                _, primary_val, primary_srcs = ranked[0]
                src_str = ", ".join(primary_srcs[:3])
                lines.append(f"- **{fld_name}**: {primary_val} *(Source: `{src_str}`)*")

                if len(ranked) > 1:
                    total_conflicts += 1
                    for _, alt_val, alt_srcs in ranked[1:]:
                        conflicts.append(
                            f"- **{fld_name}**: `{alt_val}` *(Source: `{', '.join(alt_srcs[:2])}`)* — differs from canonical `{primary_val}`"
                        )

            if conflicts:
                lines.extend(["", "## Conflicting / Alternate Mentions", *conflicts])

            subdir = _ORG_SUBDIR if ent_type == "organization" else _PEOPLE_SUBDIR
            rel_out = f"{subdir}/{_sanitize_entity_filename(display)}.md"
            profiles[rel_out] = "\n".join(lines) + "\n"

        return profiles, total_conflicts

    def curate(
        self,
        process_all: bool = False,
        dry_run: bool = False,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> CurationReport:
        """Run incremental entity fact extraction and write canonical profile cards."""
        state_path = self.vault.root_path / ".alcuin" / "curation_state.json"
        state: Dict[str, Dict[str, Any]] = {}
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load %s, starting fresh: %s", state_path, exc)

        chunks = self._load_raw_chunks()
        report = CurationReport(chunks_scanned=len(chunks))

        candidates = chunks if process_all else [c for c in chunks if _is_fact_bearing_chunk(c)]
        report.chunks_skipped = len(chunks) - len(candidates)

        new_state: Dict[str, Dict[str, Any]] = {}
        total_cands = len(candidates)

        for idx, chunk in enumerate(candidates, 1):
            cid = str(chunk.get("id", ""))
            content = str(chunk.get("content", ""))
            chash = _chunk_hash(content)
            fpath = str(chunk.get("file_path", ""))
            ctype = str(chunk.get("chunk_type", ""))

            if progress_cb:
                progress_cb(idx, total_cands, fpath)

            prior = state.get(cid)
            if prior and prior.get("hash") == chash:
                new_state[cid] = prior
                report.chunks_cached += 1
                continue

            try:
                facts = self._extract_chunk_facts(chunk)
                new_state[cid] = {
                    "hash": chash,
                    "file_path": fpath,
                    "authority": _source_authority(fpath, ctype),
                    "facts": facts,
                }
                report.chunks_extracted += 1
            except Exception as exc:
                logger.warning("Curation extraction failed for chunk %s (%s): %s", cid, fpath, exc)
                report.errors.append(f"{fpath}: {exc}")
                if prior:
                    new_state[cid] = prior

        profiles, conflicts = self._consolidate_profiles(new_state)
        report.conflicts_found = conflicts
        report.entities_written = sorted(profiles.keys())

        if not dry_run:
            curated_root = self.vault.root_path / _CURATED_DIR_NAME
            if curated_root.exists():
                shutil.rmtree(curated_root)
            for rel_out, md_body in profiles.items():
                dest = curated_root / rel_out
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(md_body, encoding="utf-8")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")

        return report
