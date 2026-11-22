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
_MULTI_VALUE_FIELDS = {"Key Date", "Case Number", "Role / Relationship"}

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

        profiles: Dict[str, str] = {}
        total_conflicts = 0
        overrides_data = load_curated_overrides(self.vault.root_path)
        overrides_map = overrides_data.get("overrides", {})
        aliases_map = overrides_data.get("aliases", {})
        entity_aliases: Dict[str, List[str]] = {}

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

                # Resolve chained aliases transitively with cycle guard (e.g. A -> B -> C)
                orig_display = display_name
                alias_target = None
                seen_hops = {ent_key, display_name.lower()}
                while True:
                    nxt = aliases_map.get(ent_key) or aliases_map.get(display_name.lower())
                    if not nxt:
                        break
                    nxt_key, nxt_display = _canonical_entity_key_and_name(nxt, ent_type)
                    if nxt_key in seen_hops and nxt_display.lower() in seen_hops:
                        break
                    seen_hops.add(nxt_key)
                    seen_hops.add(nxt_display.lower())
                    alias_target = nxt
                    ent_key, display_name = nxt_key, nxt_display
                if alias_target and orig_display.lower() != display_name.lower():
                    entity_aliases.setdefault(ent_key, [])
                    if orig_display not in entity_aliases[ent_key]:
                        entity_aliases[ent_key].append(orig_display)

                if ent_key not in entities:
                    entities[ent_key] = {
                        "display_name": display_name,
                        "type": ent_type,
                        "fields": {},
                    }
                elif not alias_target and len(display_name) > len(entities[ent_key]["display_name"]):
                    entities[ent_key]["display_name"] = display_name

                entities[ent_key]["fields"].setdefault(norm_fld, []).append(
                    (auth, val_raw, src)
                )

        for ent_key, ent_data in entities.items():
            display = ent_data["display_name"]
            ent_type = ent_data["type"]
            fields_map: Dict[str, List[Tuple[int, str, str]]] = ent_data["fields"]
            subdir = _ORG_SUBDIR if ent_type == "organization" else _PEOPLE_SUBDIR
            rel_out = f"{subdir}/{_sanitize_entity_filename(display)}.md"
            ent_key_prefix = f"{subdir}/{_sanitize_entity_filename(display)}"

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
                ov_key = f"{ent_key_prefix}::{fld_name}"
                if ov_key in overrides_map:
                    ov = overrides_map[ov_key]
                    if ov.get("mode") == "keep_all":
                        for entry in ov.get("values", []):
                            lines.append(f"- **{fld_name}**: {entry['value']} *(Source: `{entry['source']}`)*")
                    else:
                        lines.append(f"- **{fld_name}**: {ov['value']} *(Source: `{ov['source']}`)*")
                    continue

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
                if fld_name in _MULTI_VALUE_FIELDS:
                    for _, m_val, m_srcs in ranked:
                        lines.append(f"- **{fld_name}**: {m_val} *(Source: `{', '.join(m_srcs[:3])}`)*")
                    continue

                _, primary_val, primary_srcs = ranked[0]
                src_str = ", ".join(primary_srcs[:3])
                lines.append(f"- **{fld_name}**: {primary_val} *(Source: `{src_str}`)*")

                if len(ranked) > 1:
                    total_conflicts += 1
                    for _, alt_val, alt_srcs in ranked[1:]:
                        conflicts.append(
                            f"- **{fld_name}**: `{alt_val}` *(Source: `{', '.join(alt_srcs[:2])}`)* — differs from canonical `{primary_val}`"
                        )

            if entity_aliases.get(ent_key) and "Aliases / Former Names" not in fields_map:
                alias_str = ", ".join(sorted(entity_aliases[ent_key]))
                lines.append(f"- **Aliases / Former Names**: {alias_str} *(Source: `Human Verified`)*")

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
            import os

            curated_root = self.vault.root_path / _CURATED_DIR_NAME
            curated_root.mkdir(parents=True, exist_ok=True)
            written_paths = set()
            for rel_out, md_body in profiles.items():
                dest = curated_root / rel_out
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp_dest = dest.with_suffix(".md.tmp")
                tmp_dest.write_text(md_body, encoding="utf-8")
                os.replace(tmp_dest, dest)
                written_paths.add(dest.resolve())
            # Remove stale generated profile cards in managed subdirectories without touching verified_facts.md
            for managed_sub in (_PEOPLE_SUBDIR, _ORG_SUBDIR):
                sub_dir = curated_root / managed_sub
                if sub_dir.is_dir():
                    for existing in list(sub_dir.glob("*.md")):
                        if existing.resolve() not in written_paths:
                            existing.unlink(missing_ok=True)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_state = state_path.with_suffix(".json.tmp")
            tmp_state.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
            os.replace(tmp_state, state_path)

        return report


# ---------------------------------------------------------------------------
# SPEC-043: Assisted Curation (--review) & Knowledge Gap Resolution Helpers
# ---------------------------------------------------------------------------

def _overrides_path(vault_root: Path) -> Path:
    return vault_root / ".alcuin" / "curated_overrides.json"


def load_curated_overrides(vault_root: Path) -> Dict[str, Any]:
    """Load persisted human-adjudicated overrides from .alcuin/curated_overrides.json."""
    path = _overrides_path(vault_root)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("overrides"), dict):
                data.setdefault("aliases", {})
                data.setdefault("not_merges", [])
                return data
        except Exception:
            pass
    return {"version": 1, "overrides": {}, "aliases": {}, "not_merges": []}


def save_curated_override(vault_root: Path, key: str, value: str, source: str) -> None:
    """Persist a single human-adjudicated conflict override immediately to disk."""
    from datetime import datetime, timezone

    data = load_curated_overrides(vault_root)
    data["overrides"][key] = {
        "value": value,
        "source": source,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _overrides_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_curated_keep_all_override(
    vault_root: Path, key: str, entries: List[Dict[str, str]]
) -> None:
    """Persist a multi-value keep-all override immediately to disk."""
    from datetime import datetime, timezone

    data = load_curated_overrides(vault_root)
    data["overrides"][key] = {
        "mode": "keep_all",
        "values": [{"value": e["value"], "source": e["source"]} for e in entries],
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _overrides_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply_keep_all_resolution(vault_root: Path, item: Dict[str, Any]) -> None:
    """Keep all competing values as valid Canonical Facts and clear the conflict."""
    opts = item["options"]
    save_curated_keep_all_override(vault_root, item["override_key"], opts)

    md_path: Path = item["profile_path"]
    if not md_path.is_file():
        return

    field = item["field"]
    lines = md_path.read_text(encoding="utf-8").splitlines()
    new_lines: List[str] = []
    section = ""
    canon_prefix = f"- **{field}**:"
    inserted = False

    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            section = s
            new_lines.append(ln)
            continue
        if section == "## Canonical Facts" and s.startswith(canon_prefix):
            if not inserted:
                for opt in opts:
                    new_lines.append(f"- **{field}**: {opt['value']} *(Source: `{opt['source']}`)*")
                inserted = True
        elif section == "## Conflicting / Alternate Mentions" and s.startswith(canon_prefix):
            continue
        else:
            new_lines.append(ln)

    if not inserted:
        for opt in opts:
            new_lines.append(f"- **{field}**: {opt['value']} *(Source: `{opt['source']}`)*")

    cleaned: List[str] = []
    for idx, ln in enumerate(new_lines):
        if ln.strip() == "## Conflicting / Alternate Mentions":
            remaining = [x for x in new_lines[idx + 1 :] if x.strip()]
            if not remaining:
                while cleaned and cleaned[-1].strip() == "":
                    cleaned.pop()
                break
        cleaned.append(ln)

    md_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def collect_conflicts(vault_root: Path) -> List[Dict[str, Any]]:
    """Scan _curated/ Markdown profiles for unresolved Conflicting / Alternate Mentions."""
    curated_root = vault_root / _CURATED_DIR_NAME
    if not curated_root.is_dir():
        return []

    overrides_map = load_curated_overrides(vault_root).get("overrides", {})
    items: List[Dict[str, Any]] = []

    canon_re = re.compile(r"^- \*\*(.+?)\*\*:\s*(.+?)\s*\*\(Source:\s*`([^`]*)`\)\*$")
    alt_re = re.compile(r"^- \*\*(.+?)\*\*:\s*`(.+?)`\s*\*\(Source:\s*`([^`]*)`\)\*\s*—\s*differs from canonical")

    for md_path in sorted(curated_root.rglob("*.md")):
        if md_path.name == "verified_facts.md":
            continue
        rel = md_path.relative_to(curated_root).as_posix()
        ent_key_prefix = rel[:-3] if rel.endswith(".md") else rel
        text = md_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        entity_name = md_path.stem
        for ln in lines:
            if ln.startswith("# "):
                entity_name = ln[2:].split("—")[0].strip()
                break

        section = ""
        canonical: Dict[str, Tuple[str, str]] = {}
        alternates: Dict[str, List[Tuple[str, str]]] = {}

        for ln in lines:
            s = ln.strip()
            if s.startswith("## "):
                section = s
                continue
            if section == "## Canonical Facts":
                m = canon_re.match(s)
                if m:
                    canonical[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
            elif section == "## Conflicting / Alternate Mentions":
                m = alt_re.match(s)
                if m:
                    alternates.setdefault(m.group(1), []).append((m.group(2).strip(), m.group(3).strip()))

        for fld, alts in alternates.items():
            ov_key = f"{ent_key_prefix}::{fld}"
            if ov_key in overrides_map:
                continue
            options: List[Dict[str, str]] = []
            if fld in canonical:
                c_val, c_src = canonical[fld]
                options.append({"value": c_val, "source": c_src, "is_canonical": "true"})
            for a_val, a_src in alts:
                options.append({"value": a_val, "source": a_src, "is_canonical": "false"})
            items.append({
                "profile_path": md_path,
                "rel_path": rel,
                "entity": entity_name,
                "field": fld,
                "override_key": ov_key,
                "options": options,
            })

    return items


def apply_conflict_resolution(
    vault_root: Path,
    item: Dict[str, Any],
    chosen_value: str,
    chosen_source: str,
) -> None:
    """Persist override and update the _curated/ profile card in place."""
    save_curated_override(vault_root, item["override_key"], chosen_value, chosen_source)

    md_path: Path = item["profile_path"]
    if not md_path.is_file():
        return

    field = item["field"]
    lines = md_path.read_text(encoding="utf-8").splitlines()
    new_lines: List[str] = []
    section = ""
    canon_prefix = f"- **{field}**:"
    replaced_canon = False

    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            section = s
            new_lines.append(ln)
            continue
        if section == "## Canonical Facts" and s.startswith(canon_prefix):
            new_lines.append(f"- **{field}**: {chosen_value} *(Source: `{chosen_source}`)*")
            replaced_canon = True
        elif section == "## Conflicting / Alternate Mentions" and s.startswith(canon_prefix):
            continue
        else:
            new_lines.append(ln)

    if not replaced_canon:
        new_lines.append(f"- **{field}**: {chosen_value} *(Source: `{chosen_source}`)*")

    # Strip empty ## Conflicting / Alternate Mentions header if no conflict bullets remain
    cleaned: List[str] = []
    for idx, ln in enumerate(new_lines):
        if ln.strip() == "## Conflicting / Alternate Mentions":
            remaining = [x for x in new_lines[idx + 1 :] if x.strip()]
            if not remaining:
                while cleaned and cleaned[-1].strip() == "":
                    cleaned.pop()
                break
        cleaned.append(ln)

    md_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def collect_open_gaps(vault_root: Path) -> List[Dict[str, Any]]:
    """Load unresolved knowledge gaps from .alcuin/knowledge_gaps.json."""
    gaps_path = vault_root / ".alcuin" / "knowledge_gaps.json"
    if not gaps_path.is_file():
        return []
    try:
        data = json.loads(gaps_path.read_text(encoding="utf-8"))
        gaps = data.get("gaps", {})
        return [
            dict(g, id=gid)
            for gid, g in gaps.items()
            if isinstance(g, dict) and g.get("status", "unresolved") == "unresolved"
        ]
    except Exception:
        return []


def resolve_gap(vault_root: Path, gap_id: str, answer: Optional[str] = None) -> bool:
    """Resolve a knowledge gap immediately, optionally writing a verified fact to _curated/verified_facts.md."""
    from datetime import datetime, timezone

    gaps_path = vault_root / ".alcuin" / "knowledge_gaps.json"
    if not gaps_path.is_file():
        return False

    data = json.loads(gaps_path.read_text(encoding="utf-8"))
    gaps = data.get("gaps", {})
    if gap_id not in gaps:
        return False

    now_iso = datetime.now(timezone.utc).isoformat()
    query_text = str(gaps[gap_id].get("query", gap_id)).strip()
    wrote_fact = False

    if answer and answer.strip():
        verified_path = vault_root / _CURATED_DIR_NAME / "verified_facts.md"
        verified_path.parent.mkdir(parents=True, exist_ok=True)
        if not verified_path.is_file():
            verified_path.write_text(
                "# Human-Verified Facts\n\n"
                "> Grounded answers supplied during assisted curation (`SPEC-043`).\n\n",
                encoding="utf-8",
            )
        existing = verified_path.read_text(encoding="utf-8")
        entry = (
            f"## {query_text}\n"
            f"- **Verified Answer**: {answer.strip()} *(Source: `Human Verified` — {now_iso[:10]})*\n\n"
        )
        verified_path.write_text(existing + entry, encoding="utf-8")
        wrote_fact = True

    gaps[gap_id]["status"] = "resolved"
    gaps[gap_id]["resolved_at"] = now_iso
    data["updated_at"] = now_iso
    gaps_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return wrote_fact


def _parse_profile_full(md_path: Path) -> Tuple[str, Dict[str, List[Tuple[str, str]]], List[str]]:
    """Return (display_name, {field: [(value, source), ...]}, conflict_lines) from a _curated/ profile card."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    display = md_path.stem
    for ln in lines:
        if ln.startswith("# "):
            display = ln[2:].split("—")[0].strip()
            break

    canon_re = re.compile(r"^- \*\*(.+?)\*\*:\s*(.+?)\s*\*\(Source:\s*`([^`]*)`\)\*$")
    facts: Dict[str, List[Tuple[str, str]]] = {}
    conflicts: List[str] = []
    section = ""
    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            section = s
            continue
        if section == "## Canonical Facts":
            m = canon_re.match(s)
            if m:
                facts.setdefault(m.group(1), []).append((m.group(2).strip(), m.group(3).strip()))
        elif section == "## Conflicting / Alternate Mentions":
            if s.startswith("- "):
                conflicts.append(s)
    return display, facts, conflicts


def _parse_profile_facts(md_path: Path) -> Tuple[str, Dict[str, Tuple[str, str]]]:
    """Return (display_name, {field: (first_value, source)}) from a _curated/ profile card."""
    display, full_facts, _ = _parse_profile_full(md_path)
    return display, {fld: vals[0] for fld, vals in full_facts.items() if vals}


def collect_merge_candidates(vault_root: Path) -> List[Dict[str, Any]]:
    """Detect likely duplicate entity profiles in _curated/ for Google Contacts-style merge review."""
    from difflib import SequenceMatcher

    curated_root = vault_root / _CURATED_DIR_NAME
    if not curated_root.is_dir():
        return []

    ov_data = load_curated_overrides(vault_root)
    not_merges = {tuple(sorted(pair)) for pair in ov_data.get("not_merges", []) if len(pair) == 2}

    profiles: List[Dict[str, Any]] = []
    for md_path in sorted(curated_root.rglob("*.md")):
        if md_path.name == "verified_facts.md":
            continue
        rel = md_path.relative_to(curated_root).as_posix()
        category = rel.split("/")[0] if "/" in rel else "people"
        display, facts = _parse_profile_facts(md_path)
        tokens = [t for t in re.findall(r"[a-z0-9]+", display.lower()) if t]
        profiles.append({
            "path": md_path,
            "rel_path": rel,
            "category": category,
            "display": display,
            "facts": facts,
            "tokens": tokens,
        })

    candidates: List[Dict[str, Any]] = []
    primary_id_fields = ("Date of Birth", "Email", "Phone")

    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a, b = profiles[i], profiles[j]
            if a["category"] != b["category"]:
                continue
            pair_key = tuple(sorted([a["rel_path"], b["rel_path"]]))
            if pair_key in not_merges:
                continue

            reasons: List[str] = []
            for fld in primary_id_fields:
                if fld in a["facts"] and fld in b["facts"]:
                    va = re.sub(r"\s+", " ", a["facts"][fld][0]).strip().lower()
                    vb = re.sub(r"\s+", " ", b["facts"][fld][0]).strip().lower()
                    if va and va == vb:
                        reasons.append(f"Shared {fld}: {a['facts'][fld][0]}")

            ta, tb = a["tokens"], b["tokens"]
            if ta and tb and ta[0] == tb[0]:
                if set(ta).issubset(set(tb)) or set(tb).issubset(set(ta)):
                    reasons.append(f"Name subset: '{a['display']}' / '{b['display']}'")
                elif SequenceMatcher(None, a["display"].lower(), b["display"].lower()).ratio() >= 0.85:
                    reasons.append(f"Similar name: '{a['display']}' / '{b['display']}'")

            if reasons:
                candidates.append({
                    "a": a,
                    "b": b,
                    "reason": "; ".join(reasons),
                })

    return candidates


def record_not_merge(vault_root: Path, rel_a: str, rel_b: str) -> None:
    """Persist a negative merge decision so this pair is not prompted again."""
    data = load_curated_overrides(vault_root)
    pair = sorted([rel_a, rel_b])
    if pair not in data["not_merges"]:
        data["not_merges"].append(pair)
    path = _overrides_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply_entity_merge(
    vault_root: Path,
    cand: Dict[str, Any],
    canonical_name: str,
) -> Path:
    """Merge two _curated/ profiles under canonical_name and persist aliases/overrides atomically."""
    import os

    a, b = cand["a"], cand["b"]
    curated_root = vault_root / _CURATED_DIR_NAME
    category = a["category"]
    target_stem = _sanitize_entity_filename(canonical_name)
    target_path = curated_root / category / f"{target_stem}.md"

    # Persist alias mappings, rewrite chained aliases, and re-key existing overrides
    ov_data = load_curated_overrides(vault_root)
    merged_away_names = set()
    for src_name in (a["display"], b["display"]):
        if src_name.lower() != canonical_name.lower():
            san_src = _sanitize_entity_filename(src_name)
            merged_away_names.add(src_name.lower())
            merged_away_names.add(san_src.lower())
            ov_data["aliases"][src_name.lower()] = canonical_name
            ov_data["aliases"][san_src.lower()] = canonical_name
            old_prefix = f"{category}/{san_src}::"
            new_prefix = f"{category}/{target_stem}::"
            for old_key in list(ov_data.get("overrides", {}).keys()):
                if old_key.startswith(old_prefix):
                    new_key = new_prefix + old_key[len(old_prefix):]
                    if new_key not in ov_data["overrides"]:
                        ov_data["overrides"][new_key] = ov_data["overrides"][old_key]
                    del ov_data["overrides"][old_key]
    # Rewrite any existing alias targets pointing to a merged-away entity so chained merges stay flat
    for ak, av in list(ov_data.get("aliases", {}).items()):
        if isinstance(av, str) and (av.lower() in merged_away_names or _sanitize_entity_filename(av).lower() in merged_away_names):
            ov_data["aliases"][ak] = canonical_name

    ov_path = _overrides_path(vault_root)
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_ov = ov_path.with_suffix(".json.tmp")
    tmp_ov.write_text(json.dumps(ov_data, indent=2), encoding="utf-8")
    os.replace(tmp_ov, ov_path)

    # Combine all multi-value facts, competing single-value mentions, conflicts, and aliases
    merged_facts: Dict[str, List[Tuple[str, str]]] = {}
    merged_conflicts: List[str] = []
    merged_aliases: List[str] = []
    for prof in (a, b):
        if prof["display"].lower() != canonical_name.lower() and prof["display"] not in merged_aliases:
            merged_aliases.append(prof["display"])
        if prof["path"].is_file():
            _, prof_full_facts, prof_conflicts = _parse_profile_full(prof["path"])
        else:
            prof_full_facts = {k: [v] for k, v in prof.get("facts", {}).items()}
            prof_conflicts = []
        for cline in prof_conflicts:
            if cline not in merged_conflicts:
                merged_conflicts.append(cline)
        for fld, val_list in prof_full_facts.items():
            if fld == "Aliases / Former Names":
                for val, _ in val_list:
                    for al in [x.strip() for x in val.split(",") if x.strip()]:
                        if al not in merged_aliases and al.lower() != canonical_name.lower():
                            merged_aliases.append(al)
                continue
            for val, src in val_list:
                norm_v = re.sub(r"\s+", " ", val).strip().lower()
                if fld not in merged_facts:
                    merged_facts[fld] = [(val, src)]
                else:
                    existing_norms = {re.sub(r"\s+", " ", ev).strip().lower() for ev, _ in merged_facts[fld]}
                    if norm_v in existing_norms:
                        continue
                    if fld in _MULTI_VALUE_FIELDS or len(merged_facts[fld]) > 1:
                        merged_facts[fld].append((val, src))
                    else:
                        canon_val = merged_facts[fld][0][0]
                        alt_line = f"- **{fld}**: `{val}` *(Source: `{src}`)* — differs from canonical `{canon_val}`"
                        if alt_line not in merged_conflicts:
                            merged_conflicts.append(alt_line)

    lines = [
        f"# {canonical_name} — Canonical Profile",
        "",
        "> Curated automatically by Alcuin (`SPEC-041`). Formal records take precedence over informal mentions.",
        "",
        "## Canonical Facts",
    ]
    for fld in sorted(merged_facts.keys()):
        for val, src in merged_facts[fld]:
            lines.append(f"- **{fld}**: {val} *(Source: `{src}`)*")
    if merged_aliases:
        lines.append(f"- **Aliases / Former Names**: {', '.join(sorted(merged_aliases))} *(Source: `Human Verified`)*")
    if merged_conflicts:
        lines.extend(["", "## Conflicting / Alternate Mentions"])
        lines.extend(merged_conflicts)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target_path.with_suffix(".md.tmp")
    tmp_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_target, target_path)

    for p in (a["path"], b["path"]):
        if p.resolve() != target_path.resolve() and p.is_file():
            p.unlink()

    return target_path
