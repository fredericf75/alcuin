"""Unit tests for SPEC-041 Vault Curation & Canonical Entity Profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from click.testing import CliRunner

from alcuin.cli.main import cli
from alcuin.storage.curator import (
    VaultCurator,
    _is_fact_bearing_chunk,
    _parse_facts_json,
    _source_authority,
)
from alcuin.storage.vault import Vault


class _FakeLLM:
    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[tuple] = []

    def chat(
        self,
        sys_prompt: str,
        user_prompt: str,
        return_metadata: bool = False,
        temperature: float | None = None,
    ):
        self.calls.append((sys_prompt, user_prompt))
        resp = self.responses.pop(0) if self.responses else "[]"
        if return_metadata:
            return resp, {"finish_reason": "stop"}
        return resp


def test_parse_facts_json_and_authority():
    raw = """Here is the JSON:
    [
      {"entity": "Tara Emick", "field": "DOB", "value": "1983-07-23"},
      {"entity": "Richard Sommerville Lee", "field": "dob", "value": "November 18, 1955"},
      {"entity": "Someone", "field": "Phone", "value": "unknown"}
    ]"""
    facts = _parse_facts_json(raw)
    assert len(facts) == 2
    assert facts[0]["entity"] == "Tara Emick"
    assert _source_authority("Tara Order Summary.pdf") == 3
    assert _source_authority("2025-07-18 - Email from Dan.pdf") == 1


def test_is_fact_bearing_chunk():
    assert _is_fact_bearing_chunk({"chunk_type": "table", "content": "| a | b |"})
    assert _is_fact_bearing_chunk({"chunk_type": "text", "content": "Born November 12, 2009."})
    assert not _is_fact_bearing_chunk({"chunk_type": "text", "content": "We went to the park yesterday."})


def test_curator_consolidates_conflicts_and_caches(tmp_path, monkeypatch):
    vault_dir = tmp_path / "vault"
    vault = Vault.init(vault_dir)

    chunks = [
        {
            "id": "c1",
            "file_path": "2026-03-10 - Case Summary.md",
            "section_title": "Parties",
            "chunk_type": "table",
            "content": "- Plaintiff: Tara Emick\n- Defendant: Richard Sommerville Lee, DOB November 18, 1955",
        },
        {
            "id": "c2",
            "file_path": "Tara Order Summary.pdf",
            "section_title": "Header",
            "chunk_type": "text",
            "content": "Tara Emick Date of birth: 1983-07-23",
        },
        {
            "id": "c3",
            "file_path": "2026-01-01 - Email note.md",
            "section_title": "Notes",
            "chunk_type": "text",
            "content": "Tara Emick DOB 1983-07-24 maybe?",
        },
        {
            "id": "c4",
            "file_path": "random.md",
            "section_title": "ChitChat",
            "chunk_type": "text",
            "content": "Just a quick note about groceries.",
        },
    ]

    llm = _FakeLLM(
        [
            json.dumps(
                [
                    {"entity": "Tara M. Emick", "type": "person", "field": "Role", "value": "Plaintiff"},
                    {"entity": "Richard Sommerville Lee", "type": "person", "field": "DOB", "value": "November 18, 1955"},
                    {"entity": "Acme Legal LLC", "type": "organization", "field": "Case Number", "value": "23CV001"},
                    {"entity": "Acme Legal LLC", "type": "organization", "field": "Phone", "value": "555-0100"},
                ]
            ),
            json.dumps(
                [
                    {"entity": "Emick, Tara Marie", "type": "person", "field": "Date of Birth", "value": "07/23/1983"},
                    {"entity": "Tara Emick", "type": "person", "field": "Divorce Strategy", "value": "ignore me"},
                ]
            ),
            json.dumps([{"entity": "Tara Emick", "type": "person", "field": "Date of Birth", "value": "1983-07-24"}]),
        ]
    )

    curator = VaultCurator(vault, llm=llm)
    assert VaultCurator(vault).llm.context_window == vault.config.context_window
    monkeypatch.setattr(curator, "_load_raw_chunks", lambda: chunks)

    r1 = curator.curate()
    assert r1.chunks_scanned == 4
    assert r1.chunks_extracted == 3
    assert r1.chunks_skipped == 1
    assert r1.conflicts_found == 1
    assert set(r1.entities_written) == {
        "organizations/Acme Legal LLC.md",
        "people/Richard Sommerville Lee.md",
        "people/Tara Marie Emick.md",
    }

    tara_md = (vault_dir / "_curated" / "people" / "Tara Marie Emick.md").read_text(encoding="utf-8")
    assert "**Date of Birth**: 1983-07-23" in tara_md
    assert "Divorce Strategy" not in tara_md
    assert "Tara Order Summary.pdf" in tara_md
    assert "## Conflicting / Alternate Mentions" in tara_md
    assert "1983-07-24" in tara_md
    assert "November 18, 1955" not in tara_md

    # Second run -> 100% cached, 0 LLM calls
    llm2 = _FakeLLM([])
    curator2 = VaultCurator(vault, llm=llm2)
    monkeypatch.setattr(curator2, "_load_raw_chunks", lambda: chunks)
    r2 = curator2.curate()
    assert r2.chunks_cached == 3
    assert r2.chunks_extracted == 0
    assert len(llm2.calls) == 0


def test_cli_curate_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["curate", "--help"])
    assert res.exit_code == 0
    assert "--model" in res.output
    assert "--index" in res.output
