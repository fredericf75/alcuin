"""Unit tests for Alcuin research-first CLI."""

import os
from pathlib import Path
from click.testing import CliRunner
import pytest
from alcuin.cli.main import cli
from alcuin.agent.graph import ResearchAgent


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    runner = CliRunner()
    return runner, tmp_path


def test_cli_init_and_vault_list(cli_env):
    runner, tmp_path = cli_env
    vault_path = tmp_path / "my_notes"

    # 1. Test init
    result = runner.invoke(cli, ["init", str(vault_path), "--name", "notes"])
    assert result.exit_code == 0
    assert "Initialized vault 'notes'" in result.output
    assert (vault_path / ".alcuin" / "config.toml").is_file()

    # 2. Test vault list
    list_res = runner.invoke(cli, ["vault", "list"])
    assert list_res.exit_code == 0
    assert "notes" in list_res.output


def test_cli_bare_question_research(cli_env, monkeypatch):
    runner, tmp_path = cli_env
    vault_path = tmp_path / "health_records"
    vault_path.mkdir()

    # Add doc
    doc = vault_path / "pediatrician.md"
    doc.write_text("# Pediatrician\n\nCharlie's pediatrician is Dr. Smith at 123 Main St.", encoding="utf-8")

    # Init and index
    runner.invoke(cli, ["init", str(vault_path), "--name", "health"])
    runner.invoke(cli, ["index", "-v", "health"])

    # Mock ResearchAgent.research to return fast deterministic answer
    def mock_stream(self, question):
        yield "retrieve", {"thought_history": ["Retrieved pediatrician.md"]}
        yield "synthesize", {
            "final_answer": "Dr. Smith is at 123 Main St.",
            "confidence": 0.95,
            "sources": ["pediatrician.md"],
            "thought_history": ["Done"],
        }

    monkeypatch.setattr(ResearchAgent, "stream_research", mock_stream)

    # Invoke bare question
    res = runner.invoke(cli, ["-v", "health", "what", "is", "the", "address", "for", "Charlie's", "pediatrician"])
    assert res.exit_code == 0
    assert "123 Main St" in res.output


def test_cli_evidence_flags_and_json(cli_env, monkeypatch):
    runner, tmp_path = cli_env
    vault_path = tmp_path / "legal_docs"
    vault_path.mkdir()

    # Add doc
    doc = vault_path / "contract.md"
    doc.write_text("# Contract\n\nNotice must be given within 14 days.", encoding="utf-8")

    runner.invoke(cli, ["init", str(vault_path), "--name", "legal"])
    runner.invoke(cli, ["index", "-v", "legal"])

    mock_evidence = [
        {
            "breadcrumbs": "Contract > Section 1",
            "file_path": "contract.md",
            "score": 0.88,
            "content": "Notice must be given within 14 days.",
        }
    ]

    def mock_research(self, question, evidence_only=False):
        return {
            "question": question,
            "evidence": mock_evidence,
            "final_answer": "Notice must be given in 14 days." if not evidence_only else "",
            "confidence": 0.9,
            "sources": ["contract.md"],
        }

    def mock_stream(self, question, evidence_only=False):
        yield "retrieve", {"evidence": mock_evidence, "thought_history": ["Found 1 doc"]}
        if not evidence_only:
            yield "synthesize", {
                "final_answer": "Notice must be given in 14 days.",
                "confidence": 0.9,
                "sources": ["contract.md"],
                "evidence": mock_evidence,
            }

    monkeypatch.setattr(ResearchAgent, "research", mock_research)
    monkeypatch.setattr(ResearchAgent, "stream_research", mock_stream)

    # 1. Test -E (evidence-only)
    res_e = runner.invoke(cli, ["-v", "legal", "what is the notice deadline", "-E"])
    assert res_e.exit_code == 0
    assert "GROUNDING EVIDENCE DOSSIER" in res_e.output
    assert "Notice must be given within 14 days." in res_e.output
    assert "Notice must be given in 14 days." not in res_e.output

    # 2. Test -e (with-evidence)
    res_both = runner.invoke(cli, ["-v", "legal", "what is the notice deadline", "-e"])
    assert res_both.exit_code == 0
    assert "Notice must be given in 14 days." in res_both.output
    assert "GROUNDING EVIDENCE DOSSIER" in res_both.output

    # 3. Test -E --json
    import json
    res_json_e = runner.invoke(cli, ["-v", "legal", "what is the notice deadline", "-E", "--json"])
    assert res_json_e.exit_code == 0
    data_e = json.loads(res_json_e.output)
    assert "chunks" in data_e
    assert data_e["evidence_count"] == 1
    assert data_e["chunks"][0]["file_path"] == "contract.md"

    # 4. Test -e --json
    res_json_both = runner.invoke(cli, ["-v", "legal", "what is the notice deadline", "-e", "--json"])
    assert res_json_both.exit_code == 0
    data_both = json.loads(res_json_both.output)
    assert "final_answer" in data_both
    assert "evidence" in data_both
    assert len(data_both["evidence"]) == 1
