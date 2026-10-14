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
