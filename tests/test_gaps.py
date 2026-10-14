"""Unit tests for Knowledge Gap Tracking and Unanswered Query Logging."""

import json
from pathlib import Path
from click.testing import CliRunner
import pytest

from alcuin.agent.graph import ResearchAgent
from alcuin.cli.main import cli
from alcuin.storage.engine import VaultStorage
from alcuin.storage.gaps import KnowledgeGapTracker, normalize_query_key, suggest_title_from_query
from alcuin.storage.linter import LintCategory, VaultLinter
from alcuin.storage.vault import Vault


@pytest.fixture
def gap_vault(tmp_path, monkeypatch):
    """Fixture providing an isolated vault and global config for gaps testing."""
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    v_dir = tmp_path / "gap_vault"
    v_dir.mkdir()

    # Note about dentist
    note = v_dir / "Charlie.md"
    note.write_text("# Charlie\nCharlie is 8 years old.", encoding="utf-8")

    vault = Vault.init(v_dir, name="gap_vault")
    storage = VaultStorage(vault)
    storage.index()
    return vault, storage, v_dir


def test_normalize_and_suggest_title():
    q = "what is Charlie's dentist policy number?"
    norm = normalize_query_key(q)
    assert norm == "what is charlies dentist policy number"

    title = suggest_title_from_query(q)
    assert title == "Charlie's Dentist Policy Number"

    q2 = "where is the spare car key?"
    assert suggest_title_from_query(q2) == "The Spare Car Key"


def test_record_unanswered_query_and_deduplication(gap_vault):
    vault, storage, v_dir = gap_vault
    tracker = KnowledgeGapTracker(vault)

    # 1. Record unanswered query (0 evidence)
    gap1 = tracker.record_query(
        query="what is Charlie's dentist policy number?",
        confidence=0.0,
        evidence_count=0,
        answer="No relevant documents or evidence found in vault.",
    )
    assert gap1 is not None
    assert gap1.ask_count == 1
    assert gap1.status == "unresolved"
    assert (vault.alcuin_dir / "knowledge_gaps.json").is_file()

    # 2. Re-ask same question with different punctuation / casing -> deduplicate
    gap2 = tracker.record_query(
        query="What is Charlies dentist policy number",
        confidence=0.1,
        evidence_count=1,
        answer="The information is not found in the vault.",
    )
    assert gap2 is not None
    assert gap2.id == gap1.id
    assert gap2.ask_count == 2
    assert gap2.status == "unresolved"

    # Verify listing
    unresolved = tracker.list_gaps(include_resolved=False)
    assert len(unresolved) == 1
    assert unresolved[0].ask_count == 2


def test_auto_resolution_on_high_confidence(gap_vault):
    vault, storage, v_dir = gap_vault
    tracker = KnowledgeGapTracker(vault)

    # Log gap
    tracker.record_query(
        query="who is Charlie's pediatrician?",
        confidence=0.0,
        evidence_count=0,
        answer="No relevant documents.",
    )
    assert len(tracker.list_gaps(include_resolved=False)) == 1

    # Later: successfully answered
    tracker.record_query(
        query="Who is Charlie's pediatrician?",
        confidence=0.85,
        evidence_count=2,
        answer="Dr. Smith at 123 Main St.",
    )

    # Unresolved list is now empty; resolved list contains it
    assert len(tracker.list_gaps(include_resolved=False)) == 0
    all_gaps = tracker.list_gaps(include_resolved=True)
    assert len(all_gaps) == 1
    assert all_gaps[0].status == "resolved"
    assert all_gaps[0].resolved_at is not None


def test_linter_detects_gap_and_creates_stub(gap_vault):
    vault, storage, v_dir = gap_vault
    tracker = KnowledgeGapTracker(vault)

    tracker.record_query(
        query="what is our home insurance policy number?",
        confidence=0.0,
        evidence_count=0,
        answer="No relevant documents.",
    )

    linter = VaultLinter(vault)
    report = linter.lint()

    gap_issues = [i for i in report.issues if i.category == LintCategory.KNOWLEDGE_GAP]
    assert len(gap_issues) == 1
    issue = gap_issues[0]
    assert "home insurance" in issue.message.lower()
    assert issue.fixable is True

    # Remediate with stub creation
    fix_res = linter.fix(create_stubs=True, dry_run=False)
    assert len(fix_res.errors) == 0

    # Verify stub file created
    stubs = list(v_dir.glob("*Home_Insurance*.md")) + list(v_dir.glob("*Home Insurance*.md"))
    assert len(stubs) == 1
    stub_content = stubs[0].read_text(encoding="utf-8")
    assert "Knowledge gap recorded by Alcuin" in stub_content
    assert "home insurance policy number" in stub_content

    # Re-linting should show stub is now created
    clean_report = linter.lint()
    g_issues = [i for i in clean_report.issues if i.category == LintCategory.KNOWLEDGE_GAP]
    assert len(g_issues) == 1
    assert g_issues[0].fixable is False  # Stub is already created


def test_cli_gaps_suite(gap_vault):
    vault, storage, v_dir = gap_vault
    tracker = KnowledgeGapTracker(vault)

    tracker.record_query(
        query="what is Charlie's dentist phone number?",
        confidence=0.0,
        evidence_count=0,
        answer="No relevant documents.",
    )

    runner = CliRunner(env={"COLUMNS": "160"})

    # 1. List gaps
    res = runner.invoke(cli, ["gaps", "list", "-v", "gap_vault"])
    assert res.exit_code == 0
    assert "dentist phone number" in res.output
    assert "Unresolved" in res.output

    # 2. Resolve gap
    res_res = runner.invoke(cli, ["gaps", "resolve", "dentist", "-v", "gap_vault"])
    assert res_res.exit_code == 0
    assert "Resolved knowledge gap" in res_res.output

    # 3. List should show empty (unless --all)
    res_empty = runner.invoke(cli, ["gaps", "list", "-v", "gap_vault"])
    assert "No knowledge gaps found" in res_empty.output

    res_all = runner.invoke(cli, ["gaps", "list", "-v", "gap_vault", "--all"])
    assert "Resolved" in res_all.output

    # 4. Clear resolved
    res_clear = runner.invoke(cli, ["gaps", "clear", "-v", "gap_vault"])
    assert res_clear.exit_code == 0
    assert "Cleared 1 gap record" in res_clear.output


def test_research_agent_records_gap_integration(gap_vault, monkeypatch):
    vault, storage, v_dir = gap_vault
    from alcuin.agent.graph import LLMClient

    # Mock LLM to explicitly report information not found
    def mock_chat(self, sys_prompt, user_prompt):
        return "The passport expiry date is not found in the vault documents."

    monkeypatch.setattr(LLMClient, "chat", mock_chat)

    agent = ResearchAgent(vault)
    result = agent.research("what is Charlie's passport expiry date?")
    assert result["confidence"] == 0.0

    # Check that gap was automatically logged
    tracker = KnowledgeGapTracker(vault)
    gaps = tracker.list_gaps(include_resolved=False)
    assert len(gaps) == 1
    assert "passport expiry" in gaps[0].query
