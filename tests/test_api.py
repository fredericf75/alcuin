"""Unit tests for FastAPI endpoints and Web dashboard."""

import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from alcuin.api.server import app
from alcuin.storage.vault import Vault, VaultRegistry


@pytest.fixture
def api_test_env(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "global_config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))

    vault_dir = tmp_path / "web_vault"
    vault = Vault.init(vault_dir, name="web-vault", description="Web test vault")
    doc = vault_dir / "sample.md"
    doc.write_text("# Web Test\n\nSample document content.", encoding="utf-8")

    client = TestClient(app)
    return client, vault


def test_health_and_index_page(api_test_env):
    client, vault = api_test_env
    # Health check
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # HTML Dashboard
    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "Alcuin Knowledge Engine" in html_resp.text


def test_vault_management_api(api_test_env):
    client, vault = api_test_env

    # List vaults
    list_resp = client.get("/api/v1/vaults")
    assert list_resp.status_code == 200
    vaults = list_resp.json()["vaults"]
    assert any(v["name"] == "web-vault" for v in vaults)

    # Trigger indexing
    idx_resp = client.post("/api/v1/vaults/web-vault/index")
    assert idx_resp.status_code == 200
    assert "sample.md" in idx_resp.json()["added"]

    # Get vault details
    detail_resp = client.get("/api/v1/vaults/web-vault")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["file_count"] == 1
    assert detail_resp.json()["config"]["max_hops"] == 3

    # Update settings
    put_resp = client.put(
        "/api/v1/vaults/web-vault/settings",
        json={"max_hops": 5, "model_name": "gemma2:9b", "index_interval_seconds": 60},
    )
    assert put_resp.status_code == 200

    # Verify updated settings
    re_detail = client.get("/api/v1/vaults/web-vault")
    assert re_detail.json()["config"]["max_hops"] == 5
    assert re_detail.json()["config"]["model_name"] == "gemma2:9b"
    assert re_detail.json()["config"]["index_interval_seconds"] == 60


def test_research_api_sse(api_test_env, monkeypatch):
    client, vault = api_test_env
    from alcuin.agent.graph import ResearchAgent

    def mock_stream(self, question):
        yield "retrieve", {"thought_history": ["Found sample.md"]}
        yield "synthesize", {
            "final_answer": "Sample answer.",
            "confidence": 0.9,
            "sources": ["sample.md"],
            "thought_history": ["Done"],
        }

    monkeypatch.setattr(ResearchAgent, "stream_research", mock_stream)

    resp = client.post(
        "/api/v1/research",
        json={"vault": "web-vault", "question": "What is in sample?"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "event: step" in text
    assert "event: result" in text
    assert "Sample answer." in text


def test_models_endpoint(api_test_env, monkeypatch):
    client, vault = api_test_env
    from alcuin.agent.graph import LLMClient

    monkeypatch.setattr(LLMClient, "get_available_models", lambda self: ["llama3.2:3b", "qwen3:8b"])

    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "llama3.2:3b" in data["available_models"]
    assert "qwen3:8b" in data["available_models"]


def test_vault_lifecycle_and_prune_api(api_test_env, tmp_path):
    client, vault = api_test_env

    # 1. Init new vault via API
    new_vault_dir = tmp_path / "api_created_vault"
    init_resp = client.post(
        "/api/v1/vaults/init",
        json={"path": str(new_vault_dir), "name": "api-vault", "description": "API created"},
    )
    assert init_resp.status_code == 200
    assert init_resp.json()["name"] == "api-vault"

    # 2. Set default vault
    def_resp = client.post("/api/v1/vaults/api-vault/default")
    assert def_resp.status_code == 200
    assert def_resp.json()["default_vault"] == "api-vault"

    # 3. Register another folder
    another_dir = tmp_path / "another_folder"
    Vault.init(another_dir, name="another-vault")
    reg_resp = client.post(
        "/api/v1/vaults/register",
        json={"path": str(another_dir), "name": "another-vault"},
    )
    assert reg_resp.status_code == 200

    # 4. Prune stale test entries
    prune_resp = client.post("/api/v1/vaults/prune")
    assert prune_resp.status_code == 200
    assert "removed_count" in prune_resp.json()

    # 5. Delete / unregister
    del_resp = client.delete("/api/v1/vaults/another-vault")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "unregistered"


def test_linter_and_gaps_api(api_test_env):
    client, vault = api_test_env

    # 1. Lint endpoint
    lint_resp = client.get("/api/v1/vaults/web-vault/lint")
    assert lint_resp.status_code == 200
    assert "issues" in lint_resp.json()

    # 2. Fix endpoint
    fix_resp = client.post("/api/v1/vaults/web-vault/lint/fix")
    assert fix_resp.status_code == 200
    assert "fixed_count" in fix_resp.json()

    # 3. Gaps list
    gaps_resp = client.get("/api/v1/vaults/web-vault/gaps")
    assert gaps_resp.status_code == 200
    assert "gaps" in gaps_resp.json()


def test_notes_upload_and_save_finding_api(api_test_env):
    client, vault = api_test_env

    # 1. Create a markdown note directly
    note_resp = client.post(
        "/api/v1/vaults/web-vault/notes",
        json={
            "title": "Town Regulations",
            "filename": "town_regs.md",
            "content": "Section 1: Noise limits are 55 dB.",
            "subfolder": "regulations",
        },
    )
    assert note_resp.status_code == 200
    assert "town_regs.md" in note_resp.json()["file_path"]

    # 2. Upload file
    upload_resp = client.post(
        "/api/v1/vaults/web-vault/upload?filename=memo.txt&target=inbox",
        content=b"Staffing rules memo content",
        headers={"Content-Type": "text/plain"},
    )
    assert upload_resp.status_code == 200
    assert "inbox/memo.txt" in upload_resp.json()["path"]

    # 3. Save finding as note (compounding knowledge)
    save_resp = client.post(
        "/api/v1/vaults/web-vault/save-finding",
        json={
            "query": "What are the noise limits?",
            "answer": "Noise limits are 55 dB according to regulations.",
            "confidence": 0.95,
            "sources": ["town_regs.md"],
        },
    )
    assert save_resp.status_code == 200
    assert "findings/" in save_resp.json()["file_path"]
    created_finding = vault.root_path / save_resp.json()["file_path"]
    assert created_finding.exists()
    assert "[[town_regs.md]]" in created_finding.read_text(encoding="utf-8")
