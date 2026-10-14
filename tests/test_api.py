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
