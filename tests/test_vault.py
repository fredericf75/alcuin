"""Unit tests for Alcuin Vault and VaultRegistry."""

import os
from pathlib import Path
import pytest
from alcuin.storage.vault import Vault, VaultConfig, VaultRegistry


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setenv("ALCUIN_CONFIG_DIR", str(cfg_dir))
    return cfg_dir


def test_vault_init_and_load(tmp_path, temp_config_dir):
    vault_dir = tmp_path / "my_docs"
    vault = Vault.init(vault_dir, name="test-vault", description="Test Description")

    assert vault.name == "test-vault"
    assert (vault_dir / ".alcuin").is_dir()
    assert (vault_dir / ".alcuin" / "config.toml").is_file()
    assert (vault_dir / ".alcuin" / "manifest.json").is_file()

    # Registry check
    vaults = VaultRegistry.list_vaults()
    assert "test-vault" in vaults
    assert vaults["test-vault"] == vault_dir.resolve()

    # Load check
    loaded = Vault.load(vault_dir)
    assert loaded.name == "test-vault"
    assert loaded.config.description == "Test Description"
    assert loaded.config.max_hops == 3


def test_vault_overrides(tmp_path, temp_config_dir):
    vault_dir = tmp_path / "research"
    vault = Vault.init(vault_dir, name="research")

    # Apply overrides
    loaded = Vault.load(
        vault_dir,
        overrides={
            "max_hops": 5,
            "model_name": "gemini-2.5-flash",
            "model_endpoint": "https://api.example.com",
            "temperature": 0.7,
        },
    )
    assert loaded.config.max_hops == 5
    assert loaded.config.model_name == "gemini-2.5-flash"
    assert loaded.config.model_endpoint == "https://api.example.com"
    assert loaded.config.temperature == 0.7


def test_vault_resolution_walkup(tmp_path, temp_config_dir):
    vault_dir = tmp_path / "workspace"
    sub_dir = vault_dir / "src" / "deep" / "folder"
    sub_dir.mkdir(parents=True)

    Vault.init(vault_dir, name="workspace-vault")

    # Resolve from deep subfolder without target
    resolved = Vault.resolve(cwd=sub_dir)
    assert resolved is not None
    assert resolved.root_path == vault_dir.resolve()
    assert resolved.name == "workspace-vault"


def test_vault_resolution_by_name_and_path(tmp_path, temp_config_dir):
    vault_dir = tmp_path / "knowledge"
    Vault.init(vault_dir, name="knowledge-base")

    # Resolve by registered name
    by_name = Vault.resolve(target="knowledge-base")
    assert by_name is not None
    assert by_name.root_path == vault_dir.resolve()

    # Resolve by path
    by_path = Vault.resolve(target=str(vault_dir))
    assert by_path is not None
    assert by_path.root_path == vault_dir.resolve()


def test_vault_unregister(tmp_path, temp_config_dir):
    vault_dir = tmp_path / "temp_vault"
    Vault.init(vault_dir, name="temp")
    assert "temp" in VaultRegistry.list_vaults()

    assert VaultRegistry.unregister("temp") is True
    assert "temp" not in VaultRegistry.list_vaults()
    # Directory still intact
    assert (vault_dir / ".alcuin").is_dir()
