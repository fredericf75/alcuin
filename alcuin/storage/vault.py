"""
alcuin.storage.vault - In-situ Vault and Cross-Platform Registry Manager

Manages local .alcuin/ metadata directories and system-wide vault registry
across Linux, macOS, and Windows.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import platformdirs


VAULT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def get_global_config_path() -> Path:
    custom = os.environ.get("ALCUIN_CONFIG_DIR")
    base = Path(custom) if custom else Path(platformdirs.user_config_dir("alcuin"))
    return base / "config.toml"


def load_global_config() -> Dict[str, Any]:
    cfg_file = get_global_config_path()
    if cfg_file.exists():
        try:
            with open(cfg_file, "rb") as f:
                return tomllib.load(f)
        except Exception:
            pass
    return {}


@dataclass
class VaultConfig:
    name: str = "default"
    description: str = ""
    max_hops: int = 3
    max_iterations: int = 4
    model_endpoint: str = ""
    model_name: str = ""
    temperature: float = 0.2
    embedding_model: str = "nomic-embed-text"
    fts_weight: float = 0.3
    vector_weight: float = 0.7
    periodic_indexing: bool = True
    index_interval_seconds: int = 300
    file_extensions: List[str] = field(
        default_factory=lambda: [".pdf", ".docx", ".md", ".txt"]
    )
    exclude_patterns: List[str] = field(
        default_factory=lambda: [".git/*", "node_modules/*", "tmp/*", ".alcuin/*"]
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VaultConfig:
        global_cfg = load_global_config()
        g_agent = global_cfg.get("agent", {})
        g_storage = global_cfg.get("storage", {})

        vault_sec = data.get("vault", {})
        agent_sec = data.get("agent", {})
        storage_sec = data.get("storage", {})
        ingest_sec = data.get("ingestion", {})

        def_endpoint = os.environ.get("ALCUIN_ENDPOINT") or g_agent.get("model_endpoint") or "http://localhost:11434"
        def_model = os.environ.get("ALCUIN_MODEL") or g_agent.get("model_name") or ""
        def_embed = g_storage.get("embedding_model") or "nomic-embed-text"

        return cls(
            name=vault_sec.get("name", "default"),
            description=vault_sec.get("description", ""),
            max_hops=int(agent_sec.get("max_hops", g_agent.get("max_hops", 3))),
            max_iterations=int(agent_sec.get("max_iterations", g_agent.get("max_iterations", 4))),
            model_endpoint=agent_sec.get("model_endpoint", def_endpoint),
            model_name=agent_sec.get("model_name", def_model),
            temperature=float(agent_sec.get("temperature", g_agent.get("temperature", 0.2))),
            embedding_model=storage_sec.get("embedding_model", def_embed),
            fts_weight=float(storage_sec.get("fts_weight", 0.3)),
            vector_weight=float(storage_sec.get("vector_weight", 0.7)),
            periodic_indexing=bool(ingest_sec.get("periodic_indexing", True)),
            index_interval_seconds=int(ingest_sec.get("index_interval_seconds", 300)),
            file_extensions=list(
                ingest_sec.get("file_extensions", [".pdf", ".docx", ".md", ".txt"])
            ),
            exclude_patterns=list(
                ingest_sec.get(
                    "exclude_patterns",
                    [".git/*", "node_modules/*", "tmp/*", ".alcuin/*"],
                )
            ),
        )

    def apply_overrides(self, overrides: Dict[str, Any]) -> VaultConfig:
        """Apply non-None overrides (e.g. from CLI flags or env vars)."""
        valid_fields = set(self.__dataclass_fields__.keys())
        for k, v in overrides.items():
            if v is not None and k in valid_fields:
                field_type = type(getattr(self, k))
                if field_type is int:
                    setattr(self, k, int(v))
                elif field_type is float:
                    setattr(self, k, float(v))
                elif field_type is bool:
                    setattr(self, k, bool(v))
                else:
                    setattr(self, k, v)
        return self

    def to_toml(self) -> str:
        exts = ", ".join(f'"{ext}"' for ext in self.file_extensions)
        excludes = ", ".join(f'"{pat}"' for pat in self.exclude_patterns)
        
        agent_lines = [
            f"max_hops = {self.max_hops}",
            f"max_iterations = {self.max_iterations}",
            f"temperature = {self.temperature}",
        ]
        if self.model_name:
            agent_lines.append(f'model_name = "{self.model_name}"')
        if self.model_endpoint:
            agent_lines.append(f'model_endpoint = "{self.model_endpoint}"')
        agent_block = "\n".join(agent_lines)

        return f"""# Alcuin Vault Configuration

[vault]
name = "{self.name}"
description = "{self.description}"

[agent]
{agent_block}

[storage]
embedding_model = "{self.embedding_model}"
fts_weight = {self.fts_weight}
vector_weight = {self.vector_weight}

[ingestion]
periodic_indexing = {"true" if self.periodic_indexing else "false"}
index_interval_seconds = {self.index_interval_seconds}
file_extensions = [{exts}]
exclude_patterns = [{excludes}]
"""


class VaultRegistry:
    """Manages system-wide vault registry in user config directory."""

    @staticmethod
    def get_config_dir() -> Path:
        custom = os.environ.get("ALCUIN_CONFIG_DIR")
        if custom:
            path = Path(custom)
        else:
            path = Path(platformdirs.user_config_dir("alcuin"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_registry_path(cls) -> Path:
        return cls.get_config_dir() / "vaults.json"

    @classmethod
    def load_registry(cls) -> Dict[str, Any]:
        reg_file = cls.get_registry_path()
        if not reg_file.exists():
            return {"vaults": {}, "default_vault": None}
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {"vaults": {}, "default_vault": None}
                data.setdefault("vaults", {})
                return data
        except Exception:
            return {"vaults": {}, "default_vault": None}

    @classmethod
    def save_registry(cls, data: Dict[str, Any]) -> None:
        reg_file = cls.get_registry_path()
        with open(reg_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def list_vaults(cls) -> Dict[str, Path]:
        reg = cls.load_registry()
        result: Dict[str, Path] = {}
        for name, path_str in reg.get("vaults", {}).items():
            result[name] = Path(path_str).resolve()
        return result

    @classmethod
    def register(cls, name: str, path: Path, set_default: bool = False) -> None:
        name = name.strip()
        if not VAULT_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid vault name '{name}'. Must contain only alphanumeric, dash, dot, or underscore."
            )
        abs_path = path.resolve()
        reg = cls.load_registry()
        reg["vaults"][name] = str(abs_path)
        if set_default or not reg.get("default_vault"):
            reg["default_vault"] = name
        cls.save_registry(reg)

    @classmethod
    def unregister(cls, name: str) -> bool:
        reg = cls.load_registry()
        if name in reg.get("vaults", {}):
            del reg["vaults"][name]
            if reg.get("default_vault") == name:
                reg["default_vault"] = next(iter(reg["vaults"].keys()), None)
            cls.save_registry(reg)
            return True
        return False

    @classmethod
    def get_vault_path(cls, name: str) -> Optional[Path]:
        vaults = cls.list_vaults()
        return vaults.get(name)


class Vault:
    """Represents a localized in-situ Alcuin document vault."""

    def __init__(self, root_path: Path, config: VaultConfig):
        self.root_path = root_path.resolve()
        self.alcuin_dir = self.root_path / ".alcuin"
        self.config_path = self.alcuin_dir / "config.toml"
        self.manifest_path = self.alcuin_dir / "manifest.json"
        self.lancedb_dir = self.alcuin_dir / ".lancedb"
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @classmethod
    def init(
        cls,
        path: Path | str,
        name: Optional[str] = None,
        description: str = "",
        register: bool = True,
    ) -> Vault:
        """Initialize a new in-situ vault at target directory."""
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        alcuin_dir = root / ".alcuin"
        alcuin_dir.mkdir(parents=True, exist_ok=True)

        vault_name = name or root.name
        if not VAULT_NAME_PATTERN.match(vault_name):
            vault_name = "default"

        config_path = alcuin_dir / "config.toml"
        if not config_path.exists():
            cfg = VaultConfig(name=vault_name, description=description)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(cfg.to_toml())
        else:
            cfg = cls.read_config(config_path)

        manifest_path = alcuin_dir / "manifest.json"
        if not manifest_path.exists():
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"files": {}, "wikilinks": {}}, f, indent=2)

        vault = cls(root, cfg)
        if register:
            VaultRegistry.register(vault_name, root)

        return vault

    @classmethod
    def read_config(cls, config_path: Path) -> VaultConfig:
        if not config_path.exists():
            return VaultConfig()
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return VaultConfig.from_dict(data)

    @classmethod
    def load(
        cls, path: Path | str, overrides: Optional[Dict[str, Any]] = None
    ) -> Vault:
        """Load an existing in-situ vault."""
        root = Path(path).resolve()
        alcuin_dir = root / ".alcuin"
        if not alcuin_dir.is_dir():
            raise FileNotFoundError(
                f"Directory '{root}' is not an Alcuin vault (missing .alcuin/ directory)."
            )
        config_path = alcuin_dir / "config.toml"
        cfg = cls.read_config(config_path)
        if overrides:
            cfg.apply_overrides(overrides)
        return cls(root, cfg)

    @classmethod
    def find_parent_vault(cls, start: Optional[Path | str] = None) -> Optional[Path]:
        """Walk up from start path looking for .alcuin/ directory."""
        current = (Path(start) if start else Path.cwd()).resolve()
        for p in [current, *current.parents]:
            if (p / ".alcuin").is_dir():
                return p
        return None

    @classmethod
    def resolve(
        cls,
        target: Optional[str] = None,
        cwd: Optional[Path | str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[Vault]:
        """Resolve a vault by target name/path, or walk-up from cwd."""
        if target:
            # Check direct path
            p = Path(target)
            if p.exists() and (p / ".alcuin").is_dir():
                return cls.load(p, overrides)
            # Check registry by name
            reg_path = VaultRegistry.get_vault_path(target)
            if reg_path and (reg_path / ".alcuin").is_dir():
                return cls.load(reg_path, overrides)
            raise ValueError(
                f"Could not resolve vault '{target}' as directory path or registered vault name."
            )

        # Walk-up from cwd
        parent_vault = cls.find_parent_vault(cwd)
        if parent_vault:
            return cls.load(parent_vault, overrides)

        return None
