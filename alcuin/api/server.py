"""
alcuin.api.server - FastAPI Server with SSE Research Streaming & Web Dashboard
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from alcuin.agent.graph import LLMClient, ModelNotFoundError, ResearchAgent
from alcuin.api.daemon import PeriodicIndexer
from alcuin.storage.engine import VaultStorage
from alcuin.storage.gaps import KnowledgeGapTracker
from alcuin.storage.ingest import InboxIngestionEngine
from alcuin.storage.linter import VaultLinter
from alcuin.storage.vault import Vault, VaultConfig, VaultRegistry

periodic_indexer = PeriodicIndexer(check_interval=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch background periodic indexing
    await periodic_indexer.start()
    yield
    # Shutdown: cleanly cancel background tasks
    await periodic_indexer.stop()


app = FastAPI(title="Alcuin Knowledge Engine", version="0.2.0", lifespan=lifespan)


class ResearchRequest(BaseModel):
    vault: Optional[str] = None
    question: str
    max_hops: Optional[int] = None
    max_iterations: Optional[int] = None
    model_name: Optional[str] = None
    model_endpoint: Optional[str] = None
    temperature: Optional[float] = None


class VaultSettingsUpdate(BaseModel):
    max_hops: Optional[int] = None
    max_iterations: Optional[int] = None
    model_endpoint: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    periodic_indexing: Optional[bool] = None
    index_interval_seconds: Optional[int] = None


class VaultInitRequest(BaseModel):
    path: str
    name: Optional[str] = None
    description: Optional[str] = ""


class VaultRegisterRequest(BaseModel):
    path: str
    name: Optional[str] = None


class CreateNoteRequest(BaseModel):
    title: str
    content: str
    filename: Optional[str] = None
    subfolder: Optional[str] = ""


class SaveFindingRequest(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: Optional[List[str]] = None


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/api/v1/models")
async def list_models(endpoint: Optional[str] = None):
    client = LLMClient(endpoint=endpoint)
    models = client.get_available_models()
    return {
        "endpoint": client.endpoint,
        "active_model": client.model,
        "available_models": models,
    }


@app.get("/api/v1/vaults")
async def list_vaults():
    reg = VaultRegistry.load_registry()
    default_vault = reg.get("default_vault")
    vaults = VaultRegistry.list_vaults()
    result = []
    for name, path in vaults.items():
        is_valid = (path / ".alcuin").is_dir()
        chunk_count = 0
        if is_valid:
            try:
                v = Vault.load(path)
                storage = VaultStorage(v)
                chunk_count = storage.table.count_rows()
            except Exception:
                pass
        result.append(
            {
                "name": name,
                "path": str(path),
                "valid": is_valid,
                "chunks": chunk_count,
                "is_default": (name == default_vault),
            }
        )
    return {"vaults": result, "default_vault": default_vault}


@app.post("/api/v1/vaults/init")
async def init_vault(req: VaultInitRequest):
    target_path = Path(req.path).resolve()
    vault = Vault.init(target_path, name=req.name, description=req.description or "")
    return {"status": "created", "name": vault.name, "path": str(vault.root_path)}


@app.post("/api/v1/vaults/register")
async def register_vault(req: VaultRegisterRequest):
    p = Path(req.path).resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Directory '{p}' does not exist.")
    name = req.name or p.name
    VaultRegistry.register(name, p)
    return {"status": "registered", "name": name, "path": str(p)}


@app.delete("/api/v1/vaults/{name}")
async def unregister_vault(name: str):
    if VaultRegistry.unregister(name):
        return {"status": "unregistered", "name": name}
    raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")


@app.post("/api/v1/vaults/prune")
async def prune_vaults():
    count, removed = VaultRegistry.prune()
    return {"status": "pruned", "removed_count": count, "removed_vaults": removed}


@app.post("/api/v1/vaults/{name}/default")
async def set_default_vault(name: str):
    if VaultRegistry.set_default(name):
        return {"status": "default_updated", "default_vault": name}
    raise HTTPException(status_code=404, detail=f"Vault '{name}' not found in registry.")


@app.get("/api/v1/vaults/{name}")
async def get_vault(name: str):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)
    storage = VaultStorage(vault)
    manifest = storage.load_manifest()

    return {
        "name": vault.name,
        "path": str(vault.root_path),
        "config": {
            "max_hops": vault.config.max_hops,
            "max_iterations": vault.config.max_iterations,
            "model_endpoint": vault.config.model_endpoint,
            "model_name": vault.config.model_name,
            "temperature": vault.config.temperature,
            "embedding_model": vault.config.embedding_model,
            "periodic_indexing": vault.config.periodic_indexing,
            "index_interval_seconds": vault.config.index_interval_seconds,
            "file_extensions": vault.config.file_extensions,
        },
        "file_count": len(manifest.get("files", {})),
        "chunk_count": storage.table.count_rows(),
        "files": manifest.get("files", {}),
    }


@app.put("/api/v1/vaults/{name}/settings")
async def update_vault_settings(name: str, update: VaultSettingsUpdate):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")

    vault = Vault.load(path)
    overrides = update.model_dump(exclude_none=True)
    vault.config.apply_overrides(overrides)

    with open(vault.config_path, "w", encoding="utf-8") as f:
        f.write(vault.config.to_toml())

    return {"status": "updated", "config": overrides}


@app.post("/api/v1/vaults/{name}/index")
async def trigger_vault_index(name: str, force: bool = False):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")

    vault = Vault.load(path)
    storage = VaultStorage(vault)
    report = storage.index(force=force)

    return {
        "status": "indexed",
        "added": report.added_files,
        "modified": report.modified_files,
        "deleted": report.deleted_files,
        "unchanged": len(report.unchanged_files),
        "total_chunks": report.total_chunks_indexed,
        "duration_seconds": report.duration_seconds,
    }


@app.get("/api/v1/vaults/{name}/lint")
async def lint_vault(name: str):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)
    linter = VaultLinter(vault)
    report = linter.lint()
    return report.to_dict()


@app.post("/api/v1/vaults/{name}/lint/fix")
async def fix_vault_lint(name: str):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)
    linter = VaultLinter(vault)
    fix_result = linter.fix()
    data = fix_result.to_dict()
    data["fixed_count"] = data.get("repaired_count", len(fix_result.repaired))
    return data


@app.get("/api/v1/vaults/{name}/gaps")
async def get_gaps(name: str, include_resolved: bool = True):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)
    tracker = KnowledgeGapTracker(vault)
    gaps = tracker.list_gaps(include_resolved=include_resolved)
    return {"vault": name, "total_gaps": len(gaps), "gaps": [g.to_dict() for g in gaps]}


@app.post("/api/v1/vaults/{name}/gaps/{gap_id}/resolve")
async def resolve_gap(name: str, gap_id: str):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)
    tracker = KnowledgeGapTracker(vault)
    if tracker.resolve_gap(gap_id):
        return {"status": "resolved", "gap_id": gap_id}
    raise HTTPException(status_code=404, detail=f"Gap '{gap_id}' not found.")


@app.post("/api/v1/vaults/{name}/notes")
async def create_note(name: str, req: CreateNoteRequest):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)

    sub = (req.subfolder or "").strip().strip("/")
    target_dir = vault.root_path / sub if sub else vault.root_path
    target_dir.mkdir(parents=True, exist_ok=True)

    fname = req.filename or f"{re.sub(r'[^a-zA-Z0-9_-]', '_', req.title.lower())}.md"
    if not fname.endswith(".md"):
        fname += ".md"

    note_path = (target_dir / fname).resolve()
    if not str(note_path).startswith(str(vault.root_path)):
        raise HTTPException(status_code=400, detail="Invalid path traversal outside vault root.")

    content = req.content
    if not content.startswith("# "):
        content = f"# {req.title}\n\n{content}"

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(content)

    storage = VaultStorage(vault)
    rep = storage.index()
    return {
        "status": "created",
        "file_path": note_path.relative_to(vault.root_path).as_posix(),
        "chunks_indexed": rep.total_chunks_indexed,
    }


@app.post("/api/v1/vaults/{name}/upload")
async def upload_file(
    name: str,
    request: Request,
    filename: Optional[str] = None,
    target: str = "inbox",
):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)

    target_folder = vault.root_path / "inbox" if target == "inbox" else vault.root_path
    target_folder.mkdir(parents=True, exist_ok=True)

    fname = (
        filename
        or request.headers.get("X-Filename")
        or f"upload_{int(datetime.datetime.now().timestamp())}.txt"
    )
    dest = target_folder / Path(fname).name

    content = await request.body()
    with open(dest, "wb") as f:
        f.write(content)

    chunks = 0
    if target != "inbox" and dest.suffix in vault.config.file_extensions:
        storage = VaultStorage(vault)
        rep = storage.index()
        chunks = rep.total_chunks_indexed

    return {
        "status": "uploaded",
        "filename": dest.name,
        "path": dest.relative_to(vault.root_path).as_posix(),
        "chunks_indexed": chunks,
    }


@app.post("/api/v1/vaults/{name}/ingest")
async def ingest_inbox(name: str):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)
    engine = InboxIngestionEngine(vault)
    res = engine.ingest_inbox()
    storage = VaultStorage(vault)
    rep = storage.index()
    return {
        "status": "ingested",
        "ingested_notes": res.ingested_notes,
        "skipped_duplicates": res.skipped_duplicates,
        "errors": res.errors,
        "chunks_indexed": rep.total_chunks_indexed,
    }


@app.post("/api/v1/vaults/{name}/save-finding")
async def save_finding(name: str, req: SaveFindingRequest):
    path = VaultRegistry.get_vault_path(name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{name}' not found.")
    vault = Vault.load(path)

    findings_dir = vault.root_path / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", req.query.lower()).strip("_")[:40]
    fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{slug}.md"
    dest = findings_dir / fname

    src_links = (
        "\n".join(f"- [[{s}]]" for s in (req.sources or []))
        if req.sources
        else "- (None)"
    )

    content = (
        f"---\n"
        f"title: \"{req.query}\"\n"
        f"type: research-finding\n"
        f"date: {date_str}\n"
        f"confidence: {req.confidence:.2f}\n"
        f"---\n\n"
        f"# Finding: {req.query}\n\n"
        f"{req.answer}\n\n"
        f"## Admitted Grounding Sources\n"
        f"{src_links}\n"
    )
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)

    storage = VaultStorage(vault)
    rep = storage.index()
    return {
        "status": "saved",
        "file_path": dest.relative_to(vault.root_path).as_posix(),
        "chunks_indexed": rep.total_chunks_indexed,
    }


@app.post("/api/v1/research")
async def research(req: ResearchRequest):
    vault_name = req.vault
    if not vault_name:
        vaults = VaultRegistry.list_vaults()
        if not vaults:
            raise HTTPException(status_code=400, detail="No registered vaults found.")
        vault_name = next(iter(vaults.keys()))

    path = VaultRegistry.get_vault_path(vault_name)
    if not path or not (path / ".alcuin").is_dir():
        raise HTTPException(status_code=404, detail=f"Vault '{vault_name}' not found.")

    overrides = {
        "max_hops": req.max_hops,
        "max_iterations": req.max_iterations,
        "model_name": req.model_name,
        "model_endpoint": req.model_endpoint,
        "temperature": req.temperature,
    }
    vault = Vault.load(path, overrides=overrides)
    agent = ResearchAgent(vault)

    def event_generator():
        final_state = {}
        try:
            for node_name, state_update in agent.stream_research(req.question):
                final_state.update(state_update)
                thought = ""
                if "thought_history" in state_update and state_update["thought_history"]:
                    thought = state_update["thought_history"][-1]
                data = json.dumps({"node": node_name, "thought": thought})
                yield f"event: step\ndata: {data}\n\n"

            if not final_state.get("final_answer"):
                final_state = agent.research(req.question)
        except ModelNotFoundError as e:
            err_data = json.dumps(
                {
                    "error": str(e),
                    "model": e.model,
                    "endpoint": e.endpoint,
                    "available_models": e.available_models,
                }
            )
            yield f"event: error\ndata: {err_data}\n\n"
            return

        result_data = json.dumps(
            {
                "answer": final_state.get("final_answer", ""),
                "confidence": final_state.get("confidence", 0.0),
                "sources": final_state.get("sources", []),
                "friction_alert": final_state.get("friction_alert"),
            }
        )
        yield f"event: result\ndata: {result_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Alcuin Knowledge Engine - Cockpit & Operations</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-cyan-500 selection:text-slate-950">
  <div class="max-w-6xl mx-auto p-6 space-y-6">
    <!-- TOP HEADER -->
    <header class="flex flex-wrap justify-between items-center gap-4 pb-6 border-b border-slate-800/80">
      <div class="flex items-center gap-3">
        <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <span class="text-xl font-black text-slate-950">A</span>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Alcuin Knowledge Engine
            <span class="text-[10px] bg-cyan-950/80 text-cyan-400 border border-cyan-800/60 px-2 py-0.5 rounded-full font-mono uppercase">v0.2.0</span>
          </h1>
          <p class="text-xs text-slate-400">Autonomous Agentic Reasoning & Multi-Vault Operations Suite</p>
        </div>
      </div>

      <!-- GLOBAL CONTROLS IN HEADER -->
      <div class="flex flex-wrap items-center gap-2.5 text-xs">
        <div class="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 flex items-center gap-2">
          <span class="text-slate-400">Vault:</span>
          <select id="vaultSelect" class="bg-transparent text-cyan-300 font-medium focus:outline-none cursor-pointer"></select>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 flex items-center gap-2">
          <span class="text-slate-400">Model:</span>
          <select id="modelSelect" class="bg-transparent text-emerald-400 font-mono focus:outline-none cursor-pointer">
            <option value="">(Loading...)</option>
          </select>
        </div>

        <button onclick="syncVault()" id="syncBtn" class="bg-slate-800 hover:bg-slate-700 border border-slate-700/80 hover:border-cyan-500/50 text-slate-200 px-3 py-1.5 rounded-lg transition font-medium flex items-center gap-1.5 shadow-sm">
          <span>⚡</span> Re-Index
        </button>
      </div>
    </header>

    <!-- NAVIGATION TABS -->
    <nav class="flex gap-2 border-b border-slate-800/80 text-sm">
      <button onclick="switchTab('cockpit')" id="tabBtn-cockpit" class="pb-3 px-4 border-b-2 border-cyan-400 text-cyan-400 font-semibold flex items-center gap-2 transition">
        <span>🔍</span> Research Cockpit
      </button>
      <button onclick="switchTab('notes')" id="tabBtn-notes" class="pb-3 px-4 border-b-2 border-transparent text-slate-400 hover:text-white font-medium flex items-center gap-2 transition">
        <span>📝</span> Notes & Intake
      </button>
      <button onclick="switchTab('maintenance')" id="tabBtn-maintenance" class="pb-3 px-4 border-b-2 border-transparent text-slate-400 hover:text-white font-medium flex items-center gap-2 transition">
        <span>🛠️</span> Health & Maintenance
      </button>
      <button onclick="switchTab('registry')" id="tabBtn-registry" class="pb-3 px-4 border-b-2 border-transparent text-slate-400 hover:text-white font-medium flex items-center gap-2 transition">
        <span>🏛️</span> Vault Registry
      </button>
      <button onclick="switchTab('settings')" id="tabBtn-settings" class="pb-3 px-4 border-b-2 border-transparent text-slate-400 hover:text-white font-medium flex items-center gap-2 transition">
        <span>⚙️</span> Settings
      </button>
    </nav>

    <!-- TAB 1: RESEARCH COCKPIT -->
    <main id="tab-cockpit" class="space-y-6">
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-2xl backdrop-blur">
        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Autonomous Research Inquiry</label>
        <div class="flex gap-3">
          <input id="questionInput" type="text" placeholder="e.g. Under Section 22-109, what are the maximum decibel limits between 9 PM and 2 AM?" 
                 class="flex-1 bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/20 text-white placeholder-slate-600 transition" 
                 onkeydown="if(event.key==='Enter') executeResearch()"/>
          <button onclick="executeResearch()" id="researchBtn" class="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold px-6 py-3 rounded-xl text-sm transition shadow-lg shadow-cyan-500/25">
            Research
          </button>
        </div>
      </div>

      <!-- ERROR ALERT -->
      <div id="errorBox" class="hidden bg-red-950/40 border border-red-800/80 rounded-2xl p-5 text-red-200 space-y-2">
        <h3 id="errorTitle" class="text-sm font-bold text-red-400 flex items-center gap-2">
          <span>⚠️</span> Error
        </h3>
        <p id="errorMsg" class="text-xs leading-relaxed"></p>
        <div id="errorAvailModels" class="hidden pt-2 text-xs text-red-300">
          <span class="font-semibold text-white">Available models on endpoint:</span>
          <ul id="errorModelsList" class="list-disc pl-5 mt-1 font-mono text-xs text-slate-300"></ul>
        </div>
      </div>

      <!-- HIGH RETRIEVAL FRICTION ALERT -->
      <div id="frictionBox" class="hidden bg-amber-950/40 border border-amber-800/80 rounded-2xl p-4 text-xs text-amber-200 flex flex-wrap items-center justify-between gap-3 shadow-lg">
        <div class="flex items-center gap-3">
          <span class="text-xl">⚡</span>
          <div>
            <span class="font-bold text-amber-300">High Retrieval Friction Detected</span>
            <p id="frictionMsg" class="text-xs text-amber-200/80 mt-0.5 leading-relaxed"></p>
          </div>
        </div>
        <button onclick="saveCurrentFinding()" class="bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 px-3.5 py-1.5 rounded-lg font-medium transition text-xs flex items-center gap-1.5 shadow-sm">
          <span>💾</span> Save Finding as Note
        </button>
      </div>

      <!-- ANSWER CARD -->
      <div id="answerCard" class="hidden bg-slate-900/90 border border-slate-800/90 rounded-2xl p-6 shadow-2xl space-y-5">
        <div class="flex flex-wrap justify-between items-center gap-3 border-b border-slate-800/80 pb-4">
          <div class="flex items-center gap-3">
            <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">Synthesized Finding</span>
            <span id="confidenceBadge" class="text-xs bg-emerald-950 text-emerald-300 border border-emerald-800/60 px-3 py-0.5 rounded-full font-medium"></span>
          </div>
          <button onclick="saveCurrentFinding()" id="saveFindingBtn" class="bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-xs px-3.5 py-1.5 rounded-lg transition font-medium flex items-center gap-1.5">
            <span>💾</span> Save Finding as Note
          </button>
        </div>

        <div id="answerContent" class="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap font-sans"></div>

        <div id="sourcesBox" class="text-xs text-slate-400 pt-3 border-t border-slate-800/80 flex flex-wrap items-center gap-2">
          <span class="font-semibold text-slate-300">Admitted Grounding Sources:</span>
          <div id="sourcesBadges" class="flex flex-wrap gap-1.5 font-mono text-xs"></div>
        </div>
      </div>

      <!-- PROGRESS FEED: REASONING & TRAVERSAL TRACE (COLLAPSIBLE, COLLAPSED BY DEFAULT) -->
      <details id="progressBox" class="hidden group bg-slate-900/60 border border-slate-800/80 rounded-2xl overflow-hidden transition">
        <summary class="flex items-center justify-between p-5 cursor-pointer list-none select-none hover:bg-slate-800/30 transition [&::-webkit-details-marker]:hidden">
          <div class="flex items-center gap-2.5">
            <span id="pulseIndicator" class="animate-pulse text-cyan-400">●</span>
            <h3 class="text-xs font-bold uppercase tracking-wider text-cyan-400">Reasoning & Traversal Trace</h3>
            <span id="stepCountBadge" class="hidden text-[10px] bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded-full font-mono">0 steps</span>
          </div>
          <div class="flex items-center gap-2 text-slate-400">
            <span id="traversalCounter" class="text-[11px] font-mono"></span>
            <svg class="w-4 h-4 text-slate-400 transition-transform duration-200 group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </summary>
        <div class="p-5 pt-0 border-t border-slate-800/40">
          <ul id="stepsList" class="text-xs text-slate-300 space-y-2 font-mono pt-3"></ul>
        </div>
      </details>
    </main>

    <!-- TAB 2: NOTES & INTAKE -->
    <section id="tab-notes" class="hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- DRAG AND DROP FILE UPLOAD -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div class="flex justify-between items-center">
            <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <span>📥</span> Document Intake
            </h2>
            <button onclick="triggerInboxIngest()" class="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded-lg transition font-medium shadow-sm">
              Process Inbox (OCR/VLM)
            </button>
          </div>
          <p class="text-xs text-slate-400 leading-relaxed">
            Upload files (PDF, DOCX, TXT, MD) into <code class="text-cyan-300">inbox/</code> for automated extraction or directly into the vault.
          </p>

          <div id="dropZone" class="border-2 border-dashed border-slate-700/80 hover:border-cyan-500 rounded-xl p-8 text-center cursor-pointer transition bg-slate-950/40">
            <input type="file" id="fileInput" class="hidden" onchange="handleFileSelect(event)" multiple/>
            <div class="space-y-2">
              <span class="text-3xl">📄</span>
              <p class="text-xs text-slate-300 font-medium">Drag & drop documents here, or <span class="text-cyan-400 underline">browse</span></p>
              <p class="text-[10px] text-slate-500">Target: <code class="text-slate-400">inbox/</code> (automatically ingested)</p>
            </div>
          </div>
          <div id="uploadStatus" class="hidden text-xs font-mono text-emerald-400"></div>
        </div>

        <!-- IN-BROWSER QUICK NOTE CREATOR -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
          <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <span>✏️</span> Create New Vault Note
          </h2>
          <div class="space-y-3">
            <input id="noteTitle" type="text" placeholder="Note Title (e.g. Zoning Exception Policy)" 
                   class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"/>
            <input id="noteSubfolder" type="text" placeholder="Subfolder (optional, e.g. policies)" 
                   class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"/>
            <textarea id="noteContent" rows="6" placeholder="# Content\n\nEnter markdown notes, bullets, [[wikilinks]]..." 
                      class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500"></textarea>
            <button onclick="submitNewNote()" class="w-full bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-bold py-2 rounded-xl text-xs transition shadow-md">
              Save Note & Index
            </button>
          </div>
        </div>
      </div>

      <!-- CATALOG OF INDEXED DOCUMENTS -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
        <div class="flex justify-between items-center">
          <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200">Indexed Document Catalog</h2>
          <span id="catalogCount" class="text-xs text-slate-400 font-mono">0 files</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th class="py-2.5 px-3">File Path</th>
                <th class="py-2.5 px-3">Chunks</th>
                <th class="py-2.5 px-3">Size (bytes)</th>
                <th class="py-2.5 px-3">Last Modified</th>
              </tr>
            </thead>
            <tbody id="catalogTableBody" class="divide-y divide-slate-800/60 font-mono text-[11px]"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 3: HEALTH & MAINTENANCE -->
    <section id="tab-maintenance" class="hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-1">
          <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Total Indexed Chunks</span>
          <div id="healthChunks" class="text-2xl font-bold text-white font-mono">-</div>
        </div>
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-1">
          <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Tracked Knowledge Gaps</span>
          <div id="healthGaps" class="text-2xl font-bold text-amber-400 font-mono">-</div>
        </div>
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-1">
          <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Active Embedding Model</span>
          <div id="healthEmbed" class="text-xs font-mono text-cyan-300 mt-2 truncate">-</div>
        </div>
      </div>

      <!-- LINTER SUITE -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
        <div class="flex flex-wrap justify-between items-center gap-3">
          <div>
            <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <span>🩺</span> Vault Diagnostic Linter
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">Audits wikilink integrity, orphan notes, and frontmatter syntax.</p>
          </div>
          <div class="flex gap-2">
            <button onclick="runLinter()" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs px-3.5 py-2 rounded-xl transition font-medium">
              Run Audit
            </button>
            <button onclick="fixLinter()" class="bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs px-3.5 py-2 rounded-xl transition shadow-md">
              Auto-Fix Issues
            </button>
          </div>
        </div>
        <div id="lintResultsBox" class="space-y-2">
          <div id="lintSummary" class="text-xs text-slate-300 font-medium">Click "Run Audit" to inspect vault health.</div>
          <ul id="lintIssuesList" class="space-y-1.5 text-xs font-mono"></ul>
        </div>
      </div>

      <!-- KNOWLEDGE GAPS TRACKER -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
        <div class="flex justify-between items-center">
          <div>
            <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <span>🎯</span> Knowledge Gaps Tracker
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">Identifies queries where evidence was insufficient or missing.</p>
          </div>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th class="py-2.5 px-3">Gap Query</th>
                <th class="py-2.5 px-3">Confidence</th>
                <th class="py-2.5 px-3">Status</th>
                <th class="py-2.5 px-3">Action</th>
              </tr>
            </thead>
            <tbody id="gapsTableBody" class="divide-y divide-slate-800/60 font-mono text-[11px]"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 4: VAULT REGISTRY -->
    <section id="tab-registry" class="hidden space-y-6">
      <div class="flex flex-wrap justify-between items-center gap-3">
        <div>
          <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200">System Vault Registry</h2>
          <p class="text-xs text-slate-400">Curate registered knowledge repositories across your filesystem.</p>
        </div>
        <button onclick="pruneVaults()" class="bg-amber-600 hover:bg-amber-500 text-slate-950 font-bold text-xs px-4 py-2 rounded-xl transition shadow-md flex items-center gap-1.5">
          <span>🧹</span> Prune Stale Vaults
        </button>
      </div>

      <!-- VAULT GRID -->
      <div id="registryGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>

      <!-- REGISTER & INIT FORMS -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-800/80">
        <!-- REGISTER EXISTING -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-200">Register Existing Folder</h3>
          <input id="regPath" type="text" placeholder="/absolute/path/to/vault" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"/>
          <input id="regName" type="text" placeholder="Vault Name (optional, defaults to dir)" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"/>
          <button onclick="submitRegisterVault()" class="w-full bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold py-2 rounded-xl text-xs transition">
            Register Vault
          </button>
        </div>

        <!-- INIT NEW -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-200">Initialize New Vault</h3>
          <input id="initPath" type="text" placeholder="/path/to/create/new_vault" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"/>
          <input id="initName" type="text" placeholder="Vault Name" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"/>
          <input id="initDesc" type="text" placeholder="Description (optional)" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white"/>
          <button onclick="submitInitVault()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold py-2 rounded-xl text-xs transition">
            Create Vault
          </button>
        </div>
      </div>
    </section>

    <!-- TAB 5: SETTINGS -->
    <section id="tab-settings" class="mt-6 hidden bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-5">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-300">Active Vault Configuration</h2>
      <div class="grid grid-cols-2 gap-4 text-sm">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Max Hops</label>
          <input id="cfgMaxHops" type="number" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Max Iterations</label>
          <input id="cfgMaxIterations" type="number" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Model Endpoint</label>
          <input id="cfgModelEndpoint" type="text" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Model Name</label>
          <input id="cfgModelName" type="text" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Periodic Indexing</label>
          <select id="cfgPeriodicIndexing" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-white text-xs">
            <option value="true">Enabled</option>
            <option value="false">Disabled</option>
          </select>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Interval (Seconds)</label>
          <input id="cfgInterval" type="number" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-white text-xs"/>
        </div>
      </div>
      <button onclick="saveSettings()" class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold px-5 py-2.5 rounded-xl text-xs transition">
        Save Settings
      </button>
    </section>
  </div>

  <script>
    let activeVault = '';
    let latestFinding = null;

    async function loadModels() {
      try {
        const res = await fetch('/api/v1/models');
        const data = await res.json();
        const sel = document.getElementById('modelSelect');
        sel.innerHTML = '';

        if (data.available_models && data.available_models.length > 0) {
          data.available_models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === data.active_model) opt.selected = true;
            sel.appendChild(opt);
          });
        } else {
          const opt = document.createElement('option');
          opt.value = data.active_model || '';
          opt.textContent = data.active_model || 'Default Model';
          sel.appendChild(opt);
        }
      } catch (e) {
        console.error('Failed to load models:', e);
      }
    }

    async function loadVaults() {
      const res = await fetch('/api/v1/vaults');
      const data = await res.json();
      const sel = document.getElementById('vaultSelect');
      sel.innerHTML = '';

      data.vaults.forEach((v, idx) => {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = `${v.name} (${v.chunks} chunks)${v.is_default ? ' ★' : ''}`;
        if (v.is_default || (!activeVault && idx === 0)) {
          opt.selected = true;
          activeVault = v.name;
        }
        sel.appendChild(opt);
      });

      if (!activeVault && data.vaults.length > 0) {
        activeVault = data.vaults[0].name;
      }

      sel.onchange = (e) => {
        activeVault = e.target.value;
        refreshActiveVaultData();
      };

      renderRegistry(data.vaults);
      if (activeVault) refreshActiveVaultData();
    }

    async function refreshActiveVaultData() {
      loadSettings();
      loadCatalog();
      loadHealth();
    }

    async function loadCatalog() {
      if (!activeVault) return;
      try {
        const res = await fetch(`/api/v1/vaults/${activeVault}`);
        const data = await res.json();
        const tbody = document.getElementById('catalogTableBody');
        tbody.innerHTML = '';
        const files = data.files || {};
        const entries = Object.entries(files);
        document.getElementById('catalogCount').innerText = `${entries.length} documents`;

        entries.forEach(([fp, info]) => {
          const tr = document.createElement('tr');
          const date = info.mtime ? new Date(info.mtime * 1000).toLocaleString() : '-';
          tr.innerHTML = `
            <td class="py-2.5 px-3 text-cyan-300">${fp}</td>
            <td class="py-2.5 px-3">${info.chunks || 0}</td>
            <td class="py-2.5 px-3 text-slate-400">${info.size || 0}</td>
            <td class="py-2.5 px-3 text-slate-500">${date}</td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error('Failed to load catalog:', e);
      }
    }

    async function loadHealth() {
      if (!activeVault) return;
      try {
        const resV = await fetch(`/api/v1/vaults/${activeVault}`);
        const dataV = await resV.json();
        document.getElementById('healthChunks').innerText = dataV.chunk_count || 0;
        document.getElementById('healthEmbed').innerText = dataV.config.embedding_model || 'nomic-embed-text';

        const resG = await fetch(`/api/v1/vaults/${activeVault}/gaps`);
        const dataG = await resG.json();
        document.getElementById('healthGaps').innerText = (dataG.gaps || []).filter(g => !g.resolved).length;
        renderGapsTable(dataG.gaps || []);
      } catch (e) {
        console.error('Failed to load health:', e);
      }
    }

    function renderGapsTable(gaps) {
      const tbody = document.getElementById('gapsTableBody');
      tbody.innerHTML = '';
      if (!gaps || gaps.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="py-4 text-center text-slate-500">No knowledge gaps flagged.</td></tr>`;
        return;
      }
      gaps.forEach(g => {
        const tr = document.createElement('tr');
        const st = g.resolved ? '<span class="text-emerald-400">Resolved</span>' : '<span class="text-amber-400">Open</span>';
        const act = g.resolved ? '-' : `<button onclick="resolveGap('${g.id}')" class="text-cyan-400 hover:underline">Resolve</button>`;
        tr.innerHTML = `
          <td class="py-2.5 px-3 text-slate-200">${g.question}</td>
          <td class="py-2.5 px-3 text-slate-400">${Math.round(g.confidence * 100)}%</td>
          <td class="py-2.5 px-3 font-semibold">${st}</td>
          <td class="py-2.5 px-3">${act}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function resolveGap(gapId) {
      await fetch(`/api/v1/vaults/${activeVault}/gaps/${gapId}/resolve`, {method: 'POST'});
      loadHealth();
    }

    async function runLinter() {
      if (!activeVault) return;
      const res = await fetch(`/api/v1/vaults/${activeVault}/lint`);
      const data = await res.json();
      const list = document.getElementById('lintIssuesList');
      list.innerHTML = '';
      document.getElementById('lintSummary').innerText = `Audit complete: ${data.total_issues} issue(s) detected.`;

      (data.issues || []).forEach(iss => {
        const li = document.createElement('li');
        const color = iss.severity === 'error' ? 'text-red-400' : 'text-amber-400';
        li.className = `${color} flex items-start gap-2`;
        li.innerHTML = `<span>•</span> <span><strong>[${iss.category}]</strong> ${iss.message} (${iss.file_path || 'vault'})</span>`;
        list.appendChild(li);
      });
    }

    async function fixLinter() {
      if (!activeVault) return;
      const res = await fetch(`/api/v1/vaults/${activeVault}/lint/fix`, {method: 'POST'});
      const data = await res.json();
      alert(`Auto-fix complete: ${data.fixed_count} issue(s) resolved.`);
      runLinter();
      loadHealth();
    }

    function renderRegistry(vaults) {
      const grid = document.getElementById('registryGrid');
      grid.innerHTML = '';

      vaults.forEach(v => {
        const card = document.createElement('div');
        card.className = 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-3';
        const validBadge = v.valid 
          ? '<span class="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800/60 px-2 py-0.5 rounded-full">Valid</span>'
          : '<span class="text-[10px] bg-red-950 text-red-300 border border-red-800/60 px-2 py-0.5 rounded-full">Missing Path</span>';

        card.innerHTML = `
          <div class="flex justify-between items-start">
            <h4 class="font-bold text-sm text-white">${v.name} ${v.is_default ? '<span class="text-amber-400">★</span>' : ''}</h4>
            ${validBadge}
          </div>
          <div class="text-[11px] font-mono text-slate-400 truncate" title="${v.path}">${v.path}</div>
          <div class="text-xs text-slate-400">${v.chunks} chunks indexed</div>
          <div class="flex gap-2 pt-2 border-t border-slate-800/60 text-xs">
            <button onclick="setDefaultVault('${v.name}')" class="text-cyan-400 hover:underline">Set Default</button>
            <button onclick="deleteVault('${v.name}')" class="text-red-400 hover:underline ml-auto">Unregister</button>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    async function setDefaultVault(name) {
      await fetch(`/api/v1/vaults/${name}/default`, {method: 'POST'});
      loadVaults();
    }

    async function deleteVault(name) {
      if (!confirm(`Unregister vault '${name}'? (Local files will be preserved)`)) return;
      await fetch(`/api/v1/vaults/${name}`, {method: 'DELETE'});
      loadVaults();
    }

    async function pruneVaults() {
      const res = await fetch('/api/v1/vaults/prune', {method: 'POST'});
      const data = await res.json();
      alert(`Pruned ${data.removed_count} stale/temporary vault(s).`);
      loadVaults();
    }

    async function submitRegisterVault() {
      const path = document.getElementById('regPath').value.trim();
      const name = document.getElementById('regName').value.trim();
      if (!path) return alert('Path is required.');
      const res = await fetch('/api/v1/vaults/register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path, name: name || null})
      });
      if (res.ok) {
        alert('Vault registered successfully.');
        document.getElementById('regPath').value = '';
        document.getElementById('regName').value = '';
        loadVaults();
      } else {
        const err = await res.json();
        alert('Error: ' + err.detail);
      }
    }

    async function submitInitVault() {
      const path = document.getElementById('initPath').value.trim();
      const name = document.getElementById('initName').value.trim();
      const description = document.getElementById('initDesc').value.trim();
      if (!path) return alert('Path is required.');
      const res = await fetch('/api/v1/vaults/init', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path, name: name || null, description})
      });
      if (res.ok) {
        alert('Vault initialized successfully.');
        document.getElementById('initPath').value = '';
        document.getElementById('initName').value = '';
        document.getElementById('initDesc').value = '';
        loadVaults();
      } else {
        const err = await res.json();
        alert('Error: ' + err.detail);
      }
    }

    async function submitNewNote() {
      if (!activeVault) return;
      const title = document.getElementById('noteTitle').value.trim();
      const subfolder = document.getElementById('noteSubfolder').value.trim();
      const content = document.getElementById('noteContent').value.trim();
      if (!title || !content) return alert('Title and content are required.');

      const res = await fetch(`/api/v1/vaults/${activeVault}/notes`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title, subfolder, content})
      });
      if (res.ok) {
        alert('Note created and indexed successfully.');
        document.getElementById('noteTitle').value = '';
        document.getElementById('noteSubfolder').value = '';
        document.getElementById('noteContent').value = '';
        loadCatalog();
        loadHealth();
      } else {
        const err = await res.json();
        alert('Error: ' + err.detail);
      }
    }

    // DRAG AND DROP HANDLING
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    dropZone.onclick = () => fileInput.click();
    dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-cyan-400'); };
    dropZone.ondragleave = () => dropZone.classList.remove('border-cyan-400');
    dropZone.ondrop = (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-cyan-400');
      if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    };

    function handleFileSelect(e) {
      if (e.target.files.length) uploadFiles(e.target.files);
    }

    async function uploadFiles(files) {
      if (!activeVault) return alert('Please select a vault first.');
      const statusDiv = document.getElementById('uploadStatus');
      statusDiv.classList.remove('hidden');
      statusDiv.innerText = `Uploading ${files.length} file(s) into inbox/...`;

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        await fetch(`/api/v1/vaults/${activeVault}/upload?filename=${encodeURIComponent(file.name)}&target=inbox`, {
          method: 'POST',
          headers: {'Content-Type': file.type || 'application/octet-stream'},
          body: file
        });
      }

      statusDiv.innerText = `Uploaded ${files.length} file(s) into inbox/. Click "Process Inbox" to run OCR/VLM.`;
    }

    async function triggerInboxIngest() {
      if (!activeVault) return;
      const res = await fetch(`/api/v1/vaults/${activeVault}/ingest`, {method: 'POST'});
      const data = await res.json();
      alert(`Inbox processed: ${data.ingested_notes.length} note(s) converted and indexed.`);
      loadCatalog();
      loadHealth();
    }

    async function saveCurrentFinding() {
      if (!latestFinding || !activeVault) return;
      const btn = document.getElementById('saveFindingBtn');
      btn.innerText = 'Saving...';
      const res = await fetch(`/api/v1/vaults/${activeVault}/save-finding`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(latestFinding)
      });
      const data = await res.json();
      btn.innerText = '✓ Saved in findings/';
      alert(`Finding saved as permanent note: ${data.file_path}`);
      loadCatalog();
      loadHealth();
    }

    async function loadSettings() {
      if (!activeVault) return;
      const res = await fetch(`/api/v1/vaults/${activeVault}`);
      const data = await res.json();
      document.getElementById('cfgMaxHops').value = data.config.max_hops;
      document.getElementById('cfgMaxIterations').value = data.config.max_iterations;
      document.getElementById('cfgModelEndpoint').value = data.config.model_endpoint;
      document.getElementById('cfgModelName').value = data.config.model_name;
      document.getElementById('cfgPeriodicIndexing').value = data.config.periodic_indexing ? 'true' : 'false';
      document.getElementById('cfgInterval').value = data.config.index_interval_seconds;
    }

    async function saveSettings() {
      const payload = {
        max_hops: parseInt(document.getElementById('cfgMaxHops').value),
        max_iterations: parseInt(document.getElementById('cfgMaxIterations').value),
        model_endpoint: document.getElementById('cfgModelEndpoint').value,
        model_name: document.getElementById('cfgModelName').value,
        periodic_indexing: document.getElementById('cfgPeriodicIndexing').value === 'true',
        index_interval_seconds: parseInt(document.getElementById('cfgInterval').value),
      };
      await fetch(`/api/v1/vaults/${activeVault}/settings`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      alert('Vault settings updated.');
    }

    async function syncVault() {
      if (!activeVault) return;
      const btn = document.getElementById('syncBtn');
      btn.innerText = 'Syncing...';
      const res = await fetch(`/api/v1/vaults/${activeVault}/index`, {method: 'POST'});
      const data = await res.json();
      btn.innerText = '⚡ Re-Index';
      alert(`Indexed in ${data.duration_seconds}s. Total chunks: ${data.total_chunks}`);
      loadVaults();
    }

    function switchTab(tab) {
      const tabs = ['cockpit', 'notes', 'maintenance', 'registry', 'settings'];
      tabs.forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        const btn = document.getElementById(`tabBtn-${t}`);
        if (el) el.classList.toggle('hidden', t !== tab);
        if (btn) {
          btn.className = t === tab 
            ? 'pb-3 px-4 border-b-2 border-cyan-400 text-cyan-400 font-semibold flex items-center gap-2 transition'
            : 'pb-3 px-4 border-b-2 border-transparent text-slate-400 hover:text-white font-medium flex items-center gap-2 transition';
        }
      });
    }

    async function executeResearch() {
      const q = document.getElementById('questionInput').value.trim();
      if (!q) return;

      const pBox = document.getElementById('progressBox');
      const stepsList = document.getElementById('stepsList');
      const aCard = document.getElementById('answerCard');
      const errBox = document.getElementById('errorBox');
      const btn = document.getElementById('researchBtn');
      const selectedModel = document.getElementById('modelSelect').value;
      const stepBadge = document.getElementById('stepCountBadge');
      const pulse = document.getElementById('pulseIndicator');
      let stepCount = 0;

      pBox.classList.remove('hidden');
      pBox.removeAttribute('open');
      if (pulse) {
        pulse.classList.add('animate-pulse');
        pulse.classList.remove('text-emerald-400', 'text-red-400');
        pulse.classList.add('text-cyan-400');
      }
      if (stepBadge) {
        stepBadge.classList.add('hidden');
        stepBadge.innerText = '0 steps';
      }
      document.getElementById('traversalCounter').innerText = 'Researching...';
      aCard.classList.add('hidden');
      errBox.classList.add('hidden');
      const fBox = document.getElementById('frictionBox');
      if (fBox) fBox.classList.add('hidden');
      stepsList.innerHTML = '';
      btn.disabled = true;
      btn.innerText = 'Researching...';
      latestFinding = null;

      const payload = {vault: activeVault, question: q};
      if (selectedModel) payload.model_name = selectedModel;

      const resp = await fetch('/api/v1/research', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\\n\\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          const matchEvent = line.match(/^event: (\\w+)/m);
          const matchData = line.match(/^data: (.+)$/m);
          if (matchEvent && matchData) {
            const ev = matchEvent[1];
            const data = JSON.parse(matchData[1]);

            if (ev === 'step' && data.thought) {
              stepCount++;
              if (stepBadge) {
                stepBadge.classList.remove('hidden');
                stepBadge.innerText = `${stepCount} step${stepCount === 1 ? '' : 's'}`;
              }
              document.getElementById('traversalCounter').innerText = `[${data.node}]`;
              const li = document.createElement('li');
              li.className = 'flex items-start gap-2';
              li.innerHTML = `<span class="text-cyan-400 font-bold">[${data.node}]</span> <span>${data.thought}</span>`;
              stepsList.appendChild(li);
            } else if (ev === 'error') {
              if (pulse) {
                pulse.classList.remove('animate-pulse');
                pulse.classList.remove('text-cyan-400');
                pulse.classList.add('text-red-400');
              }
              document.getElementById('traversalCounter').innerText = 'Failed';
              errBox.classList.remove('hidden');
              document.getElementById('errorTitle').innerText = 'Model Execution Error';
              document.getElementById('errorMsg').innerText = data.error || 'Unknown error occurred.';
              const mBox = document.getElementById('errorAvailModels');
              const mList = document.getElementById('errorModelsList');
              if (data.available_models && data.available_models.length > 0) {
                mBox.classList.remove('hidden');
                mList.innerHTML = '';
                data.available_models.forEach(m => {
                  const li = document.createElement('li');
                  li.textContent = m;
                  mList.appendChild(li);
                });
              } else {
                mBox.classList.add('hidden');
              }
            } else if (ev === 'result') {
              if (pulse) {
                pulse.classList.remove('animate-pulse');
                pulse.classList.remove('text-cyan-400');
                pulse.classList.add('text-emerald-400');
              }
              document.getElementById('traversalCounter').innerText = `${stepCount} steps (Completed)`;
              latestFinding = {
                query: q,
                answer: data.answer,
                confidence: data.confidence,
                sources: data.sources || []
              };
              document.getElementById('answerContent').innerText = data.answer;
              document.getElementById('confidenceBadge').innerText = `${Math.round(data.confidence * 100)}% Confidence`;
              const badges = document.getElementById('sourcesBadges');
              badges.innerHTML = '';
              (data.sources || []).forEach(src => {
                const sBadge = document.createElement('span');
                sBadge.className = 'bg-slate-800 text-cyan-300 border border-slate-700 px-2.5 py-0.5 rounded-md';
                sBadge.innerText = src;
                badges.appendChild(sBadge);
              });
              document.getElementById('saveFindingBtn').innerText = '💾 Save Finding as Note';
              const fBox = document.getElementById('frictionBox');
              const fMsg = document.getElementById('frictionMsg');
              if (fBox && fMsg) {
                if (data.friction_alert) {
                  fBox.classList.remove('hidden');
                  fMsg.innerText = data.friction_alert;
                } else {
                  fBox.classList.add('hidden');
                }
              }
              aCard.classList.remove('hidden');
            }
          }
        }
      }
      btn.disabled = false;
      btn.innerText = 'Research';
    }

    loadModels();
    loadVaults();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return HTML_DASHBOARD

