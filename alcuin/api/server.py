"""
alcuin.api.server - FastAPI Server with SSE Research Streaming & Web Dashboard
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from alcuin.agent.graph import ResearchAgent
from alcuin.api.daemon import PeriodicIndexer
from alcuin.storage.engine import VaultStorage
from alcuin.storage.vault import Vault, VaultConfig, VaultRegistry

periodic_indexer = PeriodicIndexer(check_interval=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch background periodic indexing
    await periodic_indexer.start()
    yield
    # Shutdown: cleanly cancel background tasks
    await periodic_indexer.stop()


app = FastAPI(title="Alcuin Knowledge Engine", version="0.1.0", lifespan=lifespan)


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


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/vaults")
async def list_vaults():
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
            }
        )
    return {"vaults": result}


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
        final_state = None
        for node_name, state_update in agent.stream_research(req.question):
            final_state = state_update
            thought = ""
            if "thought_history" in state_update and state_update["thought_history"]:
                thought = state_update["thought_history"][-1]
            data = json.dumps({"node": node_name, "thought": thought})
            yield f"event: step\ndata: {data}\n\n"

        if not final_state or not final_state.get("final_answer"):
            final_state = agent.research(req.question)

        result_data = json.dumps(
            {
                "answer": final_state.get("final_answer", ""),
                "confidence": final_state.get("confidence", 0.0),
                "sources": final_state.get("sources", []),
            }
        )
        yield f"event: result\ndata: {result_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Alcuin Knowledge Engine</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans">
  <div class="max-w-5xl mx-auto p-6">
    <header class="flex justify-between items-center pb-6 border-b border-slate-800">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span class="text-cyan-400">●</span> Alcuin Knowledge Engine
        </h1>
        <p class="text-xs text-slate-400 mt-1">Autonomous, Agentic Research & Multi-Vault Storage</p>
      </div>
      <div class="flex items-center gap-3">
        <label class="text-xs text-slate-400">Active Vault:</label>
        <select id="vaultSelect" class="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-cyan-300 focus:outline-none focus:ring-1 focus:ring-cyan-400"></select>
        <button onclick="syncVault()" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs px-3 py-1.5 rounded transition">Index Now</button>
      </div>
    </header>

    <nav class="flex gap-4 mt-6 border-b border-slate-800 text-sm">
      <button onclick="switchTab('research')" id="tabBtn-research" class="pb-2 border-b-2 border-cyan-400 text-cyan-400 font-medium">Research</button>
      <button onclick="switchTab('settings')" id="tabBtn-settings" class="pb-2 border-b-2 border-transparent text-slate-400 hover:text-white font-medium">Vault Settings</button>
    </nav>

    <!-- RESEARCH TAB -->
    <main id="tab-research" class="mt-6 space-y-6">
      <div class="bg-slate-800/60 border border-slate-700/80 rounded-xl p-4 shadow-lg">
        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Research Question</label>
        <div class="flex gap-3">
          <input id="questionInput" type="text" placeholder="e.g. What is the address for Charlie's pediatrician?" 
                 class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 text-white placeholder-slate-500" 
                 onkeydown="if(event.key==='Enter') executeResearch()"/>
          <button onclick="executeResearch()" id="researchBtn" class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold px-5 py-2.5 rounded-lg text-sm transition">Research</button>
        </div>
      </div>

      <!-- PROGRESS FEED -->
      <div id="progressBox" class="hidden bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-2">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-cyan-400">Agent Reasoning Steps</h3>
        <ul id="stepsList" class="text-xs text-slate-300 space-y-1.5 font-mono"></ul>
      </div>

      <!-- ANSWER CARD -->
      <div id="answerCard" class="hidden bg-slate-800/80 border border-slate-700 rounded-xl p-6 shadow-xl space-y-4">
        <div class="flex justify-between items-center border-b border-slate-700/60 pb-3">
          <span class="text-xs font-semibold uppercase tracking-wider text-emerald-400">Synthesized Finding</span>
          <span id="confidenceBadge" class="text-xs bg-emerald-950 text-emerald-300 border border-emerald-800 px-2.5 py-0.5 rounded-full"></span>
        </div>
        <div id="answerContent" class="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap"></div>
        <div id="sourcesBox" class="text-xs text-slate-400 pt-3 border-t border-slate-700/60 flex items-center gap-2">
          <span>Sources:</span>
          <span id="sourcesList" class="text-cyan-300 font-mono"></span>
        </div>
      </div>
    </main>

    <!-- SETTINGS TAB -->
    <section id="tab-settings" class="mt-6 hidden bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 space-y-5">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-300">Vault Configuration</h2>
      <div class="grid grid-cols-2 gap-4 text-sm">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Max Hops</label>
          <input id="cfgMaxHops" type="number" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Max Iterations</label>
          <input id="cfgMaxIterations" type="number" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Model Endpoint</label>
          <input id="cfgModelEndpoint" type="text" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Model Name</label>
          <input id="cfgModelName" type="text" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-xs"/>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Periodic Indexing</label>
          <select id="cfgPeriodicIndexing" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-xs">
            <option value="true">Enabled</option>
            <option value="false">Disabled</option>
          </select>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Interval (Seconds)</label>
          <input id="cfgInterval" type="number" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-xs"/>
        </div>
      </div>
      <button onclick="saveSettings()" class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold px-4 py-2 rounded text-xs transition">Save Settings</button>
    </section>
  </div>

  <script>
    let activeVault = '';

    async function loadVaults() {
      const res = await fetch('/api/v1/vaults');
      const data = await res.json();
      const sel = document.getElementById('vaultSelect');
      sel.innerHTML = '';
      data.vaults.forEach((v, idx) => {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = `${v.name} (${v.chunks} chunks)`;
        sel.appendChild(opt);
        if (idx === 0) activeVault = v.name;
      });
      sel.onchange = (e) => { activeVault = e.target.value; loadSettings(); };
      if (activeVault) loadSettings();
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
      const res = await fetch(`/api/v1/vaults/${activeVault}/index`, {method: 'POST'});
      const data = await res.json();
      alert(`Indexed in ${data.duration_seconds}s. Total chunks upserted: ${data.total_chunks}`);
      loadVaults();
    }

    function switchTab(tab) {
      document.getElementById('tab-research').classList.toggle('hidden', tab !== 'research');
      document.getElementById('tab-settings').classList.toggle('hidden', tab !== 'settings');
      document.getElementById('tabBtn-research').className = tab === 'research' ? 'pb-2 border-b-2 border-cyan-400 text-cyan-400 font-medium' : 'pb-2 border-b-2 border-transparent text-slate-400 hover:text-white font-medium';
      document.getElementById('tabBtn-settings').className = tab === 'settings' ? 'pb-2 border-b-2 border-cyan-400 text-cyan-400 font-medium' : 'pb-2 border-b-2 border-transparent text-slate-400 hover:text-white font-medium';
    }

    async function executeResearch() {
      const q = document.getElementById('questionInput').value.trim();
      if (!q) return;

      const pBox = document.getElementById('progressBox');
      const stepsList = document.getElementById('stepsList');
      const aCard = document.getElementById('answerCard');
      const btn = document.getElementById('researchBtn');

      pBox.classList.remove('hidden');
      aCard.classList.add('hidden');
      stepsList.innerHTML = '';
      btn.disabled = true;
      btn.innerText = 'Searching...';

      const resp = await fetch('/api/v1/research', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({vault: activeVault, question: q})
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
              const li = document.createElement('li');
              li.textContent = `• [${data.node}] ${data.thought}`;
              stepsList.appendChild(li);
            } else if (ev === 'result') {
              document.getElementById('answerContent').innerText = data.answer;
              document.getElementById('confidenceBadge').innerText = `${Math.round(data.confidence * 100)}% Confidence`;
              document.getElementById('sourcesList').innerText = (data.sources || []).join(', ') || 'None';
              aCard.classList.remove('hidden');
            }
          }
        }
      }
      btn.disabled = false;
      btn.innerText = 'Research';
    }

    loadVaults();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return HTML_DASHBOARD
