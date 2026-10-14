# Alcuin

**Alcuin** is an autonomous, agentic personal knowledge engine. Designed for local-first personal dossiers, family records, technical specifications, and legal documents, it unites hierarchical clause-aware chunking, hybrid vector search (LanceDB + FTS), bidirectional markdown wikilink traversal, and cyclic LangGraph agent research.

---

## Key Features

* **In-Situ Vault Architecture**: Any folder becomes an Alcuin vault without copying files into a proprietary database. Metadata is managed locally in `.alcuin/`.
* **Zero-Friction Natural Research**: Ask questions naturally in the terminal without complex syntax:
  ```bash
  alcuin what is the address for Charlie's pediatrician
  ```
* **Multimodal Inbox Ingestion (`alcuin ingest`)**: Drop raw PDFs, scans, receipts, or photos into `<vault>/inbox/`. Alcuin extracts clean Markdown notes via Tesseract OCR or local vision models, archives the originals to `<vault>/archive/`, and keeps the active vault 100% Markdown.
* **Vault Health & Self-Healing (`alcuin lint --fix`)**: Detects dead wikilinks, sync drift, orphan database records, and missing headers. Automatically scaffolds missing folders, repairs headers, and generates stub notes.
* **Knowledge Gap Tracking (`alcuin gaps`)**: Automatically logs unanswered or low-confidence queries, tracks recurring blind spots in your knowledge base, and resolves them when answers are found.
* **Local-First & Multi-Provider**: Powered locally by [Ollama](https://ollama.ai) (`llama3.2`, `qwen2.5`, `nomic-embed-text`) or cloud endpoints (Google Gemini, OpenAI).
* **Dual Interface**: Direct CLI with clean Markdown output, plus an embedded FastAPI Web UI with real-time research streaming via Server-Sent Events (SSE).

---

## Quickstart

### 1. Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```
* *(Optional)* Tesseract OCR for physical document/receipt intake:
  * Ubuntu/Debian: `sudo apt install tesseract-ocr`
  * macOS: `brew install tesseract`
  * Windows: `winget install UB-Mannheim.TesseractOCR`

### 2. Installation
```bash
git clone https://github.com/fredericf75/alcuin.git
cd alcuin
pip install -e .
```

### 3. Initialize a Vault
Initialize any existing folder of notes or documents:
```bash
alcuin init ~/documents/personal --name personal
```
This creates the localized `.alcuin/` metadata directory along with `inbox/` and `archive/` folders.

### 4. Index Existing Documents
```bash
alcuin index
```

### 5. Research Your Knowledge Base
Ask any factual or multi-hop question directly:
```bash
alcuin what is our home insurance policy number
alcuin who is Charlie's pediatrician and what is the clinic phone number
```

---

## Command Reference

| Command | Description |
| :--- | :--- |
| `alcuin <question>` | Autonomous multi-hop research across vault documents (Default command) |
| `alcuin init [path]` | Initialize target directory as an in-situ Alcuin vault |
| `alcuin index` | Incrementally sync and vector-embed modified vault documents |
| `alcuin ingest` | Convert raw files from `inbox/` into vault markdown notes and archive originals |
| `alcuin lint` | Audit vault integrity (dead wikilinks, sync drift, hygiene). Pass `--fix` to repair |
| `alcuin gaps` | Inspect and manage unanswered knowledge gaps |
| `alcuin vault list` | List registered vaults across the system |
| `alcuin serve` | Launch the FastAPI Web UI dashboard and background indexing daemon |

For detailed flags and advanced options, run `alcuin <command> --help` or consult the [User Manual](docs/user_manual.md).

---

## Configuration Hierarchy

Configuration resolves in the following order of precedence:
1. **CLI Flags**: `--model`, `--model-endpoint`, `--max-hops`, `--temperature`
2. **Environment Variables**: `ALCUIN_MODEL`, `ALCUIN_ENDPOINT`, `GEMINI_API_KEY`
3. **Vault Configuration**: `<vault>/.alcuin/config.toml`
4. **Global Configuration**: `~/.config/alcuin/config.toml` (Linux/macOS) or `%APPDATA%\alcuin\config.toml` (Windows)

### Example `~/.config/alcuin/config.toml`:
```toml
[agent]
model_endpoint = "http://localhost:11434"
model_name = "llama3.2:3b"
temperature = 0.2
max_hops = 3
max_iterations = 4
link_pruning = true

[storage]
embedding_model = "nomic-embed-text"
fts_weight = 0.3
vector_weight = 0.7
```

---

## Documentation
* [Full User Manual](docs/user_manual.md): Comprehensive guide to vault operations, intake workflows, wikilink graphs, and model tuning.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
