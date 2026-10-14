# Alcuin User Manual

Welcome to the comprehensive user manual for **Alcuin**, an autonomous, local-first personal knowledge engine.

---

## 1. Philosophy & Architecture

Alcuin is designed around three core principles:

1. **In-Situ Vault Architecture**: Your markdown files stay where they are. Alcuin does not import notes into an opaque proprietary database. You can continue editing your files with Obsidian, VS Code, Neovim, or standard command-line tools.
2. **Local-First & Private**: Notes, embeddings, and agent reasoning run locally by default using [LanceDB](https://lancedb.com/) and [Ollama](https://ollama.ai). Your private records, financial receipts, and legal documents never leave your machine unless you choose a cloud provider.
3. **Hybrid Search & Cyclic Graph Agent**: Alcuin combines BM25 full-text search (lexical keyword matching) and vector embeddings (semantic meaning) with bidirectional markdown wikilink traversal (`[[Note Name]]`). A multi-hop cyclic agent navigates documents, follows links, and synthesizes answers with citations.

---

## 2. Prerequisites & Installation

### System Requirements
* Python 3.11 or higher
* [Ollama](https://ollama.com/) (recommended for local offline operation)
* *(Optional)* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for scanning paper receipts, documents, and PDFs

### Installing Dependencies

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install python3-pip tesseract-ocr
```

#### macOS (Homebrew)
```bash
brew install tesseract
```

#### Windows
Install Tesseract via winget:
```powershell
winget install UB-Mannheim.TesseractOCR
```

### Pulling Ollama Models
For best local performance:
```bash
# Recommended embedding model (384 dimensions, fast & accurate)
ollama pull nomic-embed-text

# Recommended default local agent models
ollama pull llama3.2:3b     # Lightweight, fast
ollama pull qwen2.5:7b      # High-reasoning local model
```

### Installing Alcuin
```bash
git clone https://github.com/fredericf75/alcuin.git
cd alcuin
pip install -e .
```

Verify installation:
```bash
alcuin --help
```

---

## 3. Vault Structure

An Alcuin vault is any directory containing markdown files and a localized `.alcuin/` metadata directory.

```text
my-vault/
├── .alcuin/                <-- Managed automatically by Alcuin
│   ├── config.toml         <-- Local vault configuration & exclusions
│   ├── lancedb/            <-- Embedded vector store & full-text index
│   ├── graph.json          <-- Bidirectional wikilink graph cache
│   └── gaps.json           <-- Unanswered queries & blind spots
├── inbox/                  <-- Drop raw PDFs, images, receipts here
├── archive/                <-- Ingested raw files moved here safely
├── 00_Meta/
│   ├── Index.md
│   └── Contacts.md
├── Finance/
│   ├── Taxes 2025.md
│   └── Insurance.md
└── Family/
    └── Charlie.md
```

### The Intake Invariant
* **Active Vault is 100% Markdown**: Raw binaries (PDFs, PNGs, JPEGs) never pollute active note searches.
* **Decoupled Archive**: Files moved to `archive/` remain preserved. Even if an archived file is deleted or pruned, your synthesized Markdown note remains intact.

---

## 4. Workflows & Commands

### 4.1 Research (`alcuin <question>`)
Alcuin uses natural language querying directly from the command line:

```bash
# Direct question
alcuin what is our home insurance policy number and deductible?

# Multi-hop inquiry across multiple documents
alcuin who is Charlie's pediatrician and what is their clinic address?
```

#### Research Options:
* `--max-hops <int>`: Maximum wikilink hops the agent will traverse (default: `3`).
* `--model <name>`: Override model for this query (e.g., `--model qwen2.5:7b`).
* `--model-endpoint <url>`: Override Ollama or API endpoint.
* `--temperature <float>`: Agent creativity/sampling temperature (default: `0.2`).
* `--verbose`: Stream internal agent thought steps and hops to the terminal.

---

### 4.2 Document Ingestion (`alcuin ingest`)
Drop physical document scans, digital receipts, statements, and images into `inbox/`, then process them:

```bash
alcuin ingest
```

What happens:
1. Alcuin inspects `inbox/` for supported files (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`).
2. Deduplication check: Computes SHA-256 hash to prevent re-ingesting duplicate files.
3. OCR / Extraction: Runs Tesseract OCR on images/scans and extracts text layers from PDFs.
4. Generates a clean Markdown note in the vault with extraction metadata and source headers.
5. Moves the original raw file from `inbox/` to `archive/`.

#### Ingest Options:
* `--dry-run`: Preview files to be ingested without making changes.
* `--force`: Re-process files even if previously ingested.
* `--inbox-dir <path>`: Custom intake directory (default: `<vault>/inbox`).
* `--archive-dir <path>`: Custom archive directory (default: `<vault>/archive`).

---

### 4.3 Vault Health & Self-Healing (`alcuin lint`)
Audit vault integrity, sync state, and wikilink connectivity:

```bash
alcuin lint
```

Fix issues automatically:
```bash
alcuin lint --fix
```

#### What `lint` Detects and Repairs:
* **Dead Wikilinks**: Finds `[[Links]]` that reference non-existent notes. Use `--create-stubs` with `--fix` to automatically generate stub notes for missing targets.
* **Sync Drift**: Identifies notes added, updated, or removed outside Alcuin.
* **Missing Intake Folders**: Automatically upgrades older vaults by creating `inbox/` and `archive/` and adding them to `.alcuin/config.toml` exclusions.
* **Heading Hygiene**: Detects notes missing H1 titles and inserts them from filenames.

#### Lint Options:
* `--fix`: Automatically repair safe issues (scaffolding folders, fixing headers).
* `--create-stubs`: Generate empty placeholder notes for dead wikilinks during `--fix`.
* `--strict`: Return non-zero exit code on any warning.
* `--json`: Output report as JSON for CI or automation.

---

### 4.4 Knowledge Gap Tracking (`alcuin gaps`)
When Alcuin cannot find an answer to your question, it records a knowledge gap in `.alcuin/gaps.json`:

```bash
# List all active gaps
alcuin gaps list

# Resolve a gap once the information has been added
alcuin gaps resolve <gap-id> "Pediatrician.md"

# Clear all resolved gaps
alcuin gaps clear
```

---

### 4.5 Indexing (`alcuin index`)
Re-indexes and updates vector embeddings for modified documents:

```bash
alcuin index
```

To force a complete re-index of the entire vault:
```bash
alcuin index --force
```

---

### 4.6 Web UI & Daemon (`alcuin serve`)
Alcuin includes an embedded web dashboard with real-time SSE streaming for interactive visual research:

```bash
alcuin serve --port 8000
```
Open `http://localhost:8000` in your browser.

---

## 5. Configuration

Configuration is loaded hierarchically:
1. **CLI Flags** (highest priority)
2. **Environment Variables** (`ALCUIN_MODEL`, `ALCUIN_ENDPOINT`, `GEMINI_API_KEY`)
3. **Vault Configuration** (`<vault>/.alcuin/config.toml`)
4. **Global Configuration** (`~/.config/alcuin/config.toml`)

### Example Configuration (`config.toml`)

```toml
[agent]
model_endpoint = "http://localhost:11434"
model_name = "llama3.2:3b"
temperature = 0.2
max_hops = 3
max_iterations = 4

[storage]
embedding_model = "nomic-embed-text"
fts_weight = 0.3
vector_weight = 0.7
exclude_dirs = [".alcuin", "archive", "inbox"]

[ingest]
ocr_engine = "tesseract"
inbox_dir = "inbox"
archive_dir = "archive"
```

### Using Cloud Models

#### Google Gemini
Set your API key:
```bash
export GEMINI_API_KEY="your-gemini-key"
```
Run with Gemini:
```bash
alcuin --model gemini/gemini-2.5-flash "summarize our tax deductions"
```

#### OpenAI
Set your API key:
```bash
export OPENAI_API_KEY="your-openai-key"
```
Run with GPT:
```bash
alcuin --model openai/gpt-4o-mini "what is our insurance deductible?"
```

---

## 6. Troubleshooting

### "Tesseract not installed or not in PATH"
Ensure Tesseract is installed:
* Linux: `sudo apt install tesseract-ocr`
* Mac: `brew install tesseract`
* Verify with `tesseract --version`.

### "Dead wikilinks detected after renaming a note"
Run `alcuin lint` to see dead links, or `alcuin lint --fix --create-stubs` to generate placeholder files.

### "Rebuilding the search index"
If notes were heavily edited outside Alcuin, synchronize the index:
```bash
alcuin index --force
```
