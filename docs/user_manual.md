# Alcuin User Manual

Welcome to the official user manual for **Alcuin**, an autonomous, local-first personal knowledge engine. Alcuin turns collections of plain Markdown files and document archives into a queryable, self-healing knowledge graph using local language models, embedded vector storage, and cyclic graph reasoning.

---

## 1. Philosophy & Architecture

Alcuin is built on four core architectural principles:

1. **In-Situ Vault Architecture**: Your markdown files stay exactly where they are. Alcuin does not lock your notes into an opaque proprietary database. You can continue editing, creating, and linking your notes with [Obsidian](https://obsidian.md), VS Code, Neovim, or standard command-line tools.
2. **Local-First & Completely Private**: Embeddings, indexing, and agentic reasoning run entirely on your local hardware by default using [LanceDB](https://lancedb.com/) and [Ollama](https://ollama.ai). Your personal wiki, financial records, medical histories, and contracts never touch the cloud unless you explicitly configure a remote provider.
3. **Hybrid Search & Cyclic Graph Agent**: Alcuin combines BM25 full-text keyword search with dense vector embeddings (`nomic-embed-text`) and bidirectional Markdown wikilink traversal (`[[Note Name]]`). An autonomous multi-hop reasoning agent navigates documents, evaluates evidence sufficiency, follows promising links, and synthesizes answers with exact citations.
4. **Bounded Governors & Deterministic Rigor**: A mathematical **Context Governor** dynamically budgets evidence to prevent local LLM context overflow, an **Output Governor** audits synthesis completeness against multi-part inquiries, and a **Retrieval Friction Monitor** surfaces self-healing opportunities when queries require complex multi-hop traversal.

---

## 2. Prerequisites & Installation

### System Requirements
* **Python**: 3.11 or higher
* **Local LLM Engine**: [Ollama](https://ollama.com/) (recommended for local offline operation)
* **OCR Support (Optional)**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for extracting scanned PDFs, images, and physical documents

### Installing Dependencies

#### Ubuntu / Debian Linux
```bash
sudo apt update
sudo apt install python3-pip python3-venv tesseract-ocr
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

---

### Pulling Recommended Ollama Models

For local operation, pull the embedding model and at least one local reasoning model:

```bash
# Recommended embedding model (384-dimensional dense vectors, fast & accurate)
ollama pull nomic-embed-text

# Recommended default local reasoning models
ollama pull llama3.2:3b     # Ultra-fast, lightweight (requires ~2GB VRAM)
ollama pull qwen2.5:7b      # High-reasoning local model (requires ~5GB VRAM)
```

---

### Installing Alcuin

Clone the repository and install in editable development mode within a virtual environment:

```bash
git clone https://github.com/fredericf75/alcuin.git
cd alcuin
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

Verify your installation:
```bash
alcuin --help
```

---

## 3. Vault Architecture & Multi-Vault Management

An Alcuin vault is any folder containing markdown files and an in-situ `.alcuin/` metadata directory.

```text
my-vault/
├── .alcuin/                <-- Automatically created and managed by Alcuin
│   ├── config.toml         <-- Local vault configuration & exclusions
│   ├── lancedb/            <-- Embedded vector store & full-text index
│   ├── manifest.json       <-- File hashes, modification times, & wikilink graph
│   └── knowledge_gaps.json <-- Unanswered queries & blind spots
├── inbox/                  <-- Drop raw PDFs, DOCX, scans, and receipts here
├── archive/                <-- Ingested raw files moved here safely
├── findings/               <-- Synthesized research findings saved as notes
├── 00_Meta/
│   ├── Index.md
│   └── People.md
├── Legal/
│   ├── Lease.md
│   └── Ordinances.md
└── Personal/
    └── Charlie.md
```

### The Intake Invariant
* **Active Vault is 100% Markdown**: Raw binaries (`.pdf`, `.png`, `.docx`) are kept in `inbox/` or `archive/` so they never pollute active semantic note searches.
* **Preserved Archive**: Moving ingested files to `archive/` preserves original evidentiary records. Even if an archive file is moved or deleted, the synthesized Markdown note remains permanently intact.

---

### Managing Vaults (`alcuin vault`)

Alcuin maintains a global registry of vaults located at `~/.config/alcuin/vaults.toml`. You can register, switch, and query multiple vaults from anywhere on your system.

#### 1. Initialize a New Vault
Initialize a new vault in the current directory or a target folder:
```bash
# In the current directory
alcuin init

# In a specific directory with a custom name
alcuin vault init /path/to/my_notes --name personal
```

#### 2. List Registered Vaults
```bash
alcuin vault list
```
Displays an overview table of registered vaults, filesystem paths, and active status.

#### 3. Register and Unregister Vaults
```bash
# Register an existing vault directory
alcuin vault register /mnt/wiki --name svr_wiki

# Unregister a vault from the global registry (does not delete note files)
alcuin vault unregister svr_wiki
```

#### 4. Set the Default Vault
Set the default vault used when commands are run outside a vault directory:
```bash
alcuin vault default svr_wiki
```

#### 5. Prune Dead or Moved Vaults
Automatically remove stale registry entries for paths that no longer exist:
```bash
alcuin vault prune
```

#### 6. Targeting Vaults Across Any Command
Every Alcuin command supports the `-v, --vault <name|path>` option. You never need to `cd` into a directory to research or manage it:
```bash
alcuin --vault svr_wiki "What is our home insurance deductible?"
alcuin lint --vault personal
alcuin index --vault aia_vault
```

---

## 4. Autonomous Research Engine

Alcuin's core capability is multi-hop agentic research across your notes.

```bash
# Direct natural language question
alcuin "What are the decibel noise limits between 9 PM and 2 AM?"

# Multi-clause or multi-part inquiry
alcuin "Under the construction agreement, what is the notice deadline for differing site conditions and who acts as the initial decision maker?"
```

### Research Options & CLI Flags

| Flag | Short | Description |
| :--- | :--- | :--- |
| `-v, --vault <target>` | `-v` | Vault name or directory path |
| `--model <name>` | | Override LLM model (e.g. `qwen2.5:7b`, `llama3.2:3b`) |
| `--model-endpoint <url>` | | Override Ollama endpoint URL (e.g. `http://localhost:11434`) |
| `--temperature <float>` | | Sampling temperature (default: `0.2` for factual rigor) |
| `--max-hops <int>` | | Maximum wikilink traversal depth (default: `3`) |
| `--max-iterations <int>` | | Maximum research feedback loops (default: `4`) |
| `-e, --with-evidence` | `-e` | Append full admitted grounding evidence dossier below answer |
| `-E, --evidence-only` | `-E` | Output only admitted search evidence, bypassing LLM synthesis (zero token cost) |
| `--context-window <int>` | `--ctx` | Override model context window in tokens (default: `16384`) |
| `--headroom <int>` | | Generation runway token buffer (overrides vault config) |
| `-V, --verbose` | `-V` | Display live agent thought iterations, node evaluations, and queries |
| `--json` | | Output structured JSON containing query, answer, confidence, sources, and telemetry |

---

### Understanding the Cyclic Reasoning Trace (`--verbose`)

When running with `-V` or `--verbose`, Alcuin displays real-time execution steps:

1. **`[decompose]`**: Splits complex multi-part questions into targeted search sub-goals and sub-questions.
2. **`[retrieve]`**: Executes hybrid vector + BM25 search, applies System 1 link pruning, and expands sibling headings.
3. **`[evaluate]`**: An evidence sufficiency auditor checks whether the collected snippets completely answer the inquiry:
   - For atomic queries: directly verifies if the factual information is satisfied.
   - For multi-part queries: maintains a **Sub-Question Coverage Matrix** verifying that all sub-questions are covered before proceeding.
   - Applies **Factual & Biographical Equivalence** (e.g., `Date of Birth` satisfies `birthday`) while strictly respecting legal terms of art.
   - Clamps follow-up queries strictly to vault entities, rejecting web-search tropes.
4. **`[synthesize]`**: Transcribes verbatim numbers, citations, and synthesizes a direct answer with full confidence scoring.

---

### Safe Grounding & Evidence Inspection (`-e` and `-E`)

To inspect exactly what evidence the agent admitted into context:

```bash
# Append the evidentiary dossier below the answer
alcuin --vault legal -e "What is the penalty for early lease termination?"

# View ONLY the admitted evidence blocks without calling the LLM synthesizer (instant, zero token cost)
alcuin --vault legal -E "What is the penalty for early lease termination?"
```

---

## 5. The Modern Web Cockpit (`alcuin serve`)

Alcuin includes a responsive, browser-based operations suite with real-time SSE streaming.

```bash
alcuin serve --port 8080
```
Open **`http://localhost:8080`** in your browser.

The Web Cockpit is divided into 4 operational tabs:

### Tab 1: Overview & Research
* **Real-Time Streaming**: Watch intermediate agent thoughts stream live via Server-Sent Events.
* **Model Selector**: Automatically queries `GET /api/tags` on your Ollama daemon to discover and switch between installed models on the fly.
* **Collapsible Reasoning & Traversal Trace**: Traversal iterations and evaluation steps are positioned directly beneath the synthesized finding and are collapsible-by-default to keep the primary finding prominent.
* **One-Click Finding Preservation**: Click **Save Finding as Note** to automatically write a timestamped research finding note into `<vault>/findings/` and immediately index it into memory.
* **High Retrieval Friction Alerts**: When a query requires multi-hop exploration ($\ge 3$ turns), an actionable amber banner surfaces, offering one-click saving to accelerate future queries to $O(1)$.

### Tab 2: Vault Operations
* Switch active vaults with a single click.
* Inspect vault paths, total document counts, and configuration settings.
* Initialize new in-situ vaults directly from the browser.
* Register existing directories and prune stale or moved vault paths.
* Set system-wide default vaults.

### Tab 3: Notes & Document Intake
* **In-Browser Note Drafting**: Quickly compose and save new markdown notes with custom titles and subfolder destinations.
* **Drag-and-Drop Document Intake**: Drag PDFs, Word documents (`.docx`), text files, or scanned images directly onto the drop zone to stage them into `<vault>/inbox/`.
* **Process Inbox**: Trigger OCR and text extraction with a single button.

### Tab 4: Maintenance & Diagnostics
* **Live Vault Linter**: Run a complete integrity check on the active vault to discover broken wikilinks, sync drift, or structural anomalies.
* **One-Click Repair**: Execute `lint --fix` to generate missing folders and repair heading hygiene.
* **Knowledge Gap Manager**: Inspect unanswered questions, review ask counts and confidence levels, and resolve gaps with target notes.
* **Reindex Trigger**: Trigger incremental or forced index rebuilding directly from the UI.

---

## 6. Document Ingestion & Intake (`alcuin ingest`)

Drop physical scans, PDFs, digital receipts, statements, and image files into your vault's `inbox/` folder:

```bash
alcuin ingest
```

### Ingestion Workflow
1. **Discovery**: Scans `inbox/` for supported files (`.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.txt`, `.md`).
2. **Deduplication**: Computes SHA-256 hashes to guarantee duplicate files are never re-ingested.
3. **Extraction & OCR**: Extracts native PDF/DOCX text layers and runs Tesseract OCR on scanned images and receipts.
4. **Note Generation**: Produces a clean Markdown note in the vault with document headers, timestamps, and structured text.
5. **Safe Archival**: Moves original files from `inbox/` to `archive/`.

#### Ingest Options:
* `--dry-run`: Preview files to be processed without modifying files.
* `--force`: Force re-processing of previously ingested files.
* `--inbox-dir <path>`: Override intake directory.
* `--archive-dir <path>`: Override archive destination.

---

## 7. Vault Health, Maintenance & Self-Healing

### Vault Linting (`alcuin lint`)

Keep your vault clean, connected, and structurally sound:

```bash
# Audit vault integrity
alcuin lint

# Automatically repair safe issues
alcuin lint --fix

# Automatically generate placeholder stub notes for dead wikilinks
alcuin lint --fix --create-stubs
```

#### What `lint` Audits:
* **Dead Wikilinks**: Identifies `[[Broken Link]]` references pointing to non-existent notes.
* **Sync Drift**: Detects notes modified, added, or deleted outside Alcuin.
* **Folder Structure**: Ensures `inbox/`, `archive/`, and `findings/` exist and are properly excluded from search indexes.
* **Heading Hygiene**: Detects notes lacking an H1 header and provides auto-fixes.

---

### Knowledge Gap Tracking (`alcuin gaps`)

When Alcuin encounters a query that cannot be answered with high confidence, it records an entry in `<vault>/.alcuin/knowledge_gaps.json`:

```bash
# List all recorded knowledge gaps
alcuin gaps list

# Resolve a gap once you have created the missing note
alcuin gaps resolve <gap-id> "Family/Charlie.md"

# Clear all resolved gaps
alcuin gaps clear
```

---

### Indexing (`alcuin index`)

Alcuin incrementally updates LanceDB vectors and BM25 tables whenever files are added or edited:

```bash
# Incremental synchronization (fast: only processes modified files)
alcuin index

# Full reconstruction (re-embeds all vault documents from scratch)
alcuin index --force
```

---

## 8. Effective Prompting & Multi-Clause Research

Because complex knowledge corpuses (municipal codes, commercial leases, construction contracts) distribute conditions across multiple sections, phrasing influences retrieval and synthesis precision:

### 1. Itemize Multi-Part Questions
When researching complex rules across documents, **enumerate individual questions**:
```bash
alcuin "Under the commercial lease:
1. What is the base rent due date?
2. What is the late payment grace period and penalty percentage?
3. What written notice is required before an Event of Default?"
```
*Why this works*: Alcuin's query planner recognizes numbered sub-clauses, executes dedicated multi-pass retrieval passes for each item, and validates coverage using a Sub-Question Coverage Matrix before synthesizing.

### 2. State Requirements Explicitly
* **Vague**: *"What happens if rent is late?"*
* **Targeted**: *"What is the exact grace period in days and the fee structure if rent is overdue?"*

### 3. Model Sizing & Register
* **Local 3B Models** (`llama3.2:3b`): Fast, lightweight, completely offline. Best for direct factual retrieval, single-target inquiries, and biographical lookups.
* **Local 7B–14B Models** (`qwen2.5:7b`, `qwen2.5:14b`): Excellent balance of local privacy and nuanced reasoning. Superior at distinguishing competing legal clauses and synthesis across multiple sections.
* **Frontier Models** (Gemini, Claude, GPT-4): Recommended for massive 50+ chunk legal cross-examination or complex municipal zoning analyses.

---

## 9. Configuration Reference

Alcuin resolves configuration hierarchically:
1. **CLI Flags** (highest precedence)
2. **Environment Variables** (`ALCUIN_MODEL`, `ALCUIN_ENDPOINT`, `GEMINI_API_KEY`, `OPENAI_API_KEY`)
3. **Vault Configuration** (`<vault>/.alcuin/config.toml`)
4. **Global Configuration** (`~/.config/alcuin/config.toml`)

### Annotated `config.toml`

```toml
[agent]
# LLM Endpoint & Model
model_endpoint = "http://localhost:11434"
model_name = "llama3.2:3b"
temperature = 0.2

# Graph Traversal Limits
max_hops = 3
max_iterations = 4

# Link Pruning (System 1 Semantic Gate)
link_pruning = true
link_pruning_threshold = 0.40
link_pruning_max_queue = 10

# Sibling Heading Expansion
sibling_traversal = true
sibling_pruning_threshold = 0.58
sibling_max_per_hit = 2

# Context Governor Safeguards
context_governor = true
context_window = 16384
context_headroom_tokens = 2048

[storage]
embedding_model = "nomic-embed-text"
fts_weight = 0.3                      # BM25 keyword weight
vector_weight = 0.7                   # Dense vector similarity weight
exclude_patterns = [
    ".git/*",
    "node_modules/*",
    "tmp/*",
    ".alcuin/*",
    "inbox/*",
    "archive/*"
]

[ingest]
ocr_engine = "tesseract"
inbox_dir = "inbox"
archive_dir = "archive"
```

---

## 10. Troubleshooting & Diagnostics

### Model Not Found Error (`ModelNotFoundError`)
If you specify a model name that is mistyped or not pulled in Ollama (e.g. `--model llama3`), Alcuin automatically connects to your endpoint, displays an error panel, and lists all available models currently installed:
```text
Model 'llama3' not found on endpoint 'http://localhost:11434'.
Available models on endpoint:
  • llama3.2:3b
  • qwen2.5:7b
  • nomic-embed-text:latest
```
Fix: Use the exact model tag: `alcuin --model llama3.2:3b "..."`

### "Tesseract not found in PATH"
Verify that Tesseract OCR is installed on your operating system:
* Linux: `sudo apt install tesseract-ocr`
* macOS: `brew install tesseract`
* Verify with: `tesseract --version`

### Dead Wikilinks After Renaming Files
If notes were renamed outside Alcuin, run:
```bash
alcuin lint --fix --create-stubs
```
This generates placeholder notes for all missing link destinations so your graph traversal never stalls.

### Resetting or Forcing Index Reconstruction
If notes were heavily edited outside of Alcuin or moved across folders:
```bash
alcuin index --force
```
