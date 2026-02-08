# CodeTrading-RAG

A multi-channel YouTube RAG (Retrieval-Augmented Generation) system for trading strategies. Ingest video transcripts and code from YouTube trading channels, then query them with an LLM to generate strategies, explanations, and code reviews.

## Features

- **Multi-channel support** — Register and ingest multiple YouTube channels without destroying existing data
- **Non-destructive ingestion** — Incremental updates; new videos are added without re-indexing everything
- **Three query modes** — Strategy generation, concept explanation, and code review
- **Channel filtering** — Query against specific channels or all channels at once
- **Gradio web GUI** — Manage channels, run ingestion, and query from a browser
- **CLI interface** — Full command-line access to all features
- **Pluggable LLM backends** — Claude API or LM Studio (local models)
- **Pluggable embeddings** — HuggingFace (free, local) or OpenAI

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/CodeTrading-RAG.git
cd CodeTrading-RAG

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install with CLI only
pip install -e .

# Install with GUI support
pip install -e ".[gui]"

# Install with dev/test tools
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```
ANTHROPIC_API_KEY=your-key-here
```

Or for local models via LM Studio:

```
LLM_BACKEND=lmstudio
LMSTUDIO_MODEL=your-model-name
```

### 3. Add a Channel

```bash
codetrading-rag channel add "CodeTradingCafe" "https://www.youtube.com/@CodeTradingCafe"
```

### 4. Ingest

```bash
# Ingest all videos from the channel
codetrading-rag ingest --channel codetradingcafe

# Or limit to a few videos for testing
codetrading-rag ingest --channel codetradingcafe --limit 5
```

### 5. Query

```bash
codetrading-rag query "Create a mean reversion strategy using Bollinger Bands"
```

## CLI Reference

### Channel Management

```bash
# Add a channel
codetrading-rag channel add "ChannelName" "https://www.youtube.com/@ChannelName"

# List all registered channels
codetrading-rag channel list

# Remove a channel (does not delete data files)
codetrading-rag channel remove channelname
```

### Ingestion

```bash
# Ingest a specific channel
codetrading-rag ingest --channel channelname

# Ingest all registered channels
codetrading-rag ingest --all-channels

# Limit number of videos processed
codetrading-rag ingest --channel channelname --limit 10

# Re-index from scratch (deletes existing docs for that channel first)
codetrading-rag ingest --channel channelname --reindex
```

### Querying

```bash
# Generate a trading strategy (default mode)
codetrading-rag query "Create an RSI divergence strategy"

# Explain a concept
codetrading-rag query "How does VWAP work?" --mode explain

# Review a code file
codetrading-rag query "Check for bugs" --mode review --file my_strategy.py

# Filter to specific channel(s)
codetrading-rag query "momentum strategy" --channel codetradingcafe
codetrading-rag query "compare approaches" --channel channel1 --channel channel2
```

### Other Commands

```bash
# Show current configuration
codetrading-rag config

# Show ingestion status and statistics
codetrading-rag status

# Launch the web GUI
codetrading-rag gui

# Launch GUI on a custom port
codetrading-rag gui --port 8080

# Migrate existing single-channel data to multi-channel format
codetrading-rag migrate
```

## Web GUI

Launch with `codetrading-rag gui` and open `http://localhost:7860` in your browser.

The GUI has three tabs:

- **Channels** — Add/remove channels, view status and counts
- **Ingest** — Select channels, configure limits, start ingestion with live progress
- **Query** — Choose channels and mode, ask questions, view responses with source citations

## Configuration

All settings are loaded from a `.env` file. See `.env.example` for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `claude` | LLM provider: `claude` or `lmstudio` |
| `ANTHROPIC_API_KEY` | | Required for Claude backend |
| `CLAUDE_MODEL` | `claude-sonnet-4-5-20250929` | Claude model name |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio endpoint |
| `LMSTUDIO_MODEL` | | Required for LM Studio backend |
| `EMBEDDING_BACKEND` | `huggingface` | Embeddings: `huggingface` or `openai` |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace model |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage path |
| `DATA_DIR` | `./data` | Base data directory |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `GUI_PORT` | `7860` | Gradio server port |
| `COOKIES_FROM_BROWSER` | | Browser for cookie extraction (e.g. `chrome`) |

## Cookie Setup for YouTube

YouTube may block transcript requests from your IP. To avoid this, provide authentication cookies:

**Option 1: cookies.txt file**
1. Install a browser extension like "Get cookies.txt LOCALLY"
2. Visit youtube.com while logged in
3. Export cookies in Netscape format
4. Save as `cookies.txt` in the project root

**Option 2: Browser cookie extraction**
Add to your `.env`:
```
COOKIES_FROM_BROWSER=chrome
```
Supported browsers: chrome, firefox, brave, edge, opera, safari, chromium

## Architecture

```
data/
  channels.json                    # Channel registry
  channels/
    codetradingcafe/
      metadata/*.json              # Video metadata (title, description, etc.)
      transcripts/*.json           # Video transcripts with timestamps
    another-channel/
      metadata/*.json
      transcripts/*.json
  chroma_db/                       # Shared ChromaDB vector store
```

- **Single ChromaDB collection** with `channel_id` metadata on every document
- **Channel filtering** via ChromaDB `where` clauses during retrieval
- **Incremental ingestion** using `add_documents()` (never destroys existing data)
- Documents are chunked into three types: `transcript`, `description`, and `code`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,gui]"

# Run tests
python -m pytest tests/ -v

# Run with verbose logging
codetrading-rag -v ingest --channel codetradingcafe --limit 3
```

## Requirements

- Python 3.11+
- An LLM backend (Claude API key or LM Studio running locally)

## License

MIT
