# CodeTrading-RAG Implementation Plan

## Overview
Build a RAG (Retrieval-Augmented Generation) system that ingests video transcripts and code from the **CodeTradingCafe** YouTube channel, then uses an LLM (Claude API or LM Studio local model) to generate trading strategies grounded in those techniques.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI Interface                     │
│              (ingest / query / config)               │
├─────────────────────────────────────────────────────┤
│                   RAG Pipeline                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ Retriever │→ │  Context  │→ │   LLM Backend    │ │
│  │ (ChromaDB)│  │  Builder  │  │ Claude / LMStudio│ │
│  └──────────┘  └───────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────┤
│               Document Processing                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ Transcript│  │  Chunker  │  │   Embeddings     │ │
│  │ Fetcher   │  │ & Splitter│  │ OpenAI / HF      │ │
│  └──────────┘  └───────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
CodeTrading-RAG/
├── pyproject.toml              # Project metadata & dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Ignore .env, __pycache__, chroma_db/, etc.
├── README.md                   # Usage instructions
├── src/
│   └── codetrading_rag/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (click-based)
│       ├── config.py           # Configuration loader (.env + defaults)
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── channel.py      # Fetch video list from channel via yt-dlp
│       │   ├── transcripts.py  # Fetch transcripts via youtube-transcript-api
│       │   └── processor.py    # Chunk documents, extract code blocks, build metadata
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   └── store.py        # ChromaDB wrapper (create, load, add, query)
│       ├── llm/
│       │   ├── __init__.py
│       │   └── backends.py     # LLM factory: Claude API + LM Studio
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chain.py        # LangChain retrieval chain setup
│       │   └── prompts.py      # System prompts for strategy generation
│       └── embeddings/
│           ├── __init__.py
│           └── factory.py      # Embedding model factory (OpenAI / HuggingFace)
├── data/                       # Created at runtime, gitignored
│   ├── transcripts/            # Raw transcript JSON files
│   ├── metadata/               # Video metadata JSON files
│   └── chroma_db/              # ChromaDB persistent storage
└── tests/
    ├── __init__.py
    ├── test_ingest.py
    └── test_rag.py
```

## Implementation Steps

### Step 1: Project scaffolding
**Files:** `pyproject.toml`, `.env.example`, `.gitignore`, `src/codetrading_rag/__init__.py`

- Set up `pyproject.toml` with all dependencies
- Create `.env.example` with all configuration variables
- Create `.gitignore` to exclude `.env`, `data/`, `__pycache__/`, etc.

**Dependencies:**
```
langchain >= 0.3
langchain-core >= 0.3
langchain-community >= 0.3
langchain-anthropic >= 0.3
langchain-openai >= 0.3
langchain-chroma >= 0.2
langchain-huggingface >= 0.1
langchain-text-splitters >= 0.3
chromadb >= 0.5
youtube-transcript-api >= 1.0
yt-dlp >= 2024.0
python-dotenv >= 1.0
click >= 8.0
rich >= 13.0           # Pretty CLI output
```

### Step 2: Configuration (`src/codetrading_rag/config.py`)
Loads settings from `.env` file with sensible defaults:
- `LLM_BACKEND`: "claude" | "lmstudio" (default: "claude")
- `ANTHROPIC_API_KEY`: API key for Claude
- `CLAUDE_MODEL`: Model name (default: "claude-sonnet-4-5-20250929")
- `LMSTUDIO_BASE_URL`: LM Studio endpoint (default: "http://localhost:1234/v1")
- `LMSTUDIO_MODEL`: Model name loaded in LM Studio
- `EMBEDDING_BACKEND`: "openai" | "huggingface" (default: "huggingface")
- `OPENAI_API_KEY`: API key for OpenAI embeddings (optional)
- `HF_EMBEDDING_MODEL`: HuggingFace model (default: "sentence-transformers/all-mpnet-base-v2")
- `CHROMA_DB_PATH`: Path to ChromaDB storage (default: "./data/chroma_db")
- `CHANNEL_URL`: YouTube channel URL (default: "https://www.youtube.com/@CodeTradingCafe")
- `RETRIEVER_K`: Number of chunks to retrieve (default: 5)

### Step 3: Channel video fetcher (`src/codetrading_rag/ingest/channel.py`)
Uses `yt-dlp` to:
- Fetch list of all videos from the CodeTradingCafe channel
- Extract metadata: video_id, title, description, upload_date, duration, URL
- Save metadata to `data/metadata/{video_id}.json`
- Support incremental updates (skip already-fetched videos)

**Key function:**
```python
def fetch_channel_videos(channel_url: str, limit: int | None = None) -> list[VideoMetadata]
```

### Step 4: Transcript fetcher (`src/codetrading_rag/ingest/transcripts.py`)
Uses `youtube-transcript-api` v1.x (new instance-based API):
- Fetch transcript for each video ID
- Handle errors gracefully (some videos may lack transcripts)
- Save raw transcripts to `data/transcripts/{video_id}.json`
- Extract code blocks from video descriptions (regex for Python/Pine Script blocks)

**Key functions:**
```python
def fetch_transcript(video_id: str) -> FetchedTranscript | None
def extract_code_from_description(description: str) -> list[CodeBlock]
```

### Step 5: Document processor (`src/codetrading_rag/ingest/processor.py`)
Converts raw transcripts + metadata into LangChain `Document` objects with proper chunking:

- **Transcript chunks**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
- **Code chunks**: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\ndef ", "\nclass ", "\n\n", "\n"])`
- **Description chunks**: Kept as single documents (usually short)

**Metadata per chunk:**
```python
{
    "video_id": str,
    "video_title": str,
    "video_url": str,
    "upload_date": str,
    "chunk_type": "transcript" | "code" | "description",
    "start_time": float | None,  # For transcript chunks
    "end_time": float | None,    # For transcript chunks
}
```

**Key function:**
```python
def process_video(video_metadata: VideoMetadata, transcript: FetchedTranscript) -> list[Document]
```

### Step 6: Embedding factory (`src/codetrading_rag/embeddings/factory.py`)
Factory function that returns the configured embedding model:

```python
def get_embeddings(config: Config) -> Embeddings:
    if config.embedding_backend == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        return HuggingFaceEmbeddings(model_name=config.hf_embedding_model)
```

### Step 7: Vector store (`src/codetrading_rag/vectorstore/store.py`)
ChromaDB wrapper with persistence:

```python
class VectorStore:
    def __init__(self, config: Config)
    def create_from_documents(self, documents: list[Document]) -> None
    def add_documents(self, documents: list[Document]) -> None
    def as_retriever(self, k: int = 5) -> BaseRetriever
    def similarity_search(self, query: str, k: int = 5) -> list[Document]
    @property
    def document_count(self) -> int
```

### Step 8: LLM backends (`src/codetrading_rag/llm/backends.py`)
Factory for LLM instances:

```python
def get_llm(config: Config) -> BaseChatModel:
    if config.llm_backend == "claude":
        return ChatAnthropic(
            model=config.claude_model,
            anthropic_api_key=config.anthropic_api_key,
            temperature=0.3,
            max_tokens=4096,
        )
    elif config.llm_backend == "lmstudio":
        return ChatOpenAI(
            base_url=config.lmstudio_base_url,
            api_key="lm-studio",
            model=config.lmstudio_model,
            temperature=0.3,
        )
```

### Step 9: RAG prompts (`src/codetrading_rag/rag/prompts.py`)
Specialized system prompts for trading strategy generation:

- **Strategy generation prompt**: Instructs the LLM to generate a complete Python trading strategy using techniques from the retrieved context, including entry/exit signals, risk management, backtesting code
- **Technique explanation prompt**: Explains a concept/indicator based on how the channel teaches it
- **Code review prompt**: Reviews/improves user-provided strategy code using channel best practices

### Step 10: RAG chain (`src/codetrading_rag/rag/chain.py`)
LangChain retrieval chain using `create_retrieval_chain`:

```python
class RAGChain:
    def __init__(self, config: Config)
    def query(self, question: str, mode: str = "strategy") -> RAGResponse
    # mode: "strategy" | "explain" | "review"
```

Internally:
1. Gets retriever from VectorStore
2. Gets LLM from backends factory
3. Selects prompt template based on mode
4. Builds chain with `create_stuff_documents_chain` + `create_retrieval_chain`
5. Returns answer + source documents

### Step 11: CLI interface (`src/codetrading_rag/cli.py`)
Click-based CLI with rich output:

```
codetrading-rag ingest          # Fetch all videos & transcripts, build vector store
codetrading-rag ingest --limit 10  # Fetch only 10 most recent videos
codetrading-rag query "Create a mean-reversion strategy using Bollinger Bands"
codetrading-rag query --mode explain "How does Monte Carlo simulation work for backtesting?"
codetrading-rag query --mode review --file my_strategy.py
codetrading-rag config          # Show current configuration
codetrading-rag status          # Show ingestion stats (videos, chunks, etc.)
```

### Step 12: Tests
Basic tests for:
- Configuration loading
- Transcript processing and chunking
- Vector store operations
- LLM backend instantiation
- End-to-end RAG chain (with mocked LLM)
