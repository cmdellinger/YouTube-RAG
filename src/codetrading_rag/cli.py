"""CLI interface for CodeTrading-RAG."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from codetrading_rag.config import Config

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose/debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CodeTrading-RAG: Generate trading strategies from CodeTradingCafe videos."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.from_env()


# ─── ingest ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of videos to process (default: all).",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data",
    help="Data directory for transcripts and metadata.",
)
@click.pass_context
def ingest(ctx: click.Context, limit: int | None, data_dir: str) -> None:
    """Fetch videos and transcripts, then build the vector store."""
    config: Config = ctx.obj["config"]
    data_path = Path(data_dir)

    # Lazy imports to speed up CLI startup
    from codetrading_rag.embeddings.factory import get_embeddings
    from codetrading_rag.ingest.channel import fetch_channel_videos, load_all_metadata
    from codetrading_rag.ingest.processor import process_video
    from codetrading_rag.ingest.transcripts import IpBlockedError, fetch_transcript
    from codetrading_rag.vectorstore.store import VectorStore

    # Step 1: Fetch video metadata
    console.print("\n[bold blue]Step 1:[/] Fetching video metadata...\n")
    new_videos = fetch_channel_videos(
        channel_url=config.channel_url,
        data_dir=data_path,
        limit=limit,
    )
    all_videos = load_all_metadata(data_path)
    console.print(f"  Videos: [green]{len(new_videos)} new[/], {len(all_videos)} total\n")

    # Apply --limit to the total number of videos to process
    videos_to_process = all_videos[:limit] if limit else all_videos

    # Step 2: Fetch transcripts
    console.print(f"[bold blue]Step 2:[/] Fetching transcripts for {len(videos_to_process)} videos...\n")
    transcripts = {}
    failed = 0
    ip_blocked = False
    for i, video in enumerate(videos_to_process, 1):
        console.print(
            f"  [{i}/{len(videos_to_process)}] {video.title[:60]}...",
            end="",
        )
        try:
            transcript = fetch_transcript(video.video_id, data_dir=data_path, config=config)
            if transcript:
                transcripts[video.video_id] = transcript
                console.print(f" [green]{len(transcript.segments)} segments[/]")
            else:
                failed += 1
                console.print(" [yellow]no transcript[/]")
        except IpBlockedError:
            failed += 1
            ip_blocked = True
            console.print(" [bold red]IP BLOCKED[/]")
            console.print(
                "\n  [bold red]YouTube is blocking your IP.[/] "
                "Stopping transcript fetching to avoid making it worse.\n"
                "  Already-fetched transcripts will still be processed.\n"
                "  Wait 30-60 minutes, then run [cyan]codetrading-rag ingest[/] again.\n"
            )
            break

    console.print(
        f"\n  Transcripts: [green]{len(transcripts)} fetched[/], "
        f"[yellow]{failed} unavailable[/]\n"
    )

    # Step 3: Process into documents (use ALL videos that have transcripts/metadata)
    console.print("[bold blue]Step 3:[/] Processing documents...\n")
    all_documents = []
    for video in videos_to_process:
        transcript = transcripts.get(video.video_id)
        docs = process_video(video, transcript)
        all_documents.extend(docs)

    console.print(f"  Total chunks: [green]{len(all_documents)}[/]\n")

    if not all_documents:
        console.print("[yellow]No documents to index. Exiting.[/]")
        return

    # Step 4: Build vector store
    console.print("[bold blue]Step 4:[/] Building vector store...\n")
    embeddings = get_embeddings(config)
    store = VectorStore(config, embeddings)
    store.create_from_documents(all_documents)
    console.print(
        f"  Vector store: [green]{store.document_count} documents indexed[/] "
        f"at {config.chroma_db_path}\n"
    )

    if ip_blocked:
        console.print(
            "[bold yellow]Partial ingestion complete.[/] "
            "Run [cyan]codetrading-rag ingest[/] again later to fetch remaining transcripts.\n"
        )
    else:
        console.print("[bold green]Ingestion complete![/]\n")


# ─── query ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("question")
@click.option(
    "--mode",
    type=click.Choice(["strategy", "explain", "review"]),
    default="strategy",
    help="Query mode (default: strategy).",
)
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a code file (for review mode).",
)
@click.pass_context
def query(ctx: click.Context, question: str, mode: str, file_path: str | None) -> None:
    """Query the RAG system with a question or strategy request."""
    config: Config = ctx.obj["config"]

    issues = config.validate()
    if issues:
        for issue in issues:
            console.print(f"[red]Config error:[/] {issue}")
        sys.exit(1)

    from codetrading_rag.rag.chain import RAGChain

    # If review mode with a file, prepend file contents to the question
    if file_path and mode == "review":
        code = Path(file_path).read_text()
        question = f"Please review this code:\n\n```python\n{code}\n```\n\n{question}"

    chain = RAGChain(config)
    console.print(f"\n[bold]Mode:[/] {mode}")
    console.print(f"[bold]Query:[/] {question[:100]}{'...' if len(question) > 100 else ''}\n")

    with console.status("[bold green]Thinking..."):
        response = chain.query(question, mode=mode)

    # Display answer
    console.print(Panel(Markdown(response.answer), title="Response", border_style="green"))

    # Display sources
    if response.source_documents:
        console.print("\n[bold]Sources:[/]")
        table = Table(show_header=True)
        table.add_column("Video", style="cyan", max_width=50)
        table.add_column("Type", style="green")
        table.add_column("URL", style="blue")

        seen = set()
        for doc in response.source_documents:
            meta = doc.metadata
            key = (meta.get("video_id"), meta.get("chunk_type"))
            if key not in seen:
                seen.add(key)
                table.add_row(
                    meta.get("video_title", "Unknown"),
                    meta.get("chunk_type", "unknown"),
                    meta.get("video_url", ""),
                )

        console.print(table)
        console.print()


# ─── config ──────────────────────────────────────────────────────────────────


@cli.command("config")
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Show current configuration."""
    config: Config = ctx.obj["config"]

    table = Table(title="CodeTrading-RAG Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("LLM Backend", config.llm_backend)
    table.add_row(
        "Claude Model",
        config.claude_model if config.llm_backend == "claude" else "(not active)",
    )
    table.add_row(
        "Anthropic API Key",
        "***" + config.anthropic_api_key[-4:] if config.anthropic_api_key else "(not set)",
    )
    table.add_row(
        "LM Studio URL",
        config.lmstudio_base_url if config.llm_backend == "lmstudio" else "(not active)",
    )
    table.add_row(
        "LM Studio Model",
        config.lmstudio_model or "(not set)",
    )
    table.add_row("Embedding Backend", config.embedding_backend)
    table.add_row("HF Model", config.hf_embedding_model)
    table.add_row("ChromaDB Path", config.chroma_db_path)
    table.add_row("Channel URL", config.channel_url)
    table.add_row("Retriever K", str(config.retriever_k))

    console.print()
    console.print(table)

    issues = config.validate()
    if issues:
        console.print()
        for issue in issues:
            console.print(f"[yellow]Warning:[/] {issue}")
    console.print()


# ─── status ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data",
    help="Data directory.",
)
@click.pass_context
def status(ctx: click.Context, data_dir: str) -> None:
    """Show ingestion status and statistics."""
    config: Config = ctx.obj["config"]
    data_path = Path(data_dir)

    table = Table(title="Ingestion Status", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    # Count metadata files
    metadata_dir = data_path / "metadata"
    n_metadata = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0

    # Count transcript files
    transcript_dir = data_path / "transcripts"
    n_transcripts = len(list(transcript_dir.glob("*.json"))) if transcript_dir.exists() else 0

    # Count vector store documents
    n_vectors = 0
    chroma_path = Path(config.chroma_db_path)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        try:
            from codetrading_rag.embeddings.factory import get_embeddings
            from codetrading_rag.vectorstore.store import VectorStore

            embeddings = get_embeddings(config)
            store = VectorStore(config, embeddings)
            n_vectors = store.document_count
        except Exception:
            n_vectors = -1  # Error reading

    table.add_row("Videos (metadata)", str(n_metadata))
    table.add_row("Transcripts", str(n_transcripts))
    table.add_row(
        "Vector store chunks",
        str(n_vectors) if n_vectors >= 0 else "[red]error reading[/]",
    )
    table.add_row("ChromaDB path", config.chroma_db_path)

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    cli()
