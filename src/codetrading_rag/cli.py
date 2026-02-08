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
    """CodeTrading-RAG: Generate trading strategies from YouTube channel videos."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.from_env()


# ─── channel management ─────────────────────────────────────────────────────


@cli.group()
@click.pass_context
def channel(ctx: click.Context) -> None:
    """Manage YouTube channels."""
    pass


@channel.command("add")
@click.argument("name")
@click.argument("url")
@click.pass_context
def channel_add(ctx: click.Context, name: str, url: str) -> None:
    """Register a new YouTube channel.

    NAME is the display name, URL is the YouTube channel URL.
    """
    from codetrading_rag.channels.manager import ChannelManager

    config: Config = ctx.obj["config"]
    manager = ChannelManager(config.data_dir)

    try:
        info = manager.add_channel(name, url)
        console.print(f"\n[green]Added channel:[/] {info.name} (slug: {info.slug})")
        console.print(f"  URL: {info.url}")
        console.print(f"  Data dir: {manager.get_data_dir(info.slug)}\n")
    except ValueError as exc:
        console.print(f"\n[red]Error:[/] {exc}\n")
        sys.exit(1)


@channel.command("remove")
@click.argument("slug")
@click.pass_context
def channel_remove(ctx: click.Context, slug: str) -> None:
    """Unregister a channel by its slug. Does not delete data files."""
    from codetrading_rag.channels.manager import ChannelManager

    config: Config = ctx.obj["config"]
    manager = ChannelManager(config.data_dir)

    try:
        manager.remove_channel(slug)
        console.print(f"\n[green]Removed channel:[/] {slug}\n")
    except ValueError as exc:
        console.print(f"\n[red]Error:[/] {exc}\n")
        sys.exit(1)


@channel.command("list")
@click.pass_context
def channel_list(ctx: click.Context) -> None:
    """List all registered channels."""
    from codetrading_rag.channels.manager import ChannelManager

    config: Config = ctx.obj["config"]
    manager = ChannelManager(config.data_dir)

    channels = manager.channels
    if not channels:
        console.print("\n[yellow]No channels registered.[/] Use [cyan]codetrading-rag channel add[/] to add one.\n")
        return

    table = Table(title="Registered Channels", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Slug", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Videos", justify="right")
    table.add_column("Transcripts", justify="right")
    table.add_column("URL", style="blue", max_width=50)

    for ch in channels:
        table.add_row(
            ch.name,
            ch.slug,
            ch.status,
            str(ch.video_count),
            str(ch.transcript_count),
            ch.url,
        )

    console.print()
    console.print(table)
    console.print()


# ─── ingest ──────────────────────────────────────────────────────────────────


def _ingest_channel(
    slug: str,
    config: Config,
    limit: int | None,
    reindex: bool,
) -> None:
    """Run the ingest pipeline for a single channel."""
    from codetrading_rag.channels.manager import ChannelManager
    from codetrading_rag.embeddings.factory import get_embeddings
    from codetrading_rag.ingest.channel import fetch_channel_videos, load_all_metadata
    from codetrading_rag.ingest.processor import process_video
    from codetrading_rag.ingest.transcripts import IpBlockedError, fetch_transcript
    from codetrading_rag.vectorstore.store import VectorStore

    manager = ChannelManager(config.data_dir)
    ch = manager.get(slug)
    if not ch:
        console.print(f"[red]Channel '{slug}' not found.[/]")
        return

    data_dir = manager.get_data_dir(slug)
    manager.update_status(slug, "ingesting")

    console.print(f"\n[bold blue]Ingesting channel:[/] {ch.name} ({slug})\n")

    # Step 1: Fetch video metadata
    console.print("[bold blue]Step 1:[/] Fetching video metadata...\n")
    new_videos = fetch_channel_videos(
        channel_url=ch.url,
        data_dir=data_dir,
        limit=limit,
    )
    all_videos = load_all_metadata(data_dir)
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
            transcript = fetch_transcript(video.video_id, data_dir=data_dir, config=config)
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
                "  Wait 30-60 minutes, then run ingest again.\n"
            )
            break

    console.print(
        f"\n  Transcripts: [green]{len(transcripts)} fetched[/], "
        f"[yellow]{failed} unavailable[/]\n"
    )

    # Step 3: Process into documents
    console.print("[bold blue]Step 3:[/] Processing documents...\n")
    all_documents = []
    for video in videos_to_process:
        transcript = transcripts.get(video.video_id)
        docs = process_video(video, transcript, channel_id=slug, channel_name=ch.name)
        all_documents.extend(docs)

    console.print(f"  Total chunks: [green]{len(all_documents)}[/]\n")

    if not all_documents:
        console.print("[yellow]No documents to index.[/]")
        manager.update_status(slug, "error" if ip_blocked else "ready")
        return

    # Step 4: Build vector store (incremental or reindex)
    console.print("[bold blue]Step 4:[/] Updating vector store...\n")
    embeddings = get_embeddings(config)
    store = VectorStore(config, embeddings)

    if reindex:
        deleted = store.delete_by_channel(slug)
        console.print(f"  Deleted {deleted} existing documents for channel '{slug}'")

    store.add_documents(all_documents)
    console.print(
        f"  Vector store: [green]{store.document_count} total documents[/] "
        f"({store.get_channel_document_count(slug)} for {slug})\n"
    )

    # Update channel status
    manager.refresh_counts(slug)
    status = "ready" if not ip_blocked else "ready"
    manager.update_status(
        slug,
        status,
        last_ingested_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    )

    if ip_blocked:
        console.print(
            "[bold yellow]Partial ingestion complete.[/] "
            "Run ingest again later to fetch remaining transcripts.\n"
        )
    else:
        console.print(f"[bold green]Ingestion complete for {ch.name}![/]\n")


@cli.command()
@click.option("--channel", "channel_slug", type=str, default=None, help="Channel slug to ingest.")
@click.option("--all-channels", is_flag=True, help="Ingest all registered channels.")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of videos to process per channel (default: all).",
)
@click.option("--reindex", is_flag=True, help="Delete existing docs and re-index from scratch.")
@click.pass_context
def ingest(
    ctx: click.Context,
    channel_slug: str | None,
    all_channels: bool,
    limit: int | None,
    reindex: bool,
) -> None:
    """Fetch videos and transcripts, then build the vector store."""
    from codetrading_rag.channels.manager import ChannelManager

    config: Config = ctx.obj["config"]
    manager = ChannelManager(config.data_dir)

    if all_channels:
        slugs = manager.slugs
        if not slugs:
            console.print("\n[yellow]No channels registered.[/] Use [cyan]codetrading-rag channel add[/] first.\n")
            return
        for slug in slugs:
            _ingest_channel(slug, config, limit, reindex)
    elif channel_slug:
        _ingest_channel(channel_slug, config, limit, reindex)
    else:
        # Default: if only one channel exists, use it
        slugs = manager.slugs
        if len(slugs) == 1:
            _ingest_channel(slugs[0], config, limit, reindex)
        elif len(slugs) == 0:
            console.print("\n[yellow]No channels registered.[/] Use [cyan]codetrading-rag channel add[/] first.\n")
        else:
            console.print(
                "\n[yellow]Multiple channels registered.[/] "
                "Use [cyan]--channel SLUG[/] or [cyan]--all-channels[/].\n"
            )
            console.print("Available channels:")
            for slug in slugs:
                ch = manager.get(slug)
                console.print(f"  - {slug} ({ch.name if ch else 'unknown'})")
            console.print()


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
@click.option(
    "--channel",
    "channel_slugs",
    multiple=True,
    help="Filter to specific channel(s). Can be repeated.",
)
@click.pass_context
def query(
    ctx: click.Context,
    question: str,
    mode: str,
    file_path: str | None,
    channel_slugs: tuple[str, ...],
) -> None:
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
    channel_ids = list(channel_slugs) if channel_slugs else None

    console.print(f"\n[bold]Mode:[/] {mode}")
    if channel_ids:
        console.print(f"[bold]Channels:[/] {', '.join(channel_ids)}")
    console.print(f"[bold]Query:[/] {question[:100]}{'...' if len(question) > 100 else ''}\n")

    with console.status("[bold green]Thinking..."):
        response = chain.query(question, mode=mode, channel_ids=channel_ids)

    # Display answer
    console.print(Panel(Markdown(response.answer), title="Response", border_style="green"))

    # Display sources
    if response.source_documents:
        console.print("\n[bold]Sources:[/]")
        table = Table(show_header=True)
        table.add_column("Video", style="cyan", max_width=50)
        table.add_column("Channel", style="magenta")
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
                    meta.get("channel_name", ""),
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
    table.add_row("Data Directory", config.data_dir)
    table.add_row("Retriever K", str(config.retriever_k))
    table.add_row("GUI Port", str(config.gui_port))

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
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show ingestion status and statistics."""
    from codetrading_rag.channels.manager import ChannelManager

    config: Config = ctx.obj["config"]
    manager = ChannelManager(config.data_dir)
    channels = manager.channels

    if not channels:
        console.print("\n[yellow]No channels registered.[/]\n")
        return

    # Per-channel status
    table = Table(title="Channel Status", show_header=True)
    table.add_column("Channel", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Videos", justify="right")
    table.add_column("Transcripts", justify="right")
    table.add_column("Last Ingested", style="green")

    for ch in channels:
        manager.refresh_counts(ch.slug)
        ch = manager.get(ch.slug)  # Reload after refresh
        table.add_row(
            ch.name,
            ch.status,
            str(ch.video_count),
            str(ch.transcript_count),
            ch.last_ingested_at[:19] if ch.last_ingested_at else "never",
        )

    console.print()
    console.print(table)

    # Vector store stats
    n_vectors = 0
    chroma_path = Path(config.chroma_db_path)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        try:
            from codetrading_rag.embeddings.factory import get_embeddings
            from codetrading_rag.vectorstore.store import VectorStore

            embeddings = get_embeddings(config)
            store = VectorStore(config, embeddings)
            n_vectors = store.document_count

            # Per-channel vector counts
            console.print()
            vtable = Table(title="Vector Store", show_header=True)
            vtable.add_column("Channel", style="cyan")
            vtable.add_column("Chunks", justify="right", style="green")
            for ch in channels:
                count = store.get_channel_document_count(ch.slug)
                vtable.add_row(ch.name, str(count))
            vtable.add_row("[bold]Total[/]", f"[bold]{n_vectors}[/]")
            console.print(vtable)
        except Exception:
            console.print(f"\n  Vector store: [red]error reading[/]")
    else:
        console.print(f"\n  Vector store: [yellow]empty[/]")

    console.print()


# ─── migrate ─────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def migrate(ctx: click.Context) -> None:
    """Migrate existing single-channel data to multi-channel format."""
    from codetrading_rag.migration import migrate_to_multichannel

    config: Config = ctx.obj["config"]
    migrate_to_multichannel(Path(config.data_dir), config)


# ─── gui ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--port", type=int, default=None, help="Port number (default: from config).")
@click.option("--share", is_flag=True, help="Create a public Gradio link.")
@click.pass_context
def gui(ctx: click.Context, port: int | None, share: bool) -> None:
    """Launch the Gradio web interface."""
    from codetrading_rag.gui.app import create_app

    config: Config = ctx.obj["config"]
    app = create_app(config)
    app.launch(
        server_port=port or config.gui_port,
        share=share or config.gui_share,
    )


if __name__ == "__main__":
    cli()
