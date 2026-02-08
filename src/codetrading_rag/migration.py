"""Migration from single-channel to multi-channel data format."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from codetrading_rag.config import Config

console = Console()

# Default channel info for existing CodeTradingCafe data
DEFAULT_CHANNEL_NAME = "CodeTradingCafe"
DEFAULT_CHANNEL_URL = "https://www.youtube.com/@CodeTradingCafe"


def migrate_to_multichannel(data_dir: Path, config: Config) -> None:
    """Migrate existing single-channel data to the multi-channel directory structure.

    Moves:
        data/metadata/*.json -> data/channels/codetradingcafe/metadata/*.json
        data/transcripts/*.json -> data/channels/codetradingcafe/transcripts/*.json

    Creates:
        data/channels.json with CodeTradingCafe entry

    Deletes:
        data/chroma_db/ (must be re-indexed with channel_id metadata)
    """
    old_metadata = data_dir / "metadata"
    old_transcripts = data_dir / "transcripts"

    # Check if migration is needed
    if not old_metadata.exists() and not old_transcripts.exists():
        console.print("\n[yellow]No existing data to migrate.[/]")
        console.print("Use [cyan]codetrading-rag channel add[/] to register a channel.\n")
        return

    # Check if already migrated
    channels_dir = data_dir / "channels" / "codetradingcafe"
    if channels_dir.exists() and any(channels_dir.iterdir()):
        console.print("\n[yellow]Data appears to already be in multi-channel format.[/]\n")
        return

    console.print("\n[bold blue]Migrating to multi-channel format...[/]\n")

    # Create channel directories
    new_metadata = channels_dir / "metadata"
    new_transcripts = channels_dir / "transcripts"
    new_metadata.mkdir(parents=True, exist_ok=True)
    new_transcripts.mkdir(parents=True, exist_ok=True)

    # Move metadata files
    meta_count = 0
    if old_metadata.exists():
        for f in old_metadata.glob("*.json"):
            shutil.move(str(f), str(new_metadata / f.name))
            meta_count += 1
        console.print(f"  Moved [green]{meta_count}[/] metadata files")

        # Remove old directory if empty
        if not list(old_metadata.iterdir()):
            old_metadata.rmdir()

    # Move transcript files
    trans_count = 0
    if old_transcripts.exists():
        for f in old_transcripts.glob("*.json"):
            shutil.move(str(f), str(new_transcripts / f.name))
            trans_count += 1
        console.print(f"  Moved [green]{trans_count}[/] transcript files")

        # Remove old directory if empty
        if not list(old_transcripts.iterdir()):
            old_transcripts.rmdir()

    # Create channel registry
    from codetrading_rag.channels.manager import ChannelManager

    manager = ChannelManager(data_dir)
    try:
        manager.add_channel(DEFAULT_CHANNEL_NAME, DEFAULT_CHANNEL_URL)
    except ValueError:
        pass  # Already registered

    manager.update_status(
        "codetradingcafe",
        "ready",
        video_count=meta_count,
        transcript_count=trans_count,
    )
    console.print(f"  Created channel registry with [green]{DEFAULT_CHANNEL_NAME}[/]")

    # Delete old ChromaDB (needs re-indexing with channel_id metadata)
    chroma_path = Path(config.chroma_db_path)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        shutil.rmtree(chroma_path)
        console.print(f"  Deleted old ChromaDB at [yellow]{chroma_path}[/]")

    console.print("\n[bold green]Migration complete![/]")
    console.print(
        "\nRun [cyan]codetrading-rag ingest --channel codetradingcafe --reindex[/] "
        "to rebuild the vector index with channel metadata.\n"
    )
