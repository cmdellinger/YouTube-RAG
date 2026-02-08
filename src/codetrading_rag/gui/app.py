"""Gradio-based web GUI for CodeTrading-RAG."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from codetrading_rag.config import Config

logger = logging.getLogger(__name__)


def create_app(config: Config) -> gr.Blocks:
    """Create the Gradio Blocks application.

    Args:
        config: Application configuration.

    Returns:
        A Gradio Blocks app ready to launch.
    """
    from codetrading_rag.channels.manager import ChannelManager

    manager = ChannelManager(config.data_dir)

    # ─── Helper functions ────────────────────────────────────────────────

    def _get_channels_df() -> list[list[str]]:
        """Return channel data as a list of rows for the dataframe."""
        mgr = ChannelManager(config.data_dir)
        rows = []
        for ch in mgr.channels:
            mgr.refresh_counts(ch.slug)
            ch = mgr.get(ch.slug)
            rows.append([
                ch.name,
                ch.slug,
                ch.url,
                ch.status,
                str(ch.video_count),
                str(ch.transcript_count),
                ch.last_ingested_at[:19] if ch.last_ingested_at else "never",
            ])
        return rows

    def _get_channel_choices() -> list[str]:
        """Return list of channel slugs for selection components."""
        mgr = ChannelManager(config.data_dir)
        return [f"{ch.name} ({ch.slug})" for ch in mgr.channels]

    def _get_channel_slugs() -> list[str]:
        """Return list of channel slugs."""
        mgr = ChannelManager(config.data_dir)
        return mgr.slugs

    def _slug_from_choice(choice: str) -> str:
        """Extract slug from a choice string like 'Name (slug)'."""
        if "(" in choice and choice.endswith(")"):
            return choice.rsplit("(", 1)[1].rstrip(")")
        return choice

    # ─── Callbacks ───────────────────────────────────────────────────────

    def add_channel_cb(name: str, url: str):
        """Add a new channel."""
        if not name or not url:
            return (
                _get_channels_df(),
                gr.update(choices=_get_channel_choices()),
                gr.update(choices=_get_channel_choices()),
                gr.update(choices=_get_channel_slugs()),
                "Please enter both a channel name and URL.",
            )
        mgr = ChannelManager(config.data_dir)
        try:
            info = mgr.add_channel(name, url)
            msg = f"Added channel: {info.name} (slug: {info.slug})"
        except ValueError as exc:
            msg = f"Error: {exc}"
        return (
            _get_channels_df(),
            gr.update(choices=_get_channel_choices()),
            gr.update(choices=_get_channel_choices()),
            gr.update(choices=_get_channel_slugs()),
            msg,
        )

    def remove_channel_cb(selection: str):
        """Remove a channel."""
        if not selection:
            return (
                _get_channels_df(),
                gr.update(choices=_get_channel_choices()),
                gr.update(choices=_get_channel_choices()),
                gr.update(choices=_get_channel_slugs()),
                "Please select a channel to remove.",
            )
        slug = _slug_from_choice(selection)
        mgr = ChannelManager(config.data_dir)
        try:
            mgr.remove_channel(slug)
            msg = f"Removed channel: {slug}"
        except ValueError as exc:
            msg = f"Error: {exc}"
        return (
            _get_channels_df(),
            gr.update(choices=_get_channel_choices()),
            gr.update(choices=_get_channel_choices()),
            gr.update(choices=_get_channel_slugs()),
            msg,
        )

    def refresh_cb():
        """Refresh channel data."""
        return (
            _get_channels_df(),
            gr.update(choices=_get_channel_choices()),
            gr.update(choices=_get_channel_choices()),
            gr.update(choices=_get_channel_slugs()),
        )

    def ingest_cb(selected_channels: list[str], limit: int | None, reindex: bool):
        """Run ingestion for selected channels. Yields progress updates."""
        if not selected_channels:
            yield "Please select at least one channel to ingest.", _get_channels_df()
            return

        from codetrading_rag.channels.manager import ChannelManager as CM
        from codetrading_rag.embeddings.factory import get_embeddings
        from codetrading_rag.ingest.channel import fetch_channel_videos, load_all_metadata
        from codetrading_rag.ingest.processor import process_video
        from codetrading_rag.ingest.transcripts import IpBlockedError, fetch_transcript
        from codetrading_rag.vectorstore.store import VectorStore

        log_lines: list[str] = []

        def log(msg: str) -> str:
            log_lines.append(msg)
            return "\n".join(log_lines)

        limit_val = int(limit) if limit else None

        for channel_choice in selected_channels:
            slug = _slug_from_choice(channel_choice)
            mgr = CM(config.data_dir)
            ch = mgr.get(slug)
            if not ch:
                yield log(f"Channel '{slug}' not found, skipping."), _get_channels_df()
                continue

            data_dir = mgr.get_data_dir(slug)
            mgr.update_status(slug, "ingesting")

            yield log(f"\n--- Ingesting: {ch.name} ({slug}) ---"), _get_channels_df()

            # Step 1: Fetch metadata
            yield log("Step 1: Fetching video metadata..."), _get_channels_df()
            try:
                new_videos = fetch_channel_videos(
                    channel_url=ch.url,
                    data_dir=data_dir,
                    limit=limit_val,
                )
                all_videos = load_all_metadata(data_dir)
                yield log(f"  Videos: {len(new_videos)} new, {len(all_videos)} total"), _get_channels_df()
            except Exception as exc:
                yield log(f"  Error fetching metadata: {exc}"), _get_channels_df()
                mgr.update_status(slug, "error")
                continue

            videos_to_process = all_videos[:limit_val] if limit_val else all_videos

            # Step 2: Fetch transcripts
            yield log(f"Step 2: Fetching transcripts for {len(videos_to_process)} videos..."), _get_channels_df()
            transcripts = {}
            failed = 0
            ip_blocked = False
            for i, video in enumerate(videos_to_process, 1):
                try:
                    transcript = fetch_transcript(video.video_id, data_dir=data_dir, config=config)
                    if transcript:
                        transcripts[video.video_id] = transcript
                    else:
                        failed += 1
                except IpBlockedError:
                    failed += 1
                    ip_blocked = True
                    yield log(f"  IP BLOCKED at video {i}. Stopping transcript fetch."), _get_channels_df()
                    break
                except Exception:
                    failed += 1

                if i % 10 == 0 or i == len(videos_to_process):
                    yield log(f"  Progress: {i}/{len(videos_to_process)} ({len(transcripts)} transcripts)"), _get_channels_df()

            yield log(f"  Transcripts: {len(transcripts)} fetched, {failed} unavailable"), _get_channels_df()

            # Step 3: Process documents
            yield log("Step 3: Processing documents..."), _get_channels_df()
            all_documents = []
            for video in videos_to_process:
                transcript = transcripts.get(video.video_id)
                docs = process_video(video, transcript, channel_id=slug, channel_name=ch.name)
                all_documents.extend(docs)

            yield log(f"  Total chunks: {len(all_documents)}"), _get_channels_df()

            if not all_documents:
                yield log("  No documents to index."), _get_channels_df()
                mgr.update_status(slug, "ready")
                continue

            # Step 4: Update vector store
            yield log("Step 4: Updating vector store..."), _get_channels_df()
            embeddings = get_embeddings(config)
            store = VectorStore(config, embeddings)

            if reindex:
                deleted = store.delete_by_channel(slug)
                yield log(f"  Deleted {deleted} existing documents"), _get_channels_df()

            store.add_documents(all_documents)
            total = store.document_count
            channel_total = store.get_channel_document_count(slug)
            yield log(f"  Vector store: {total} total ({channel_total} for {slug})"), _get_channels_df()

            # Update status
            mgr.refresh_counts(slug)
            mgr.update_status(
                slug,
                "ready",
                last_ingested_at=datetime.now(timezone.utc).isoformat(),
            )

            status_msg = "PARTIAL (IP blocked)" if ip_blocked else "COMPLETE"
            yield log(f"  Ingestion {status_msg} for {ch.name}"), _get_channels_df()

        yield log("\n--- All done! ---"), _get_channels_df()

    def query_cb(
        question: str,
        mode: str,
        channel_filter: list[str],
        code_file: str | None,
    ):
        """Run a RAG query."""
        if not question.strip():
            return "Please enter a question.", []

        issues = config.validate()
        if issues:
            return "Config errors:\n" + "\n".join(f"- {i}" for i in issues), []

        from codetrading_rag.rag.chain import RAGChain

        # If review mode with uploaded file, prepend code
        if code_file and mode == "review":
            code = Path(code_file).read_text()
            question = f"Please review this code:\n\n```python\n{code}\n```\n\n{question}"

        channel_ids = [_slug_from_choice(c) for c in channel_filter] if channel_filter else None

        chain = RAGChain(config)
        response = chain.query(question, mode=mode, channel_ids=channel_ids)

        # Build sources table
        sources = []
        seen = set()
        for doc in response.source_documents:
            meta = doc.metadata
            key = (meta.get("video_id"), meta.get("chunk_type"))
            if key not in seen:
                seen.add(key)
                sources.append([
                    meta.get("video_title", "Unknown"),
                    meta.get("channel_name", ""),
                    meta.get("chunk_type", "unknown"),
                    meta.get("video_url", ""),
                ])

        return response.answer, sources

    # ─── Build the UI ────────────────────────────────────────────────────

    with gr.Blocks(
        title="CodeTrading-RAG",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# CodeTrading-RAG")
        gr.Markdown("Multi-channel YouTube RAG system for trading strategies")

        # ─── Tab 1: Channel Management ───────────────────────────────────

        with gr.Tab("Channels"):
            channels_table = gr.Dataframe(
                headers=["Name", "Slug", "URL", "Status", "Videos", "Transcripts", "Last Ingested"],
                value=_get_channels_df(),
                interactive=False,
                label="Registered Channels",
            )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Add Channel")
                    ch_name_input = gr.Textbox(label="Channel Name", placeholder="e.g. CodeTradingCafe")
                    ch_url_input = gr.Textbox(label="Channel URL", placeholder="e.g. https://www.youtube.com/@CodeTradingCafe")
                    add_btn = gr.Button("Add Channel", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Remove Channel")
                    remove_dropdown = gr.Dropdown(
                        choices=_get_channel_choices(),
                        label="Select Channel",
                        interactive=True,
                    )
                    remove_btn = gr.Button("Remove Channel", variant="stop")

            channel_msg = gr.Textbox(label="Status", interactive=False)
            refresh_btn = gr.Button("Refresh")

        # ─── Tab 2: Ingestion ────────────────────────────────────────────

        with gr.Tab("Ingest"):
            ingest_channels = gr.CheckboxGroup(
                choices=_get_channel_choices(),
                label="Select Channels to Ingest",
            )
            with gr.Row():
                ingest_limit = gr.Number(
                    label="Video Limit (0 = all)",
                    value=0,
                    precision=0,
                )
                ingest_reindex = gr.Checkbox(
                    label="Re-index (delete existing docs first)",
                    value=False,
                )
            ingest_btn = gr.Button("Start Ingestion", variant="primary")
            ingest_log = gr.Textbox(
                label="Ingestion Log",
                lines=20,
                max_lines=40,
                interactive=False,
            )
            ingest_status_table = gr.Dataframe(
                headers=["Name", "Slug", "URL", "Status", "Videos", "Transcripts", "Last Ingested"],
                value=_get_channels_df(),
                interactive=False,
                label="Channel Status",
            )

        # ─── Tab 3: Query ────────────────────────────────────────────────

        with gr.Tab("Query"):
            with gr.Row():
                query_channels = gr.CheckboxGroup(
                    choices=_get_channel_slugs(),
                    label="Filter by Channel (empty = all)",
                )
                query_mode = gr.Radio(
                    choices=["strategy", "explain", "review"],
                    value="strategy",
                    label="Query Mode",
                )

            query_input = gr.Textbox(
                label="Question",
                placeholder="e.g. Create a mean reversion strategy using Bollinger Bands",
                lines=3,
            )
            query_file = gr.File(
                label="Code File (for review mode)",
                file_types=[".py", ".txt"],
                type="filepath",
            )
            query_btn = gr.Button("Ask", variant="primary")

            query_output = gr.Markdown(label="Response")
            query_sources = gr.Dataframe(
                headers=["Video", "Channel", "Type", "URL"],
                label="Sources",
                interactive=False,
            )

        # ─── Wire up events ──────────────────────────────────────────────

        # Channel management events
        add_outputs = [channels_table, remove_dropdown, ingest_channels, query_channels, channel_msg]
        add_btn.click(
            add_channel_cb,
            inputs=[ch_name_input, ch_url_input],
            outputs=add_outputs,
        )
        remove_btn.click(
            remove_channel_cb,
            inputs=[remove_dropdown],
            outputs=add_outputs,
        )
        refresh_btn.click(
            refresh_cb,
            outputs=[channels_table, remove_dropdown, ingest_channels, query_channels],
        )

        # Ingestion events
        ingest_btn.click(
            ingest_cb,
            inputs=[ingest_channels, ingest_limit, ingest_reindex],
            outputs=[ingest_log, ingest_status_table],
        )

        # Query events
        query_btn.click(
            query_cb,
            inputs=[query_input, query_mode, query_channels, query_file],
            outputs=[query_output, query_sources],
        )

    return app
