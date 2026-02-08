"""Document processor: converts raw transcripts + metadata into LangChain Documents."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .channel import VideoMetadata
from .transcripts import CodeBlock, FetchedTranscript, extract_code_from_description

logger = logging.getLogger(__name__)

# Splitter for transcript text
TRANSCRIPT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Splitter for code blocks
CODE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\ndef ", "\nclass ", "\n\n", "\n", " ", ""],
)


def _base_metadata(
    meta: VideoMetadata,
    channel_id: str = "",
    channel_name: str = "",
) -> dict:
    """Return shared metadata fields for all chunks from a video."""
    return {
        "video_id": meta.video_id,
        "video_title": meta.title,
        "video_url": meta.url,
        "upload_date": meta.upload_date,
        "channel_id": channel_id,
        "channel_name": channel_name,
    }


def _process_transcript(
    meta: VideoMetadata,
    transcript: FetchedTranscript,
    channel_id: str = "",
    channel_name: str = "",
) -> list[Document]:
    """Split transcript into chunked Documents with timing metadata."""
    if not transcript.segments:
        return []

    # Build the full text and track segment boundaries
    full_text = transcript.full_text

    # Chunk the text
    chunks = TRANSCRIPT_SPLITTER.split_text(full_text)
    documents: list[Document] = []

    # Map chunks back to approximate timestamps
    seg_char_positions: list[tuple[float, float, int]] = []  # (start, end_time, char_end)

    running_pos = 0
    for seg in transcript.segments:
        seg_end = running_pos + len(seg.text) + 1  # +1 for space
        seg_char_positions.append((seg.start, seg.start + seg.duration, seg_end))
        running_pos = seg_end

    for chunk in chunks:
        base = _base_metadata(meta, channel_id, channel_name)
        base["chunk_type"] = "transcript"

        # Find approximate start/end times for this chunk
        chunk_start_char = full_text.find(chunk[:50]) if len(chunk) >= 50 else full_text.find(chunk)
        chunk_end_char = chunk_start_char + len(chunk) if chunk_start_char >= 0 else -1

        start_time = None
        end_time = None
        if chunk_start_char >= 0:
            for seg_start, seg_end_t, seg_char_end in seg_char_positions:
                if seg_char_end >= chunk_start_char and start_time is None:
                    start_time = seg_start
                if seg_char_end >= chunk_end_char:
                    end_time = seg_end_t
                    break

        base["start_time"] = start_time
        base["end_time"] = end_time

        documents.append(Document(page_content=chunk, metadata=base))

    return documents


def _process_description(
    meta: VideoMetadata,
    channel_id: str = "",
    channel_name: str = "",
) -> list[Document]:
    """Create a Document from the video description (if substantial)."""
    desc = meta.description.strip()
    if len(desc) < 50:
        return []

    base = _base_metadata(meta, channel_id, channel_name)
    base["chunk_type"] = "description"
    base["start_time"] = None
    base["end_time"] = None

    return [Document(page_content=desc, metadata=base)]


def _process_code_blocks(
    meta: VideoMetadata,
    code_blocks: list[CodeBlock],
    channel_id: str = "",
    channel_name: str = "",
) -> list[Document]:
    """Split code blocks into chunked Documents."""
    documents: list[Document] = []

    for block in code_blocks:
        base = _base_metadata(meta, channel_id, channel_name)
        base["chunk_type"] = "code"
        base["code_language"] = block.language
        base["start_time"] = None
        base["end_time"] = None

        # Split long code blocks
        if len(block.code) > 500:
            chunks = CODE_SPLITTER.split_text(block.code)
            for chunk in chunks:
                doc_meta = dict(base)
                documents.append(Document(page_content=chunk, metadata=doc_meta))
        else:
            documents.append(Document(page_content=block.code, metadata=base))

    return documents


def process_video(
    meta: VideoMetadata,
    transcript: FetchedTranscript | None,
    channel_id: str = "",
    channel_name: str = "",
) -> list[Document]:
    """Process a single video into a list of LangChain Documents.

    This creates documents from:
    1. Transcript text (chunked)
    2. Video description (if substantial)
    3. Code blocks extracted from description

    Args:
        meta: Video metadata.
        transcript: Fetched transcript (can be None if unavailable).
        channel_id: Channel slug for metadata tagging.
        channel_name: Channel display name for metadata tagging.

    Returns:
        List of LangChain Document objects ready for embedding.
    """
    documents: list[Document] = []

    # Process transcript
    if transcript:
        transcript_docs = _process_transcript(meta, transcript, channel_id, channel_name)
        documents.extend(transcript_docs)
        logger.info("  %d transcript chunks from '%s'", len(transcript_docs), meta.title)

    # Process description
    desc_docs = _process_description(meta, channel_id, channel_name)
    documents.extend(desc_docs)

    # Process code blocks from description
    code_blocks = extract_code_from_description(meta.description)
    if code_blocks:
        code_docs = _process_code_blocks(meta, code_blocks, channel_id, channel_name)
        documents.extend(code_docs)
        logger.info("  %d code chunks from '%s'", len(code_docs), meta.title)

    return documents
