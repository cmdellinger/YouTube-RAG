"""Tests for the ingestion pipeline."""

import json
import tempfile
import unittest.mock
from pathlib import Path

import pytest

from codetrading_rag.ingest.channel import (
    VideoMetadata,
    fetch_single_video,
    parse_video_id,
)
from codetrading_rag.ingest.processor import process_video
from codetrading_rag.ingest.transcripts import (
    CodeBlock,
    FetchedTranscript,
    TranscriptSegment,
    extract_code_from_description,
)


class TestTranscripts:
    """Tests for transcript processing."""

    def test_extract_fenced_python(self):
        """Should extract fenced Python code blocks."""
        desc = "Some text\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```\nMore text"
        blocks = extract_code_from_description(desc)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "pandas" in blocks[0].code

    def test_extract_fenced_pinescript(self):
        """Should extract fenced Pine Script code blocks."""
        desc = "Check this out:\n```pinescript\n//@version=5\nindicator('Test')\n```\n"
        blocks = extract_code_from_description(desc)
        assert len(blocks) == 1
        assert blocks[0].language == "pinescript"

    def test_extract_multiple_blocks(self):
        """Should extract multiple code blocks."""
        desc = "```python\nprint('a')\n```\ntext\n```python\nprint('b')\n```"
        blocks = extract_code_from_description(desc)
        assert len(blocks) == 2

    def test_no_code_blocks(self):
        """Should return empty list when no code blocks exist."""
        desc = "This is just a regular video description with no code."
        blocks = extract_code_from_description(desc)
        assert blocks == []

    def test_fetched_transcript_full_text(self):
        """FetchedTranscript.full_text should join segments."""
        transcript = FetchedTranscript(
            video_id="test123",
            segments=[
                TranscriptSegment(text="Hello world", start=0.0, duration=2.0),
                TranscriptSegment(text="this is a test", start=2.0, duration=3.0),
            ],
        )
        assert transcript.full_text == "Hello world this is a test"

    def test_fetched_transcript_roundtrip(self):
        """FetchedTranscript should survive dict serialization roundtrip."""
        original = FetchedTranscript(
            video_id="abc",
            segments=[TranscriptSegment(text="hi", start=1.0, duration=2.0)],
            language="en",
        )
        restored = FetchedTranscript.from_dict(original.to_dict())
        assert restored.video_id == original.video_id
        assert len(restored.segments) == 1
        assert restored.segments[0].text == "hi"


class TestVideoMetadata:
    """Tests for VideoMetadata."""

    def test_roundtrip(self):
        """VideoMetadata should survive dict serialization roundtrip."""
        meta = VideoMetadata(
            video_id="v123",
            title="Test Video",
            description="A test",
            upload_date="20240101",
            duration=300,
            url="https://youtube.com/watch?v=v123",
        )
        restored = VideoMetadata.from_dict(meta.to_dict())
        assert restored.video_id == meta.video_id
        assert restored.title == meta.title

    def test_unlisted_default_false(self):
        """VideoMetadata should default unlisted to False."""
        meta = VideoMetadata(
            video_id="v1", title="T", description="D",
            upload_date="20240101", duration=60,
            url="https://youtube.com/watch?v=v1",
        )
        assert meta.unlisted is False

    def test_unlisted_roundtrip(self):
        """VideoMetadata should preserve unlisted=True through serialization."""
        meta = VideoMetadata(
            video_id="v1", title="T", description="D",
            upload_date="20240101", duration=60,
            url="https://youtube.com/watch?v=v1",
            unlisted=True,
        )
        restored = VideoMetadata.from_dict(meta.to_dict())
        assert restored.unlisted is True

    def test_from_dict_without_unlisted(self):
        """Existing dicts without 'unlisted' key should default to False."""
        data = {
            "video_id": "v1", "title": "T", "description": "D",
            "upload_date": "20240101", "duration": 60,
            "url": "https://youtube.com/watch?v=v1",
        }
        meta = VideoMetadata.from_dict(data)
        assert meta.unlisted is False


class TestParseVideoId:
    """Tests for parse_video_id."""

    def test_raw_video_id(self):
        assert parse_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_raw_id_with_whitespace(self):
        assert parse_video_id("  dQw4w9WgXcQ  ") == "dQw4w9WgXcQ"

    def test_standard_url(self):
        assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert parse_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_live_url(self):
        assert parse_video_id("https://www.youtube.com/live/sBNX8A82vzw") == "sBNX8A82vzw"

    def test_shorts_url(self):
        assert parse_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx") == "dQw4w9WgXcQ"

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            parse_video_id("not-a-valid-url-or-id")


class TestFetchSingleVideo:
    """Tests for fetch_single_video."""

    @unittest.mock.patch("codetrading_rag.ingest.channel.yt_dlp.YoutubeDL")
    def test_fetch_new_video(self, mock_ydl_cls):
        """Should fetch metadata via yt-dlp and save with unlisted=True."""
        mock_ydl = mock_ydl_cls.return_value.__enter__.return_value
        mock_ydl.extract_info.return_value = {
            "id": "test12345ab",
            "title": "Test Video",
            "description": "A test description",
            "upload_date": "20240101",
            "duration": 300,
            "webpage_url": "https://www.youtube.com/watch?v=test12345ab",
        }

        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "metadata").mkdir(parents=True)

        meta = fetch_single_video("test12345ab", tmpdir, unlisted=True)

        assert meta.video_id == "test12345ab"
        assert meta.title == "Test Video"
        assert meta.unlisted is True

        saved_path = tmpdir / "metadata" / "test12345ab.json"
        assert saved_path.exists()
        saved_data = json.loads(saved_path.read_text())
        assert saved_data["unlisted"] is True

    def test_fetch_existing_video(self):
        """Should return existing metadata and update unlisted flag."""
        tmpdir = Path(tempfile.mkdtemp())
        meta_dir = tmpdir / "metadata"
        meta_dir.mkdir(parents=True)

        existing = VideoMetadata(
            video_id="existing1ab",
            title="Existing Video",
            description="",
            upload_date="20240101",
            duration=100,
            url="https://youtube.com/watch?v=existing1ab",
            unlisted=False,
        )
        (meta_dir / "existing1ab.json").write_text(json.dumps(existing.to_dict(), indent=2))

        result = fetch_single_video("existing1ab", tmpdir, unlisted=True)
        assert result.video_id == "existing1ab"
        assert result.unlisted is True

        # Verify file was updated on disk
        saved_data = json.loads((meta_dir / "existing1ab.json").read_text())
        assert saved_data["unlisted"] is True


class TestProcessor:
    """Tests for document processing."""

    def _make_meta(self, description: str = "") -> VideoMetadata:
        return VideoMetadata(
            video_id="test_vid",
            title="Test Video Title",
            description=description,
            upload_date="20240615",
            duration=600,
            url="https://youtube.com/watch?v=test_vid",
        )

    def _make_transcript(self, n_segments: int = 5) -> FetchedTranscript:
        segments = [
            TranscriptSegment(
                text=f"This is segment number {i} of the video transcript with some content.",
                start=float(i * 10),
                duration=10.0,
            )
            for i in range(n_segments)
        ]
        return FetchedTranscript(video_id="test_vid", segments=segments)

    def test_process_with_transcript(self):
        """Should create transcript chunks from a video with transcript."""
        meta = self._make_meta()
        transcript = self._make_transcript(5)
        docs = process_video(meta, transcript, channel_id="testchannel", channel_name="Test Channel")

        assert len(docs) > 0
        transcript_docs = [d for d in docs if d.metadata["chunk_type"] == "transcript"]
        assert len(transcript_docs) > 0

        # Check metadata
        for doc in transcript_docs:
            assert doc.metadata["video_id"] == "test_vid"
            assert doc.metadata["video_title"] == "Test Video Title"
            assert doc.metadata["video_url"] == "https://youtube.com/watch?v=test_vid"
            assert doc.metadata["channel_id"] == "testchannel"
            assert doc.metadata["channel_name"] == "Test Channel"

    def test_process_with_description(self):
        """Should create a description document when description is long enough."""
        meta = self._make_meta(description="A" * 100)
        docs = process_video(meta, None, channel_id="ch1", channel_name="Ch1")

        desc_docs = [d for d in docs if d.metadata["chunk_type"] == "description"]
        assert len(desc_docs) == 1
        assert desc_docs[0].metadata["channel_id"] == "ch1"

    def test_process_short_description_skipped(self):
        """Should skip descriptions shorter than 50 chars."""
        meta = self._make_meta(description="Short desc")
        docs = process_video(meta, None)
        desc_docs = [d for d in docs if d.metadata["chunk_type"] == "description"]
        assert len(desc_docs) == 0

    def test_process_with_code_blocks(self):
        """Should create code documents from description code blocks."""
        desc = "Check this:\n```python\nimport numpy as np\ndata = np.array([1,2,3])\n```"
        meta = self._make_meta(description=desc)
        docs = process_video(meta, None, channel_id="ch1", channel_name="Ch1")

        code_docs = [d for d in docs if d.metadata["chunk_type"] == "code"]
        assert len(code_docs) > 0
        assert code_docs[0].metadata["code_language"] == "python"
        assert code_docs[0].metadata["channel_id"] == "ch1"

    def test_process_no_transcript(self):
        """Should handle missing transcript gracefully."""
        meta = self._make_meta(description="A" * 100)
        docs = process_video(meta, None)

        transcript_docs = [d for d in docs if d.metadata["chunk_type"] == "transcript"]
        assert len(transcript_docs) == 0

    def test_process_default_channel_id(self):
        """Should default to empty channel_id when not specified."""
        meta = self._make_meta(description="A" * 100)
        docs = process_video(meta, None)

        for doc in docs:
            assert doc.metadata["channel_id"] == ""
            assert doc.metadata["channel_name"] == ""
