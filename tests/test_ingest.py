"""Tests for the ingestion pipeline."""

from codetrading_rag.ingest.channel import VideoMetadata
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
