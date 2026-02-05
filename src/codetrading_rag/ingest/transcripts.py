"""Fetch video transcripts using youtube-transcript-api v1.x."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single segment of a video transcript."""

    text: str
    start: float
    duration: float


@dataclass
class FetchedTranscript:
    """Complete transcript for a video."""

    video_id: str
    segments: list[TranscriptSegment]
    language: str = "en"

    @property
    def full_text(self) -> str:
        """Return the full transcript as a single string."""
        return " ".join(seg.text for seg in self.segments)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "segments": [asdict(seg) for seg in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> FetchedTranscript:
        return cls(
            video_id=data["video_id"],
            language=data.get("language", "en"),
            segments=[TranscriptSegment(**seg) for seg in data["segments"]],
        )


@dataclass
class CodeBlock:
    """A code block extracted from a video description."""

    language: str  # "python", "pinescript", "unknown"
    code: str


def _get_transcript_path(data_dir: Path, video_id: str) -> Path:
    """Return the path for a video's transcript JSON file."""
    transcript_dir = data_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    return transcript_dir / f"{video_id}.json"


def fetch_transcript(
    video_id: str,
    data_dir: Path | str = "data",
    languages: list[str] | None = None,
) -> FetchedTranscript | None:
    """Fetch a transcript for a single video.

    Args:
        video_id: YouTube video ID.
        data_dir: Directory to save transcript JSON files.
        languages: Preferred transcript languages (default: ["en"]).

    Returns:
        FetchedTranscript if successful, None if no transcript available.
    """
    data_dir = Path(data_dir)
    if languages is None:
        languages = ["en"]

    # Check if already saved
    transcript_path = _get_transcript_path(data_dir, video_id)
    if transcript_path.exists():
        try:
            data = json.loads(transcript_path.read_text())
            return FetchedTranscript.from_dict(data)
        except Exception:
            pass  # Re-fetch if corrupted

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=languages)

        segments = [
            TranscriptSegment(
                text=snippet.text,
                start=snippet.start,
                duration=snippet.duration,
            )
            for snippet in transcript_list
        ]

        result = FetchedTranscript(
            video_id=video_id,
            segments=segments,
            language=languages[0],
        )

        # Save to disk
        transcript_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Saved transcript for %s (%d segments)", video_id, len(segments))

        return result

    except Exception as exc:
        logger.warning("Could not fetch transcript for %s: %s", video_id, exc)
        return None


def extract_code_from_description(description: str) -> list[CodeBlock]:
    """Extract code blocks from a video description.

    Looks for:
    - Fenced code blocks: ```python ... ``` or ```pinescript ... ```
    - Inline code patterns that look like Python/Pine Script
    """
    blocks: list[CodeBlock] = []

    # Match fenced code blocks
    fenced_pattern = r"```(\w*)\n(.*?)```"
    for match in re.finditer(fenced_pattern, description, re.DOTALL):
        lang = match.group(1).lower() or "unknown"
        code = match.group(2).strip()
        if code:
            # Normalize common language tags
            if lang in ("py", "python3", "python"):
                lang = "python"
            elif lang in ("pine", "pinescript", "pine_script", "tradingview"):
                lang = "pinescript"
            blocks.append(CodeBlock(language=lang, code=code))

    # If no fenced blocks, look for common code patterns
    if not blocks:
        # Python-like patterns: import, def, class, if __name__
        python_pattern = r"(?:^|\n)((?:(?:import |from |def |class |if __name__).*\n?)+)"
        for match in re.finditer(python_pattern, description):
            code = match.group(1).strip()
            if len(code) > 30:  # Skip very short snippets
                blocks.append(CodeBlock(language="python", code=code))

    return blocks


def load_transcript(video_id: str, data_dir: Path | str = "data") -> FetchedTranscript | None:
    """Load a previously-saved transcript from disk."""
    data_dir = Path(data_dir)
    path = _get_transcript_path(data_dir, video_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return FetchedTranscript.from_dict(data)
    except Exception as exc:
        logger.warning("Error loading transcript %s: %s", path, exc)
        return None
