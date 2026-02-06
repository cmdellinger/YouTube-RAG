"""Fetch video transcripts using youtube-transcript-api v1.x with cookie auth."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

if TYPE_CHECKING:
    from codetrading_rag.config import Config

logger = logging.getLogger(__name__)

# Global rate limiter
_last_request_time = 0.0

# Shared session (built once, reused across all transcript requests)
_shared_session: requests.Session | None = None


def _rate_limit(min_interval: float = 1.0) -> None:
    """Enforce minimum interval between requests to avoid rate limiting."""
    global _last_request_time
    current_time = time.time()
    elapsed = current_time - _last_request_time

    if elapsed < min_interval:
        sleep_time = min_interval - elapsed + random.uniform(0, 0.5)
        logger.debug("Rate limiting: sleeping %.2fs", sleep_time)
        time.sleep(sleep_time)

    _last_request_time = time.time()


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------

def _find_cookies_file() -> Path | None:
    """Look for a cookies.txt file in common locations."""
    candidates = [
        Path("cookies.txt"),
        Path("data/cookies.txt"),
        Path.home() / ".config" / "codetrading-rag" / "cookies.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _get_session(cookies_file: Path | None = None) -> requests.Session:
    """Get or build a shared requests.Session with YouTube cookies loaded.

    The session is built once and reused for all subsequent requests.
    """
    global _shared_session
    if _shared_session is not None:
        return _shared_session

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })

    if cookies_file is None:
        cookies_file = _find_cookies_file()

    if cookies_file and cookies_file.exists():
        try:
            jar = MozillaCookieJar(str(cookies_file))
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)
            logger.info("Loaded %d cookies from %s", len(jar), cookies_file)
        except Exception as exc:
            logger.warning("Failed to load cookies from %s: %s", cookies_file, exc)
    else:
        logger.warning(
            "No cookies.txt found. YouTube may block transcript requests. "
            "See .env.example for cookie setup instructions."
        )

    _shared_session = session
    return session


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Transcript fetching
# ---------------------------------------------------------------------------

def _get_transcript_path(data_dir: Path, video_id: str) -> Path:
    """Return the path for a video's transcript JSON file."""
    transcript_dir = data_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    return transcript_dir / f"{video_id}.json"


class IpBlockedError(Exception):
    """Raised when YouTube blocks our IP — signals caller to stop all fetching."""
    pass


def _fetch_with_youtube_api(
    video_id: str,
    languages: list[str],
    session: requests.Session,
    retry_delay: float = 2.0,
) -> list[TranscriptSegment] | None:
    """Fetch transcript using YouTubeTranscriptApi with cookie auth.

    Does NOT retry on IP blocks — retrying just makes the block last longer.
    Raises IpBlockedError so the caller can stop processing more videos.
    """
    try:
        _rate_limit(retry_delay)

        ytt_api = YouTubeTranscriptApi(http_client=session)
        transcript_list = ytt_api.fetch(video_id, languages=languages)

        return [
            TranscriptSegment(
                text=snippet.text,
                start=snippet.start,
                duration=snippet.duration,
            )
            for snippet in transcript_list
        ]

    except (TranscriptsDisabled, NoTranscriptFound):
        logger.debug("No transcript available for %s (disabled or not found)", video_id)
        return None

    except (IpBlocked, RequestBlocked):
        raise IpBlockedError(
            f"YouTube blocked request for {video_id}. "
            "Wait 30-60 minutes, then try again."
        )

    except VideoUnavailable:
        logger.debug("Video unavailable: %s", video_id)
        return None

    except Exception as exc:
        logger.warning("Error fetching transcript for %s: %s", video_id, exc)
        return None


def _fetch_with_yt_dlp(
    video_id: str,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
) -> list[TranscriptSegment] | None:
    """Fetch transcript using yt-dlp as fallback (supports browser cookie extraction).

    Downloads subtitles to a temp directory, then parses the json3 file.
    """
    try:
        import tempfile

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        tmpdir = tempfile.mkdtemp()

        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
            "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
            "ignore_no_formats_error": True,
        }

        # Cookie authentication
        if cookies_file and cookies_file.exists():
            ydl_opts["cookiefile"] = str(cookies_file)
        elif cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

        _rate_limit(2.0)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)

        # Look for the downloaded subtitle file
        tmppath = Path(tmpdir)
        json3_files = list(tmppath.glob("*.json3"))
        if not json3_files:
            logger.debug("yt-dlp: no subtitle file created for %s", video_id)
            return None

        sub_data = json.loads(json3_files[0].read_text())
        events = sub_data.get("events", [])
        segments = []
        for event in events:
            segs = event.get("segs", [])
            text = "".join(s.get("utf8", "") for s in segs).strip()
            if text and text != "\n":
                segments.append(TranscriptSegment(
                    text=text,
                    start=event.get("tStartMs", 0) / 1000.0,
                    duration=event.get("dDurationMs", 0) / 1000.0,
                ))

        if segments:
            logger.info(
                "yt-dlp fetched %d subtitle segments for %s",
                len(segments), video_id,
            )
            return segments

    except Exception as exc:
        logger.debug("yt-dlp transcript fetch failed for %s: %s", video_id, exc)

    return None


def fetch_transcript(
    video_id: str,
    data_dir: Path | str = "data",
    languages: list[str] | None = None,
    config: Config | None = None,
) -> FetchedTranscript | None:
    """Fetch a transcript for a single video.

    Uses youtube-transcript-api first, then falls back to yt-dlp.
    Both methods support cookie authentication to avoid IP blocks.

    Raises IpBlockedError if YouTube blocks our IP, so the caller
    can stop processing and avoid making the block worse.

    Args:
        video_id: YouTube video ID.
        data_dir: Directory to save transcript JSON files.
        languages: Preferred transcript languages (default: ["en"]).
        config: Optional Config for retry/cookie settings.

    Returns:
        FetchedTranscript if successful, None if no transcript available.

    Raises:
        IpBlockedError: YouTube is blocking requests from this IP.
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

    # Config-based settings
    retry_delay = config.transcript_retry_delay if config else 2.0
    enable_fallbacks = config.transcript_enable_fallbacks if config else True
    cookies_from_browser = (
        config.cookies_from_browser if config and config.cookies_from_browser else None
    )

    cookies_file = _find_cookies_file()
    session = _get_session(cookies_file)

    segments = None
    method_used = "unknown"

    # Method 1: youtube-transcript-api (with cookies)
    # Raises IpBlockedError if YouTube blocks us
    segments = _fetch_with_youtube_api(
        video_id, languages,
        session=session,
        retry_delay=retry_delay,
    )
    if segments is not None:
        method_used = "youtube_api"

    elif enable_fallbacks:
        # Method 2: yt-dlp (can extract cookies from browser directly)
        logger.debug("Trying yt-dlp fallback for %s", video_id)
        segments = _fetch_with_yt_dlp(
            video_id,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
        if segments is not None:
            method_used = "yt_dlp"

    if segments is None:
        logger.warning("Could not fetch transcript for %s using any method", video_id)
        return None

    result = FetchedTranscript(
        video_id=video_id,
        segments=segments,
        language=languages[0],
    )

    # Save to disk
    try:
        transcript_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info(
            "Saved transcript for %s (%d segments, method: %s)",
            video_id, len(segments), method_used,
        )
    except Exception as exc:
        logger.warning("Failed to save transcript for %s: %s", video_id, exc)

    return result


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

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
            if lang in ("py", "python3", "python"):
                lang = "python"
            elif lang in ("pine", "pinescript", "pine_script", "tradingview"):
                lang = "pinescript"
            blocks.append(CodeBlock(language=lang, code=code))

    # If no fenced blocks, look for common code patterns
    if not blocks:
        python_pattern = r"(?:^|\n)((?:(?:import |from |def |class |if __name__).*\n?)+)"
        for match in re.finditer(python_pattern, description):
            code = match.group(1).strip()
            if len(code) > 30:
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
