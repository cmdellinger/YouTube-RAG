"""Fetch video list and metadata from a YouTube channel using yt-dlp."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a single YouTube video."""

    video_id: str
    title: str
    description: str
    upload_date: str  # YYYYMMDD
    duration: int  # seconds
    url: str
    unlisted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VideoMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _get_metadata_path(data_dir: Path, video_id: str) -> Path:
    """Return the path to a video's metadata JSON file."""
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / f"{video_id}.json"


def _load_existing_ids(data_dir: Path) -> set[str]:
    """Load the set of video IDs that have already been fetched."""
    metadata_dir = data_dir / "metadata"
    if not metadata_dir.exists():
        return set()
    return {p.stem for p in metadata_dir.glob("*.json")}


def fetch_channel_videos(
    channel_url: str,
    data_dir: Path | str = "data",
    limit: int | None = None,
    skip_existing: bool = True,
) -> list[VideoMetadata]:
    """Fetch video metadata from a YouTube channel.

    Args:
        channel_url: URL of the YouTube channel.
        data_dir: Directory to store metadata JSON files.
        limit: Maximum number of videos to fetch (None = all).
        skip_existing: If True, skip videos whose metadata is already saved.

    Returns:
        List of VideoMetadata for newly fetched videos.
    """
    data_dir = Path(data_dir)
    existing_ids = _load_existing_ids(data_dir) if skip_existing else set()

    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "force_generic_extractor": False,
    }

    # Fetch the list of video entries from the channel
    logger.info("Fetching video list from %s ...", channel_url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        channel_info = ydl.extract_info(channel_url, download=False)

    if not channel_info:
        logger.warning("Could not extract channel info from %s", channel_url)
        return []

    entries = channel_info.get("entries", [])
    if not entries:
        logger.warning("No videos found on channel %s", channel_url)
        return []

    # Flatten nested playlists (channels can have tabs → playlists → entries)
    flat_entries: list[dict] = []
    for entry in entries:
        if entry is None:
            continue
        if "entries" in entry:
            # This is a nested playlist/tab — flatten it
            for sub in entry.get("entries", []):
                if sub is not None:
                    flat_entries.append(sub)
        else:
            flat_entries.append(entry)

    logger.info("Found %d videos on channel", len(flat_entries))

    # Filter out already-fetched videos
    new_entries = [
        e for e in flat_entries if e.get("id") and e["id"] not in existing_ids
    ]
    if skip_existing and len(new_entries) < len(flat_entries):
        logger.info("Skipping %d already-fetched videos", len(flat_entries) - len(new_entries))

    if limit is not None:
        new_entries = new_entries[:limit]

    # Fetch full metadata for each new video
    results: list[VideoMetadata] = []
    detail_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    for i, entry in enumerate(new_entries, 1):
        video_id = entry.get("id", "")
        video_url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"

        logger.info("[%d/%d] Fetching metadata for: %s", i, len(new_entries), video_id)

        try:
            with yt_dlp.YoutubeDL(detail_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

            if not info:
                logger.warning("  Could not extract info for %s", video_id)
                continue

            meta = VideoMetadata(
                video_id=info.get("id", video_id),
                title=info.get("title", ""),
                description=info.get("description", ""),
                upload_date=info.get("upload_date", ""),
                duration=info.get("duration", 0) or 0,
                url=info.get("webpage_url", video_url),
            )

            # Save metadata to disk
            meta_path = _get_metadata_path(data_dir, meta.video_id)
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

            results.append(meta)
            logger.info("  Saved: %s", meta.title)

        except Exception as exc:
            logger.warning("  Error fetching %s: %s", video_id, exc)
            continue

    logger.info("Fetched metadata for %d new videos", len(results))
    return results


def load_all_metadata(data_dir: Path | str = "data") -> list[VideoMetadata]:
    """Load all previously-saved video metadata from disk."""
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "metadata"
    if not metadata_dir.exists():
        return []

    results: list[VideoMetadata] = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            results.append(VideoMetadata.from_dict(data))
        except Exception as exc:
            logger.warning("Error loading %s: %s", path, exc)

    return results


def parse_video_id(url_or_id: str) -> str:
    """Extract a YouTube video ID from a URL or return it if already an ID.

    Supports formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/live/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - VIDEO_ID (11-char alphanumeric string)

    Raises:
        ValueError: If the input cannot be parsed as a valid video ID.
    """
    url_or_id = url_or_id.strip()

    # Direct video ID (11 chars, alphanumeric + dash/underscore)
    if re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id

    # YouTube URL patterns
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def fetch_single_video(
    url_or_id: str,
    data_dir: Path | str,
    unlisted: bool = True,
) -> VideoMetadata:
    """Fetch metadata for a single YouTube video and save it.

    Designed for adding unlisted or private videos that don't appear
    in a channel's public video listing.

    Args:
        url_or_id: YouTube video URL or 11-character video ID.
        data_dir: Channel data directory (e.g., data/channels/{slug}/).
        unlisted: Whether to mark this video as unlisted.

    Returns:
        The fetched VideoMetadata.

    Raises:
        ValueError: If the URL/ID cannot be parsed or video cannot be fetched.
    """
    data_dir = Path(data_dir)
    video_id = parse_video_id(url_or_id)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Check if already exists
    existing_ids = _load_existing_ids(data_dir)
    if video_id in existing_ids:
        meta_path = _get_metadata_path(data_dir, video_id)
        data = json.loads(meta_path.read_text())
        existing = VideoMetadata.from_dict(data)
        if unlisted and not existing.unlisted:
            existing.unlisted = True
            meta_path.write_text(json.dumps(existing.to_dict(), indent=2))
        return existing

    logger.info("Fetching metadata for single video: %s", video_id)

    detail_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(detail_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    if not info:
        raise ValueError(f"Could not extract info for video: {video_id}")

    meta = VideoMetadata(
        video_id=info.get("id", video_id),
        title=info.get("title", ""),
        description=info.get("description", ""),
        upload_date=info.get("upload_date", ""),
        duration=info.get("duration", 0) or 0,
        url=info.get("webpage_url", video_url),
        unlisted=unlisted,
    )

    meta_path = _get_metadata_path(data_dir, meta.video_id)
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2))
    logger.info("Saved unlisted video: %s (%s)", meta.title, meta.video_id)

    return meta
