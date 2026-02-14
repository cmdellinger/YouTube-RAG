"""Channel registry management for multi-channel YouTube RAG."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """Metadata for a registered YouTube channel."""

    slug: str
    name: str
    url: str
    added_at: str = ""
    last_ingested_at: str = ""
    video_count: int = 0
    transcript_count: int = 0
    status: str = "new"  # new, ingesting, ready, error

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ChannelInfo:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def slugify(name: str) -> str:
    """Convert a channel name to a filesystem-safe slug.

    Examples:
        "CodeTradingCafe" -> "codetradingcafe"
        "The Trading Channel" -> "the-trading-channel"
        "@CodeTradingCafe" -> "codetradingcafe"
    """
    name = name.strip().lstrip("@")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug


class ChannelManager:
    """Manages the channel registry and per-channel data directories."""

    def __init__(self, data_dir: Path | str = "data") -> None:
        self._data_dir = Path(data_dir)
        self._registry_path = self._data_dir / "channels.json"
        self._channels_dir = self._data_dir / "channels"
        self._channels_dir.mkdir(parents=True, exist_ok=True)
        self._channels: dict[str, ChannelInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load channel registry from disk."""
        if self._registry_path.exists():
            data = json.loads(self._registry_path.read_text())
            for ch in data.get("channels", []):
                info = ChannelInfo.from_dict(ch)
                self._channels[info.slug] = info
            logger.info("Loaded %d channels from registry", len(self._channels))
        else:
            logger.info("No channel registry found, starting fresh")

    def recover_stale_ingesting(self) -> list[str]:
        """Reset channels stuck in 'ingesting' state to 'error'.

        Call once at application startup, not on every ChannelManager instantiation.

        Returns:
            List of channel slugs that were reset.
        """
        reset = []
        for ch in self._channels.values():
            if ch.status == "ingesting":
                logger.warning(
                    "Channel '%s' was left in 'ingesting' state; resetting to 'error'",
                    ch.slug,
                )
                ch.status = "error"
                reset.append(ch.slug)
        if reset:
            self._save()
        return reset

    def _save(self) -> None:
        """Persist channel registry to disk."""
        data = {"channels": [ch.to_dict() for ch in self._channels.values()]}
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(json.dumps(data, indent=2))

    @property
    def channels(self) -> list[ChannelInfo]:
        """Return all registered channels."""
        return list(self._channels.values())

    @property
    def slugs(self) -> list[str]:
        """Return all channel slugs."""
        return list(self._channels.keys())

    def get(self, slug: str) -> ChannelInfo | None:
        """Get a channel by slug."""
        return self._channels.get(slug)

    def add_channel(self, name: str, url: str) -> ChannelInfo:
        """Register a new channel.

        Args:
            name: Display name for the channel.
            url: YouTube channel URL.

        Returns:
            The created ChannelInfo.

        Raises:
            ValueError: If a channel with the same slug already exists.
        """
        slug = slugify(name)
        if slug in self._channels:
            raise ValueError(f"Channel '{slug}' already registered")

        info = ChannelInfo(
            slug=slug,
            name=name,
            url=url,
            added_at=datetime.now(timezone.utc).isoformat(),
            status="new",
        )
        self._channels[slug] = info

        # Create data directories
        channel_dir = self._channels_dir / slug
        (channel_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (channel_dir / "transcripts").mkdir(parents=True, exist_ok=True)

        self._save()
        logger.info("Added channel: %s (%s)", name, slug)
        return info

    def remove_channel(self, slug: str) -> None:
        """Unregister a channel. Does NOT delete data files or ChromaDB docs.

        Args:
            slug: Channel slug to remove.

        Raises:
            ValueError: If the channel is not found.
        """
        if slug not in self._channels:
            raise ValueError(f"Channel '{slug}' not found")
        del self._channels[slug]
        self._save()
        logger.info("Removed channel: %s", slug)

    def update_status(self, slug: str, status: str, **kwargs: object) -> None:
        """Update a channel's status and optional fields.

        Args:
            slug: Channel slug to update.
            status: New status value.
            **kwargs: Additional fields to update (e.g. video_count, transcript_count).
        """
        ch = self._channels.get(slug)
        if not ch:
            raise ValueError(f"Channel '{slug}' not found")
        ch.status = status
        for k, v in kwargs.items():
            if hasattr(ch, k):
                setattr(ch, k, v)
        self._save()

    def get_data_dir(self, slug: str) -> Path:
        """Return the data directory for a channel."""
        return self._channels_dir / slug

    def get_metadata_dir(self, slug: str) -> Path:
        """Return the metadata directory for a channel."""
        return self._channels_dir / slug / "metadata"

    def get_transcripts_dir(self, slug: str) -> Path:
        """Return the transcripts directory for a channel."""
        return self._channels_dir / slug / "transcripts"

    def refresh_counts(self, slug: str) -> None:
        """Refresh video and transcript counts from disk."""
        meta_dir = self.get_metadata_dir(slug)
        trans_dir = self.get_transcripts_dir(slug)
        video_count = len(list(meta_dir.glob("*.json"))) if meta_dir.exists() else 0
        transcript_count = len(list(trans_dir.glob("*.json"))) if trans_dir.exists() else 0
        self.update_status(
            slug,
            self._channels[slug].status,
            video_count=video_count,
            transcript_count=transcript_count,
        )
