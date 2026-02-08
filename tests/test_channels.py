"""Tests for channel management."""

import json
import tempfile
from pathlib import Path

import pytest

from codetrading_rag.channels.manager import ChannelInfo, ChannelManager, slugify


class TestSlugify:
    """Tests for the slugify function."""

    def test_simple_name(self):
        assert slugify("CodeTradingCafe") == "codetradingcafe"

    def test_name_with_spaces(self):
        assert slugify("The Trading Channel") == "the-trading-channel"

    def test_name_with_at_sign(self):
        assert slugify("@CodeTradingCafe") == "codetradingcafe"

    def test_name_with_special_chars(self):
        assert slugify("My Channel! #1") == "my-channel-1"

    def test_whitespace_stripped(self):
        assert slugify("  Padded  ") == "padded"


class TestChannelInfo:
    """Tests for ChannelInfo dataclass."""

    def test_roundtrip(self):
        """ChannelInfo should survive dict serialization roundtrip."""
        info = ChannelInfo(
            slug="test",
            name="Test Channel",
            url="https://youtube.com/@test",
            status="ready",
            video_count=10,
            transcript_count=5,
        )
        restored = ChannelInfo.from_dict(info.to_dict())
        assert restored.slug == "test"
        assert restored.name == "Test Channel"
        assert restored.status == "ready"
        assert restored.video_count == 10

    def test_defaults(self):
        """ChannelInfo should have sensible defaults."""
        info = ChannelInfo(slug="x", name="X", url="http://x")
        assert info.status == "new"
        assert info.video_count == 0
        assert info.transcript_count == 0


class TestChannelManager:
    """Tests for ChannelManager."""

    def _make_manager(self) -> tuple[ChannelManager, Path]:
        """Create a ChannelManager with a temporary data directory."""
        tmpdir = Path(tempfile.mkdtemp())
        manager = ChannelManager(tmpdir)
        return manager, tmpdir

    def test_add_channel(self):
        manager, tmpdir = self._make_manager()
        info = manager.add_channel("TestChannel", "https://youtube.com/@test")

        assert info.slug == "testchannel"
        assert info.name == "TestChannel"
        assert info.status == "new"
        assert len(manager.channels) == 1

        # Check directories were created
        assert (tmpdir / "channels" / "testchannel" / "metadata").exists()
        assert (tmpdir / "channels" / "testchannel" / "transcripts").exists()

    def test_add_duplicate_channel(self):
        manager, _ = self._make_manager()
        manager.add_channel("TestChannel", "https://youtube.com/@test")

        with pytest.raises(ValueError, match="already registered"):
            manager.add_channel("TestChannel", "https://youtube.com/@test2")

    def test_remove_channel(self):
        manager, _ = self._make_manager()
        manager.add_channel("TestChannel", "https://youtube.com/@test")
        assert len(manager.channels) == 1

        manager.remove_channel("testchannel")
        assert len(manager.channels) == 0

    def test_remove_nonexistent_channel(self):
        manager, _ = self._make_manager()

        with pytest.raises(ValueError, match="not found"):
            manager.remove_channel("nonexistent")

    def test_get_channel(self):
        manager, _ = self._make_manager()
        manager.add_channel("TestChannel", "https://youtube.com/@test")

        info = manager.get("testchannel")
        assert info is not None
        assert info.name == "TestChannel"

        assert manager.get("nonexistent") is None

    def test_update_status(self):
        manager, _ = self._make_manager()
        manager.add_channel("TestChannel", "https://youtube.com/@test")

        manager.update_status("testchannel", "ingesting", video_count=42)
        info = manager.get("testchannel")
        assert info.status == "ingesting"
        assert info.video_count == 42

    def test_persistence(self):
        """ChannelManager should persist channels across instances."""
        tmpdir = Path(tempfile.mkdtemp())

        # Add a channel with first instance
        manager1 = ChannelManager(tmpdir)
        manager1.add_channel("TestChannel", "https://youtube.com/@test")

        # Load with a new instance
        manager2 = ChannelManager(tmpdir)
        assert len(manager2.channels) == 1
        assert manager2.get("testchannel") is not None

    def test_slugs_property(self):
        manager, _ = self._make_manager()
        manager.add_channel("Channel A", "https://youtube.com/@a")
        manager.add_channel("Channel B", "https://youtube.com/@b")

        slugs = manager.slugs
        assert "channel-a" in slugs
        assert "channel-b" in slugs

    def test_get_data_dir(self):
        manager, tmpdir = self._make_manager()
        manager.add_channel("TestChannel", "https://youtube.com/@test")

        data_dir = manager.get_data_dir("testchannel")
        assert data_dir == tmpdir / "channels" / "testchannel"
