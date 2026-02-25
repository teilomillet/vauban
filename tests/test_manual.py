"""Tests for the runtime-generated manual."""

import pytest

from vauban.manual import manual_topics, render_manual


def test_topics_include_quickstart_and_sections() -> None:
    topics = manual_topics()
    assert "quickstart" in topics
    assert "cut" in topics
    assert "softprompt" in topics


def test_quickstart_topic_contains_minimal_config() -> None:
    text = render_manual("quickstart")
    assert "QUICKSTART" in text
    assert "[model]" in text
    assert 'harmful = "default"' in text


def test_section_topic_renders_only_requested_section() -> None:
    text = render_manual("cut")
    assert "SECTION [cut]" in text
    assert text.count("SECTION [") == 1
    assert "default: 1.0" in text
    assert "[cut].alpha" in text


def test_unknown_topic_raises() -> None:
    with pytest.raises(ValueError, match="Unknown manual topic"):
        render_manual("does-not-exist")
