"""Tests for the runtime-generated manual."""

import pytest

from vauban.manual import manual_topics, render_manual


def test_topics_include_quickstart_and_sections() -> None:
    topics = manual_topics()
    assert "quickstart" in topics
    assert "commands" in topics
    assert "validate" in topics
    assert "playbook" in topics
    assert "quick" in topics
    assert "examples" in topics
    assert "print" in topics
    assert "cut" in topics
    assert "softprompt" in topics


def test_quickstart_topic_contains_minimal_config() -> None:
    text = render_manual("quickstart")
    assert "QUICKSTART" in text
    assert "[model]" in text
    assert 'harmful = "default"' in text


def test_commands_topic_contains_init_and_diff() -> None:
    text = render_manual("commands")
    assert "COMMANDS" in text
    assert "vauban init [--mode MODE]" in text
    assert "vauban diff" in text
    assert "--format" in text
    assert "known modes:" in text


def test_validate_topic_mentions_fix_guidance() -> None:
    text = render_manual("validate")
    assert "VALIDATE WORKFLOW" in text
    assert "fix:" in text
    assert "--validate run.toml" in text


def test_quick_topic_lists_repl_helpers() -> None:
    text = render_manual("quick")
    assert "PYTHON QUICK API" in text
    assert "helpers:" in text
    assert "measure_direction" in text


def test_examples_topic_contains_core_flows() -> None:
    text = render_manual("examples")
    assert "EXAMPLES" in text
    assert "vauban init --mode default --output run.toml" in text
    assert "vauban diff runs/baseline runs/experiment_a" in text


def test_print_topic_contains_share_commands() -> None:
    text = render_manual("print")
    assert "PRINTING AND SHARING" in text
    assert "vauban man > VAUBAN_MANUAL.txt" in text
    assert "lpr VAUBAN_MANUAL.txt" in text


def test_section_topic_renders_only_requested_section() -> None:
    text = render_manual("cut")
    assert "SECTION [cut]" in text
    assert text.count("SECTION [") == 1
    assert "default: 1.0" in text
    assert "[cut].alpha" in text


def test_unknown_topic_raises() -> None:
    with pytest.raises(ValueError, match="Unknown manual topic"):
        render_manual("does-not-exist")
