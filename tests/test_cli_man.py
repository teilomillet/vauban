# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for the built-in manual interface."""

import sys

import pytest

from vauban.__main__ import main


def test_main_man_quickstart(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "man", "quickstart"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "QUICKSTART" in captured.out
    assert "Minimal run.toml" in captured.out
    assert "examples/benchmarks/share_doc.toml" in captured.out


def test_main_man_unknown_topic_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "man", "nope"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Unknown manual topic" in captured.err


def test_main_man_rejects_extra_args(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "man", "cut", "extra"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "expected at most one manual topic" in captured.err


def test_commands_section_contains_format_flag() -> None:
    from vauban.manual import render_manual

    output = render_manual("commands")
    assert "--format" in output
    assert "--threshold" in output
    assert "CI gate" in output
    assert "verify-bundle" in output
    assert "vauban tree" in output


def test_quick_section_contains_compare_scan() -> None:
    from vauban.manual import render_manual

    output = render_manual("quick")
    assert "compare" in output
    assert "scan" in output


def test_validate_section_mentions_key_level_typos() -> None:
    from vauban.manual import render_manual

    output = render_manual("validate")
    assert "key-level" in output


def test_main_old_flag_man_shows_migration_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "--man"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "'--man' is no longer supported" in captured.err
    assert "Use 'vauban man [topic]'" in captured.err


def test_main_typo_command_suggests_valid_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "men"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "unknown command 'men'" in captured.err
    assert "Did you mean 'man'" in captured.err


def test_main_tree_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "tree", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Visualize experiment tech tree from TOML configs" in captured.out
    assert "--format" in captured.out
