# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke tests for the checked-in environment benchmark pack."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import vauban.__main__ as cli_module
import vauban._pipeline._run as pipeline_run_module
import vauban.config._validation as validation_module

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARK_CONFIG = _REPO_ROOT / "examples" / "benchmarks" / "share_doc.toml"


def test_cli_validate_checked_in_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``vauban --validate`` should accept the canonical benchmark config."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["vauban", "--validate", str(_BENCHMARK_CONFIG)],
    )

    with pytest.raises(SystemExit) as exc:
        cli_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "No issues found." in captured.err


def test_cli_run_checked_in_benchmark_delegates_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark examples should work as normal TOML entrypoints."""
    run_calls: list[str] = []

    def _fake_run(path: str) -> None:
        run_calls.append(path)

    monkeypatch.setattr(pipeline_run_module, "run", _fake_run)
    monkeypatch.setattr(
        validation_module,
        "validate_config",
        lambda path: (_ for _ in ()).throw(
            AssertionError("validate_config() should not be called"),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["vauban", str(_BENCHMARK_CONFIG)])

    cli_module.main()

    assert run_calls == [str(_BENCHMARK_CONFIG)]
