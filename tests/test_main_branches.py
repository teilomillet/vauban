# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Additional branch coverage for ``vauban.__main__``."""

from __future__ import annotations

import runpy
import sys
from typing import TYPE_CHECKING

import pytest

import vauban.__main__ as cli_module
import vauban._pipeline._run as pipeline_run_module
import vauban.config._validation as validation_module

if TYPE_CHECKING:
    from pathlib import Path


class TestMainHelpAndUsage:
    """Tests for top-level usage handling."""

    @pytest.mark.parametrize(
        ("argv", "expected_code"),
        [
            (["vauban"], 1),
            (["vauban", "--help"], 0),
        ],
    )
    def test_main_usage_exit_codes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
        expected_code: int,
    ) -> None:
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == expected_code
        captured = capsys.readouterr()
        assert "Usage: vauban" in captured.out


class TestInitBranches:
    """Tests for uncovered ``init`` argument branches."""

    def test_init_help_after_other_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "init", "--mode", "probe", "--help"],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Usage: vauban init" in captured.out
        assert "Modes:" in captured.out

    def test_init_model_and_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "forced.toml"
        output_path.write_text("old")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vauban",
                "init",
                "--model",
                "./custom-model",
                "--output",
                str(output_path),
                "--force",
            ],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0
        assert "./custom-model" in output_path.read_text()
        captured = capsys.readouterr()
        assert f"Created {output_path}" in captured.err


class TestDiffBranches:
    """Tests for uncovered ``diff`` flag branches."""

    def test_diff_help_inside_flag_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "diff", "--format", "text", "--help"],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Usage: vauban diff" in captured.out

    def test_diff_rejects_invalid_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "diff", "--format", "json", "a", "b"],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "--format must be 'text' or 'markdown'" in captured.err

    def test_diff_rejects_non_numeric_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "diff", "--threshold", "nope", "a", "b"],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "--threshold must be a number" in captured.err

    def test_diff_rejects_unexpected_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "diff", "--bogus", "a", "b"],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "unexpected flag '--bogus'" in captured.err

    def test_main_diff_falls_through_to_explicit_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli_module, "_run_diff", lambda args: None)
        monkeypatch.setattr(sys, "argv", ["vauban", "diff", "a", "b"])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0


class TestSchemaBranches:
    """Tests for uncovered ``schema`` branches."""

    def test_schema_help_and_late_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        for argv in (
            ["vauban", "schema", "--help"],
            ["vauban", "schema", "--output", str(tmp_path / "x.json"), "--help"],
        ):
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(SystemExit) as exc:
                cli_module.main()
            assert exc.value.code == 0
            captured = capsys.readouterr()
            assert "Usage: vauban schema" in captured.out

    def test_schema_rejects_unexpected_argument(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vauban", "schema", "extra"])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "unexpected argument 'extra'" in captured.err

    def test_main_schema_falls_through_to_explicit_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli_module, "_run_schema", lambda args: None)
        monkeypatch.setattr(sys, "argv", ["vauban", "schema"])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0


class TestTreeBranches:
    """Tests for uncovered ``tree`` branch behavior."""

    def test_main_tree_falls_through_to_explicit_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli_module, "_run_tree", lambda args: None)
        monkeypatch.setattr(sys, "argv", ["vauban", "tree"])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0


class TestValidateAndRunBranches:
    """Tests for validate-mode and pipeline error branches."""

    def test_validate_requires_exactly_one_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vauban", "--validate"])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "expected 1 config path, got 0" in captured.err

    @pytest.mark.parametrize(
        ("warnings", "expected_code"),
        [
            ([], 0),
            (["warning"], 1),
        ],
    )
    def test_validate_mode_exit_codes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings: list[str],
        expected_code: int,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text('[model]\npath = "test"\n')
        validate_calls: list[str] = []

        def _fake_validate(path: str) -> list[str]:
            validate_calls.append(path)
            return warnings

        def _fail_run(path: str) -> None:
            del path
            raise AssertionError("run() should not be called in validate mode")

        monkeypatch.setattr(validation_module, "validate_config", _fake_validate)
        monkeypatch.setattr(pipeline_run_module, "run", _fail_run)
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "--validate", str(config_path)],
        )

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == expected_code
        assert validate_calls == [str(config_path)]

    def test_normal_mode_runs_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text('[model]\npath = "test"\n')
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
        monkeypatch.setattr(sys, "argv", ["vauban", str(config_path)])

        cli_module.main()

        assert run_calls == [str(config_path)]

    def test_normal_mode_reports_pipeline_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text('[model]\npath = "test"\n')

        def _boom(path: str) -> None:
            del path
            raise RuntimeError("boom")

        monkeypatch.setattr(pipeline_run_module, "run", _boom)
        monkeypatch.setattr(sys, "argv", ["vauban", str(config_path)])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Error: boom" in captured.err


def test_main_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "--help"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(cli_module.__file__), run_name="__main__")

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Usage: vauban" in captured.out
