"""Tests for vauban init — config scaffolding."""

import sys
from pathlib import Path

import pytest

from vauban._init import KNOWN_MODES, init_config


class TestInitConfig:
    def test_default_mode_produces_valid_toml(self, tmp_path: Path) -> None:
        """Default mode config should parse as valid TOML with [model]+[data]."""
        import tomllib

        content = init_config("default", output_path=tmp_path / "run.toml")
        parsed = tomllib.loads(content)
        assert "model" in parsed
        assert "data" in parsed
        assert parsed["model"]["path"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert parsed["data"]["harmful"] == "default"

    @pytest.mark.parametrize("mode", sorted(KNOWN_MODES))
    def test_every_mode_produces_valid_toml(self, mode: str) -> None:
        """Each mode template must produce parseable TOML."""
        import tomllib

        content = init_config(mode)
        parsed = tomllib.loads(content)
        assert "model" in parsed
        assert "data" in parsed

    def test_custom_model_in_output(self) -> None:
        content = init_config(model="./my-local-model")
        assert "./my-local-model" in content

    def test_probe_mode_has_probe_section(self) -> None:
        import tomllib

        content = init_config("probe")
        parsed = tomllib.loads(content)
        assert "probe" in parsed
        assert "prompts" in parsed["probe"]

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            init_config("nonexistent")

    def test_writes_to_disk(self, tmp_path: Path) -> None:
        out = tmp_path / "test.toml"
        init_config("default", output_path=out)
        assert out.exists()
        assert "[model]" in out.read_text()

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        out = tmp_path / "existing.toml"
        out.write_text("existing content")
        with pytest.raises(FileExistsError, match="already exists"):
            init_config("default", output_path=out)

    def test_overwrites_with_force(self, tmp_path: Path) -> None:
        out = tmp_path / "existing.toml"
        out.write_text("old")
        init_config("default", output_path=out, force=True)
        assert "[model]" in out.read_text()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "run.toml"
        init_config("default", output_path=out)
        assert out.exists()


class TestInitRoundtrip:
    @pytest.mark.parametrize("mode", sorted(KNOWN_MODES))
    def test_roundtrip_through_load_config(
        self, mode: str, tmp_path: Path,
    ) -> None:
        """Every mode template should roundtrip through load_config."""
        from vauban.config import load_config

        out = tmp_path / f"{mode}.toml"
        init_config(mode, output_path=out)
        config = load_config(out)
        assert config.model_path == "mlx-community/Llama-3.2-3B-Instruct-4bit"


class TestInitCli:
    def test_init_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "run.toml"
        monkeypatch.setattr(
            sys, "argv", ["vauban", "init", "--output", str(out)],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert out.exists()
        captured = capsys.readouterr()
        assert "Created" in captured.err

    def test_init_with_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "probe.toml"
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "init", "--mode", "probe", "--output", str(out)],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert "[probe]" in out.read_text()

    def test_init_bad_mode_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "init", "--mode", "bad", "--output", str(tmp_path / "x.toml")],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Unknown mode" in captured.err

    def test_init_unexpected_arg_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["vauban", "init", "--bogus"],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "unexpected argument" in captured.err

    def test_init_help(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["vauban", "init", "--help"],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Usage: vauban init" in captured.out
        assert "Modes:" in captured.out
