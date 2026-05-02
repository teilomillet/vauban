# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban init — config scaffolding."""

import sys
from pathlib import Path
from typing import cast

import pytest

from vauban._init import (
    _STANDALONE_TEMPLATES,
    KNOWN_MODES,
    _write_ai_act_supporting_files,
    init_config,
)
from vauban.ai_act import generate_deployer_readiness_bundle


def _object_dict(value: object) -> dict[str, object]:
    """Narrow a JSON-like object to a string-keyed dict for tests."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


class TestInitConfig:
    def test_default_mode_produces_valid_toml(self, tmp_path: Path) -> None:
        """Default mode config should parse as valid TOML with [model]+[data]."""
        import tomllib

        content = init_config("default", output_path=tmp_path / "run.toml")
        parsed = tomllib.loads(content)
        assert "model" in parsed
        assert "data" in parsed
        assert parsed["model"]["path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert parsed["data"]["harmful"] == "default"

    @pytest.mark.parametrize("mode", sorted(KNOWN_MODES))
    def test_every_mode_produces_valid_toml(self, mode: str) -> None:
        """Each mode template must produce parseable TOML."""
        import tomllib

        content = init_config(mode)
        parsed = tomllib.loads(content)
        if mode in _STANDALONE_TEMPLATES:
            # Standalone modes don't require [model] or [data]
            assert isinstance(parsed, dict)
        else:
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

    def test_scenario_implies_softprompt_and_serializes_environment(self) -> None:
        import tomllib

        content = init_config(scenario="share_doc")
        parsed = tomllib.loads(content)

        assert "softprompt" in parsed
        assert parsed["softprompt"]["mode"] == "gcg"
        assert parsed["output"]["dir"] == "output/share_doc"
        assert parsed["environment"]["scenario"] == "share_doc"
        assert "target" not in parsed["environment"]

    def test_scenario_rejects_incompatible_mode(self) -> None:
        with pytest.raises(ValueError, match="only supported with mode 'softprompt'"):
            init_config("probe", scenario="share_doc")

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown environment scenario"):
            init_config("softprompt", scenario="nonexistent")

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

    def test_ai_act_scaffolds_draft_evidence_templates(
        self,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "readiness.toml"
        content = init_config("ai_act", output_path=out)

        assert out.exists()
        assert "pdf_report = true" in content
        assert 'ai_literacy_record = "evidence/ai_literacy.md"' in content
        evidence_dir = tmp_path / "evidence"
        assert (evidence_dir / "ai_literacy.md").exists()
        assert (evidence_dir / "transparency_notice.md").exists()
        assert (evidence_dir / "human_oversight.md").exists()
        assert (evidence_dir / "incident_response.md").exists()
        assert (evidence_dir / "provider_docs.md").exists()
        assert (evidence_dir / "README.md").exists()
        assert "Template status: draft scaffold" in (
            evidence_dir / "ai_literacy.md"
        ).read_text()

    def test_ai_act_supporting_files_skip_existing_without_force(
        self,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "readiness.toml"
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir(parents=True)
        existing = evidence_dir / "ai_literacy.md"
        existing.write_text("existing content")

        written = _write_ai_act_supporting_files(out, force=False)

        assert existing.read_text() == "existing content"
        assert existing not in written
        assert (evidence_dir / "transparency_notice.md").exists()

    def test_roundtrip_validation_failure_raises_internal_error(self) -> None:
        from unittest.mock import patch

        with (
            patch("tomllib.loads", return_value={}),
            pytest.raises(ValueError, match="missing required sections"),
        ):
            init_config("default")

    def test_mode_descriptions_guard_raises_on_missing_registry_entry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import importlib

        import vauban._init as init_module
        from vauban.config import _mode_registry

        broken = dict(_mode_registry.EARLY_MODE_DESCRIPTION_BY_MODE)
        removed_mode = next(iter(broken))
        del broken[removed_mode]

        with monkeypatch.context() as ctx:
            ctx.setattr(_mode_registry, "EARLY_MODE_DESCRIPTION_BY_MODE", broken)
            with pytest.raises(
                AssertionError,
                match="MODE_DESCRIPTIONS is missing entries",
            ):
                importlib.reload(init_module)

        importlib.reload(init_module)


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
        if mode in _STANDALONE_TEMPLATES:
            # Standalone modes may have empty model_path
            assert isinstance(config.model_path, str)
        else:
            assert config.model_path == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_ai_act_scaffold_roundtrips_but_stays_blocked_until_filled(
        self,
        tmp_path: Path,
    ) -> None:
        from vauban.config import load_config

        out = tmp_path / "readiness.toml"
        init_config("ai_act", output_path=out)
        config = load_config(out)
        assert config.ai_act is not None

        report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config.ai_act,
        )

        assert report["overall_status"] == "blocked"
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        article4 = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.article4.ai_literacy_record"
        )
        assert article4["status"] == "unknown"
        missing_markers = article4["missing_markers"]
        assert isinstance(missing_markers, list)
        assert "replace_scaffold_placeholders" in missing_markers

    def test_scenario_scaffold_roundtrips_environment_fields(
        self,
        tmp_path: Path,
    ) -> None:
        from vauban.config import load_config

        out = tmp_path / "share_doc.toml"
        init_config("softprompt", output_path=out, scenario="share_doc")
        config = load_config(out)

        assert config.environment is not None
        assert config.environment.scenario == "share_doc"
        assert config.environment.injection_position == "infix"
        assert config.environment.benign_expected_tools == [
            "read_document_content",
        ]
        assert config.environment.rollout_every_n == 1
        assert config.environment.target.function == "share_drive_file"


class TestPublicApiLazyImport:
    def test_tuple_backed_lazy_import_resolves_original_name(self) -> None:
        import vauban
        from vauban.scan import scan

        assert vauban.__getattr__("injection_scan") is scan

    def test_dir_exposes_lazy_public_symbols(self) -> None:
        import vauban

        names = dir(vauban)

        assert "injection_scan" in names
        assert "__version__" in names
        assert "validate" in names


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

    def test_init_with_scenario(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "share_doc.toml"
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "init", "--scenario", "share_doc", "--output", str(out)],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        content = out.read_text()
        assert "[softprompt]" in content
        assert "[environment]" in content
        assert 'scenario = "share_doc"' in content
        captured = capsys.readouterr()
        assert "share_doc scenario" in captured.err

    def test_init_ai_act_mentions_scaffold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "readiness.toml"
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "init", "--mode", "ai_act", "--output", str(out)],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Scaffolded draft AI Act evidence templates" in captured.err
        assert (tmp_path / "evidence" / "ai_literacy.md").exists()

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

    def test_init_bad_scenario_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vauban",
                "init",
                "--scenario",
                "nonexistent",
                "--output",
                str(tmp_path / "x.toml"),
            ],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Unknown environment scenario" in captured.err

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
        assert "Scenarios:" in captured.out
