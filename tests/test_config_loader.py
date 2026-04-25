# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.config._loader standalone detection and config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from vauban.config import load_config
from vauban.config._loader import (
    _is_standalone_api_eval,
    _is_standalone_behavior_report,
    _is_standalone_benchmark,
    _is_standalone_token_audit,
    _resolve_single_data,
)


class TestLoadConfigMinimal:
    """Minimal valid config loads and returns correct defaults."""

    def test_minimal_valid_config(self, tmp_path: Path) -> None:
        """A config with [model] + [data] loads successfully."""
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            '[model]\npath = "mlx-community/tiny"\n'
            "[data]\n"
            'harmful = "harmful.jsonl"\n'
            'harmless = "harmless.jsonl"\n'
        )
        config = load_config(toml_file)
        assert config.model_path == "mlx-community/tiny"
        assert config.harmful_path == tmp_path / "harmful.jsonl"
        assert config.harmless_path == tmp_path / "harmless.jsonl"


class TestDefaultDataPaths:
    """'default' sentinel in [data] resolves to bundled datasets."""

    def test_default_harmful_and_harmless(self, tmp_path: Path) -> None:
        """Both harmful='default' and harmless='default' resolve to bundled paths."""
        from vauban.measure import default_prompt_paths

        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "default"\n'
            'harmless = "default"\n'
        )
        config = load_config(toml_file)
        expected_harmful, expected_harmless = default_prompt_paths()
        assert config.harmful_path == expected_harmful
        assert config.harmless_path == expected_harmless

    def test_default_only_harmful(self, tmp_path: Path) -> None:
        """Only harmful='default' resolves; harmless is a relative path."""
        from vauban.measure import default_prompt_paths

        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "default"\n'
            'harmless = "local.jsonl"\n'
        )
        config = load_config(toml_file)
        expected_harmful, _ = default_prompt_paths()
        assert config.harmful_path == expected_harmful
        assert config.harmless_path == tmp_path / "local.jsonl"


class TestMissingModel:
    """Configs without a valid [model] section raise ValueError."""

    def test_missing_model_section_raises(self, tmp_path: Path) -> None:
        """No [model] section at all raises ValueError mentioning 'model'."""
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text(
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
        )
        with pytest.raises(ValueError, match="model"):
            load_config(toml_file)

    def test_empty_model_section_raises(self, tmp_path: Path) -> None:
        """[model] present but missing 'path' key raises ValueError."""
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text(
            "[model]\n"
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
        )
        with pytest.raises(ValueError, match="model"):
            load_config(toml_file)


class TestStandaloneApiEval:
    """Standalone api_eval mode: [api_eval] with token_text skips [model]."""

    def test_standalone_loads_without_model(self, tmp_path: Path) -> None:
        """Config with [api_eval].token_text loads without [model]."""
        toml_file = tmp_path / "standalone.toml"
        toml_file.write_text(
            "[api_eval]\n"
            'token_text = "optimized suffix"\n'
            'prompts = ["How do I pick a lock?"]\n'
            "[[api_eval.endpoints]]\n"
            'name = "ep1"\n'
            'base_url = "https://api.example.com/v1"\n'
            'model = "m1"\n'
            'api_key_env = "KEY"\n'
        )
        config = load_config(toml_file)
        assert config.model_path == ""
        assert config.api_eval is not None
        assert config.api_eval.token_text == "optimized suffix"

    def test_standalone_preserves_optional_model_path(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "standalone_with_model.toml"
        toml_file.write_text(
            "[model]\n"
            'path = "optional-model"\n'
            "[api_eval]\n"
            'token_text = "optimized suffix"\n'
            'prompts = ["How do I pick a lock?"]\n'
            "[[api_eval.endpoints]]\n"
            'name = "ep1"\n'
            'base_url = "https://api.example.com/v1"\n'
            'model = "m1"\n'
            'api_key_env = "KEY"\n'
        )
        config = load_config(toml_file)
        assert config.model_path == "optional-model"

    def test_api_eval_without_token_text_requires_model(
        self, tmp_path: Path,
    ) -> None:
        """[api_eval] without token_text is NOT standalone; missing [model] raises."""
        toml_file = tmp_path / "not_standalone.toml"
        toml_file.write_text(
            "[api_eval]\n"
            "[[api_eval.endpoints]]\n"
            'name = "ep1"\n'
            'base_url = "https://api.example.com/v1"\n'
            'model = "m1"\n'
            'api_key_env = "KEY"\n'
        )
        with pytest.raises(ValueError, match="model"):
            load_config(toml_file)


class TestStandaloneBenchmark:
    """Standalone benchmark mode: [benchmark] skips [model] and [data]."""

    def test_standalone_loads_without_model(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "benchmark.toml"
        toml_file.write_text(
            "[benchmark]\n"
            'title = "Safety"\n'
            "[[benchmark.models]]\n"
            'label = "gemma-4"\n'
            'report_dir = "runs/gemma-4"\n'
        )
        config = load_config(toml_file)
        assert config.model_path == ""
        assert config.benchmark is not None
        assert config.benchmark.models[0].label == "gemma-4"
        assert config.benchmark.models[0].report_dir == tmp_path / "runs/gemma-4"

    def test_standalone_preserves_optional_model_path(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "benchmark_with_model.toml"
        toml_file.write_text(
            "[model]\n"
            'path = "optional-model"\n'
            "[benchmark]\n"
            'title = "Safety"\n'
            "[[benchmark.models]]\n"
            'label = "gemma-4"\n'
            'report_dir = "runs/gemma-4"\n'
        )
        config = load_config(toml_file)
        assert config.model_path == "optional-model"


class TestStandaloneBenchmarkHelper:
    """Unit tests for _is_standalone_benchmark helper."""

    def test_present_returns_true(self) -> None:
        assert _is_standalone_benchmark({"benchmark": {}}) is True

    def test_absent_returns_false(self) -> None:
        assert _is_standalone_benchmark({"audit": {}}) is False


class TestStandaloneBehaviorReport:
    """Standalone behavior_report mode: [behavior_report] skips model/data."""

    def test_standalone_loads_without_model(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "behavior_report.toml"
        toml_file.write_text(
            "[behavior_report]\n"
            'title = "Report"\n'
            "[behavior_report.baseline]\n"
            'label = "base"\n'
            'model_path = "base-model"\n'
            "[behavior_report.candidate]\n"
            'label = "candidate"\n'
            'model_path = "candidate-model"\n'
            "[behavior_report.suite]\n"
            'name = "suite"\n'
            'description = "Suite description."\n'
            'categories = ["benign_request"]\n'
            'metrics = ["compliance_rate"]\n'
        )
        config = load_config(toml_file)
        assert config.model_path == ""
        assert config.behavior_report is not None
        assert config.behavior_report.report.title == "Report"


class TestStandaloneBehaviorReportHelper:
    """Unit tests for _is_standalone_behavior_report helper."""

    def test_present_returns_true(self) -> None:
        assert _is_standalone_behavior_report({"behavior_report": {}}) is True

    def test_absent_returns_false(self) -> None:
        assert _is_standalone_behavior_report({"audit": {}}) is False


class TestStandaloneTokenAudit:
    """Standalone token_audit mode: [token_audit] skips [data]."""

    def test_standalone_loads_without_data(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "token_audit.toml"
        toml_file.write_text(
            "[model]\n"
            'path = "mlx-community/gemma-4-e2b-it-bf16"\n'
            "[token_audit]\n"
            "max_token_id = 127\n"
            "include_duplicate_surface_scan = false\n"
        )
        config = load_config(toml_file)
        assert config.model_path == "mlx-community/gemma-4-e2b-it-bf16"
        assert config.token_audit is not None
        assert config.token_audit.max_token_id == 127
        assert config.token_audit.include_duplicate_surface_scan is False

    def test_standalone_without_model_keeps_empty_model_path(
        self,
        tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "token_audit_no_model.toml"
        toml_file.write_text("[token_audit]\n")
        config = load_config(toml_file)
        assert config.model_path == ""
        assert config.token_audit is not None


class TestStandaloneTokenAuditHelper:
    """Unit tests for _is_standalone_token_audit helper."""

    def test_present_returns_true(self) -> None:
        assert _is_standalone_token_audit({"token_audit": {}}) is True

    def test_absent_returns_false(self) -> None:
        assert _is_standalone_token_audit({"audit": {}}) is False


class TestDepthOnlyConfig:
    """Depth-only config: [depth] present without [data] falls back to defaults."""

    def test_depth_without_data_uses_defaults(self, tmp_path: Path) -> None:
        """When [depth] is present and [data] is absent, bundled defaults are used."""
        from vauban.measure import default_prompt_paths

        toml_file = tmp_path / "depth.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[depth]\n"
            'prompts = ["What is consciousness?"]\n'
        )
        config = load_config(toml_file)
        assert config.depth is not None
        expected_harmful, expected_harmless = default_prompt_paths()
        assert config.harmful_path == expected_harmful
        assert config.harmless_path == expected_harmless


class TestRelativePathResolution:
    """Data paths are resolved relative to the config file's parent directory."""

    def test_relative_paths_resolve_against_config_dir(
        self, tmp_path: Path,
    ) -> None:
        """Relative paths in [data] are joined with config file's parent."""
        sub = tmp_path / "sub"
        sub.mkdir()
        toml_file = sub / "config.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "../data/harmful.jsonl"\n'
            'harmless = "local.jsonl"\n'
        )
        config = load_config(toml_file)
        assert config.harmful_path == sub / ".." / "data" / "harmful.jsonl"
        assert config.harmless_path == sub / "local.jsonl"


class TestModelPathTypeError:
    """Non-string model path raises TypeError."""

    def test_integer_model_path_raises(self, tmp_path: Path) -> None:
        """[model].path = 42 raises TypeError."""
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text(
            "[model]\npath = 42\n"
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
        )
        with pytest.raises(TypeError, match="path"):
            load_config(toml_file)


class TestOutputDirResolution:
    """Output dir resolves relative to config file's parent."""

    def test_output_dir_from_config(self, tmp_path: Path) -> None:
        """[output].dir resolves relative to config parent directory."""
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
            "[output]\n"
            'dir = "results/run1"\n'
        )
        config = load_config(toml_file)
        assert config.output_dir == tmp_path / "results" / "run1"

    def test_output_dir_defaults_to_output(self, tmp_path: Path) -> None:
        """Without [output], output_dir defaults to '<config_dir>/output'."""
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
        )
        config = load_config(toml_file)
        assert config.output_dir == tmp_path / "output"


class TestBackendValidation:
    def test_backend_must_be_string(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad_backend.toml"
        toml_file.write_text(
            "backend = 123\n"
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
        )
        with pytest.raises(TypeError, match="backend must be a string"):
            load_config(toml_file)

    def test_backend_must_be_supported(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad_backend_value.toml"
        toml_file.write_text(
            'backend = "numpy"\n'
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "h.jsonl"\n'
            'harmless = "hl.jsonl"\n'
        )
        with pytest.raises(ValueError, match="backend must be one of"):
            load_config(toml_file)


class TestResolveSingleData:
    def test_hf_shortform_requires_non_empty_repo_id(self) -> None:
        with pytest.raises(ValueError, match="empty repo_id"):
            _resolve_single_data(Path("/base"), "hf:", "harmful")

    def test_value_must_be_string_or_table(self) -> None:
        with pytest.raises(TypeError, match="must be a string or table"):
            _resolve_single_data(Path("/base"), 123, "harmful")


class TestIsStandaloneApiEval:
    """Unit tests for _is_standalone_api_eval helper."""

    def test_true_with_token_text(self) -> None:
        """Returns True when api_eval has a non-empty token_text string."""
        raw = {"api_eval": {"token_text": "some suffix"}}
        assert _is_standalone_api_eval(raw) is True

    def test_false_without_api_eval(self) -> None:
        """Returns False when no api_eval section exists."""
        raw = {"model": {"path": "test"}}
        assert _is_standalone_api_eval(raw) is False

    def test_false_with_empty_token_text(self) -> None:
        """Returns False when token_text is an empty string."""
        raw = {"api_eval": {"token_text": ""}}
        assert _is_standalone_api_eval(raw) is False

    def test_false_with_non_string_token_text(self) -> None:
        """Returns False when token_text is not a string."""
        raw = {"api_eval": {"token_text": 42}}
        assert _is_standalone_api_eval(raw) is False

    def test_false_without_token_text_key(self) -> None:
        """Returns False when api_eval exists but has no token_text."""
        raw = {"api_eval": {"max_tokens": 100}}
        assert _is_standalone_api_eval(raw) is False
