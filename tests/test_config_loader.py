# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.config._loader: load_config and _is_standalone_api_eval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vauban.config import load_config
from vauban.config._loader import _is_standalone_api_eval

if TYPE_CHECKING:
    from pathlib import Path


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
