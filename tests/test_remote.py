"""Tests for the [remote] config parser and mode runner."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vauban.config import load_config
from vauban.config._parse_remote import _parse_remote
from vauban.types import RemoteConfig

# ── Parser tests ─────────────────────────────────────────────────────


class TestParseRemote:
    def test_absent_returns_none(self) -> None:
        assert _parse_remote({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_remote({"remote": "bad"})

    def test_loader_surfaces_remote_type_error_without_model(
        self,
        tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "remote_bad.toml"
        toml_file.write_text('remote = "bad"\n')

        with pytest.raises(TypeError, match=r"\[remote\] must be a table"):
            load_config(toml_file)

    def test_minimal_valid(self) -> None:
        cfg = _parse_remote({
            "remote": {
                "backend": "jsinfer",
                "api_key_env": "MY_KEY",
                "models": ["model-1"],
                "prompts": ["Hello"],
            },
        })
        assert cfg is not None
        assert cfg.backend == "jsinfer"
        assert cfg.api_key_env == "MY_KEY"
        assert cfg.models == ["model-1"]
        assert cfg.prompts == ["Hello"]
        assert cfg.activations is False
        assert cfg.activation_layers == []
        assert cfg.max_tokens == 512
        assert cfg.timeout == 600

    def test_full_config(self) -> None:
        cfg = _parse_remote({
            "remote": {
                "backend": "jsinfer",
                "api_key_env": "JS_KEY",
                "models": ["m1", "m2"],
                "prompts": ["Hi", "Bye"],
                "activations": True,
                "activation_layers": [0, 10, 20],
                "activation_modules": ["model.layers.{layer}.mlp"],
                "max_tokens": 256,
                "timeout": 300,
            },
        })
        assert cfg is not None
        assert cfg.activations is True
        assert cfg.activation_layers == [0, 10, 20]
        assert cfg.activation_modules == ["model.layers.{layer}.mlp"]
        assert cfg.max_tokens == 256
        assert cfg.timeout == 300

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            _parse_remote({
                "remote": {
                    "backend": "openai",
                    "api_key_env": "K",
                    "models": ["m"],
                    "prompts": ["p"],
                },
            })

    def test_empty_models(self) -> None:
        with pytest.raises(ValueError, match="models must be non-empty"):
            _parse_remote({
                "remote": {
                    "backend": "jsinfer",
                    "api_key_env": "K",
                    "models": [],
                    "prompts": ["p"],
                },
            })

    def test_empty_prompts(self) -> None:
        with pytest.raises(ValueError, match="prompts must be non-empty"):
            _parse_remote({
                "remote": {
                    "backend": "jsinfer",
                    "api_key_env": "K",
                    "models": ["m"],
                    "prompts": [],
                },
            })

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValueError, match=r"api_key_env.*required"):
            _parse_remote({
                "remote": {
                    "backend": "jsinfer",
                    "models": ["m"],
                    "prompts": ["p"],
                },
            })

    def test_frozen(self) -> None:
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["m"],
            prompts=["p"],
        )
        with pytest.raises(AttributeError):
            cfg.backend = "other"  # type: ignore[misc]


# ── Mode runner tests ────────────────────────────────────────────────


class TestRemoteModeRunner:
    def _make_context(
        self,
        remote_cfg: RemoteConfig,
        tmp_path: Path,
    ) -> MagicMock:
        """Build a minimal EarlyModeContext mock."""
        import time

        ctx = MagicMock()
        ctx.config.remote = remote_cfg
        ctx.config.verbose = False
        ctx.config.output_dir = tmp_path
        ctx.config.meta = None
        ctx.config.model_path = ""
        ctx.config_path = str(tmp_path / "test.toml")
        ctx.t0 = time.monotonic()
        return ctx

    @patch.dict("os.environ", {"TEST_KEY": "fake-api-key"})
    @patch("vauban.remote._query_model")
    def test_run_collects_responses(
        self,
        mock_query: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_query.return_value = [
            {"prompt": "Hello", "response": "Hi there"},
        ]

        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="TEST_KEY",
            models=["test-model"],
            prompts=["Hello"],
        )
        ctx = self._make_context(cfg, tmp_path)

        from vauban._pipeline._mode_remote import _run_remote_mode

        _run_remote_mode(ctx)

        report_path = tmp_path / "remote_report.json"
        assert report_path.exists()

        import json

        report = json.loads(report_path.read_text())
        assert report["backend"] == "jsinfer"
        assert report["n_models"] == 1
        assert report["n_prompts"] == 1

    def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="NONEXISTENT_KEY_12345",
            models=["m"],
            prompts=["p"],
        )
        ctx = self._make_context(cfg, tmp_path)

        from vauban._pipeline._mode_remote import _run_remote_mode

        with pytest.raises(ValueError, match="NONEXISTENT_KEY_12345"):
            _run_remote_mode(ctx)
