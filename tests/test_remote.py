"""Tests for the [remote] config parser, mode runner, and probe orchestrator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vauban.config import load_config
from vauban.config._parse_remote import _parse_remote
from vauban.types import RemoteActivationResult, RemoteChatResult, RemoteConfig

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


# ── Fake backend for probe tests ─────────────────────────────────────


class _FakeBackend:
    """Test backend implementing the RemoteBackend protocol."""

    def __init__(self, responses: list[RemoteChatResult] | None = None) -> None:
        self.chat_calls: list[tuple[str, list[str], int]] = []
        self.act_calls: list[tuple[str, list[str], list[str]]] = []
        self._responses = responses

    async def chat(
        self,
        model_id: str,
        prompts: list[str],
        max_tokens: int,
    ) -> list[RemoteChatResult]:
        """Record call and return canned responses."""
        self.chat_calls.append((model_id, prompts, max_tokens))
        if self._responses is not None:
            return self._responses
        return [RemoteChatResult(prompt=p, response="Hi there") for p in prompts]

    async def activations(
        self,
        model_id: str,
        prompts: list[str],
        modules: list[str],
    ) -> list[RemoteActivationResult]:
        """Record call and return synthetic activations."""
        self.act_calls.append((model_id, prompts, modules))
        return [
            RemoteActivationResult(prompt_index=i, module_name=m, data=[[1.0, 2.0]])
            for i in range(len(prompts))
            for m in modules
        ]


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
    @patch("vauban.remote.run_remote_probe")
    def test_writes_report_and_logs(
        self,
        mock_probe: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_probe.return_value = {
            "backend": "jsinfer",
            "n_models": 1,
            "n_prompts": 1,
            "activations_requested": False,
            "models": {
                "test-model": {
                    "model_id": "test-model",
                    "responses": [{"prompt": "Hello", "response": "Hi there"}],
                },
            },
        }

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


# ── Probe orchestrator tests ─────────────────────────────────────────


class TestRunRemoteProbe:
    def test_chat_responses_collected(self, tmp_path: Path) -> None:
        """Verify prompts reach the backend and responses appear in report."""
        backend = _FakeBackend()
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["test-model"],
            prompts=["Hello", "World"],
        )

        from vauban.remote import run_remote_probe

        result = run_remote_probe(
            cfg=cfg,
            api_key="fake",
            output_dir=tmp_path,
            verbose=False,
            backend=backend,
        )

        # Backend received the call
        assert len(backend.chat_calls) == 1
        assert backend.chat_calls[0][0] == "test-model"
        assert backend.chat_calls[0][1] == ["Hello", "World"]

        # Report has correct shape
        assert result["n_models"] == 1
        assert result["n_prompts"] == 2
        assert isinstance(result["models"], dict)

        models = result["models"]
        assert isinstance(models, dict)
        entry = models["test-model"]
        assert isinstance(entry, dict)
        responses = entry["responses"]
        assert isinstance(responses, list)
        assert len(responses) == 2
        assert responses[0] == {"prompt": "Hello", "response": "Hi there"}
        assert responses[1] == {"prompt": "World", "response": "Hi there"}

    def test_max_tokens_forwarded(self, tmp_path: Path) -> None:
        """Verify max_tokens from config reaches the backend."""
        backend = _FakeBackend()
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["m"],
            prompts=["p"],
            max_tokens=42,
        )

        from vauban.remote import run_remote_probe

        run_remote_probe(
            cfg=cfg,
            api_key="fake",
            output_dir=tmp_path,
            verbose=False,
            backend=backend,
        )

        assert backend.chat_calls[0][2] == 42

    def test_activations_saved_as_npy(self, tmp_path: Path) -> None:
        """Verify activation files are written when activations are requested."""
        backend = _FakeBackend()
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["test/model"],
            prompts=["Hello"],
            activations=True,
            activation_layers=[0],
            activation_modules=["model.layers.{layer}.mlp.down_proj"],
        )

        from vauban.remote import run_remote_probe

        result = run_remote_probe(
            cfg=cfg,
            api_key="fake",
            output_dir=tmp_path,
            verbose=False,
            backend=backend,
        )

        # Backend received activation call
        assert len(backend.act_calls) == 1
        assert backend.act_calls[0][2] == ["model.layers.0.mlp.down_proj"]

        # Files were saved
        models = result["models"]
        assert isinstance(models, dict)
        entry = models["test/model"]
        assert isinstance(entry, dict)
        act_files = entry["activation_files"]
        assert isinstance(act_files, list)
        assert len(act_files) == 1

        # File exists on disk
        filepath = tmp_path / act_files[0]
        assert filepath.exists()

        import numpy as np

        arr = np.load(str(filepath))
        assert arr.shape == (1, 2)
        assert arr.dtype == np.float32

    def test_error_responses_preserved(self, tmp_path: Path) -> None:
        """Verify error results appear in the report."""
        backend = _FakeBackend(responses=[
            RemoteChatResult(prompt="Hello", error="timeout"),
        ])
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["m"],
            prompts=["Hello"],
        )

        from vauban.remote import run_remote_probe

        result = run_remote_probe(
            cfg=cfg,
            api_key="fake",
            output_dir=tmp_path,
            verbose=False,
            backend=backend,
        )

        models = result["models"]
        assert isinstance(models, dict)
        entry = models["m"]
        assert isinstance(entry, dict)
        responses = entry["responses"]
        assert isinstance(responses, list)
        assert responses[0] == {"prompt": "Hello", "error": "timeout"}

    def test_multi_model_probe(self, tmp_path: Path) -> None:
        """Verify all models are probed."""
        backend = _FakeBackend()
        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["model-a", "model-b", "model-c"],
            prompts=["Hi"],
        )

        from vauban.remote import run_remote_probe

        result = run_remote_probe(
            cfg=cfg,
            api_key="fake",
            output_dir=tmp_path,
            verbose=False,
            backend=backend,
        )

        assert result["n_models"] == 3
        assert len(backend.chat_calls) == 3

        models = result["models"]
        assert isinstance(models, dict)
        assert set(models.keys()) == {"model-a", "model-b", "model-c"}


# ── Registry tests ───────────────────────────────────────────────────


class TestBackendRegistry:
    def test_get_unknown_backend_raises(self) -> None:
        from vauban.remote._registry import get_backend

        with pytest.raises(ValueError, match="Unknown remote backend 'nonexistent'"):
            get_backend("nonexistent", "key")

    def test_jsinfer_registered(self) -> None:
        from vauban.remote._registry import _REGISTRY

        assert "jsinfer" in _REGISTRY

    def test_register_custom_backend(self) -> None:
        from vauban.remote._registry import _REGISTRY, register_backend

        def _factory(api_key: str) -> _FakeBackend:
            return _FakeBackend()

        register_backend("test_custom", _factory)  # type: ignore[arg-type]
        assert "test_custom" in _REGISTRY

        # Cleanup
        del _REGISTRY["test_custom"]
