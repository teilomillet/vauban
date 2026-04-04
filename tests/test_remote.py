# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the [remote] config parser, mode runner, and probe orchestrator."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, cast
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


class _FakeArrayWithToList:
    """Activation container that exposes ``tolist`` like numpy arrays."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        """Return nested list data."""
        return self._rows


class _FakeArrayIterable:
    """Activation container that only supports iteration."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def __iter__(self) -> object:
        """Yield rows so ``list(arr)`` produces nested lists."""
        return iter(self._rows)


class _FakeJsinferMessage:
    """Minimal jsinfer message stub."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class _FakeChatCompletionRequest:
    """Minimal jsinfer chat request stub."""

    def __init__(self, custom_id: str, messages: list[_FakeJsinferMessage]) -> None:
        self.custom_id = custom_id
        self.messages = messages


class _FakeActivationsRequest:
    """Minimal jsinfer activation request stub."""

    def __init__(
        self,
        custom_id: str,
        messages: list[_FakeJsinferMessage],
        module_names: list[str],
    ) -> None:
        self.custom_id = custom_id
        self.messages = messages
        self.module_names = module_names


class _FakeChatResponse:
    """Chat completion result wrapper."""

    def __init__(self, content: str) -> None:
        self.messages = [_FakeJsinferMessage(role="assistant", content=content)]


class _FakeActivationResponse:
    """Activation result wrapper."""

    def __init__(self, activations: dict[str, object]) -> None:
        self.activations = activations


class _FakeBatchInferenceClient:
    """BatchInferenceClient stub used to exercise lazy imports."""

    instances: ClassVar[list[_FakeBatchInferenceClient]] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.chat_requests: list[_FakeChatCompletionRequest] = []
        self.chat_model: str | None = None
        self.activation_requests: list[_FakeActivationsRequest] = []
        self.activation_model: str | None = None
        self.__class__.instances.append(self)

    async def chat_completions(
        self,
        requests: list[_FakeChatCompletionRequest],
        *,
        model: str,
    ) -> dict[str, _FakeChatResponse]:
        """Record chat requests and omit one response on purpose."""
        self.chat_requests = requests
        self.chat_model = model
        return {"p000": _FakeChatResponse("reply-0")}

    async def activations(
        self,
        requests: list[_FakeActivationsRequest],
        *,
        model: str,
    ) -> dict[str, _FakeActivationResponse]:
        """Record activation requests and omit one response on purpose."""
        self.activation_requests = requests
        self.activation_model = model
        return {
            "act_000": _FakeActivationResponse({
                "layer.0": _FakeArrayWithToList([[1.0, 2.0]]),
                "layer.1": _FakeArrayIterable([[3.0, 4.0]]),
            }),
        }


def _make_fake_jsinfer_module() -> ModuleType:
    """Build a fake jsinfer module with the symbols the backend imports."""
    module = ModuleType("jsinfer")
    module.BatchInferenceClient = _FakeBatchInferenceClient  # type: ignore[attr-defined]
    module.ChatCompletionRequest = _FakeChatCompletionRequest  # type: ignore[attr-defined]
    module.ActivationsRequest = _FakeActivationsRequest  # type: ignore[attr-defined]
    module.Message = _FakeJsinferMessage  # type: ignore[attr-defined]
    return module


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

    def test_missing_remote_config_raises(self, tmp_path: Path) -> None:
        """Remote mode requires an explicit [remote] section."""
        from vauban._pipeline._mode_remote import _run_remote_mode

        ctx = MagicMock()
        ctx.config.remote = None
        ctx.config.verbose = False
        ctx.config.output_dir = tmp_path
        ctx.config_path = str(tmp_path / "test.toml")
        ctx.t0 = 0.0

        with pytest.raises(ValueError, match=r"\[remote\] section is required"):
            _run_remote_mode(ctx)


class TestRemoteModeHelpers:
    """Helper coverage for dotenv loading and summary rendering."""

    def test_load_dotenv_manual_parser_handles_quotes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When python-dotenv is absent, .env should still be parsed safely."""
        from vauban._pipeline._mode_remote import _load_dotenv

        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        (tmp_path / ".env").write_text(
            "A=\"quoted\"\n"
            "B='single'\n"
            "EMPTY=\n"
            "# comment\n"
            "EXISTING=ignored\n",
        )
        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        monkeypatch.delenv("EMPTY", raising=False)
        monkeypatch.setenv("EXISTING", "kept")

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_dotenv(config_path)

        assert os.environ["A"] == "quoted"
        assert os.environ["B"] == "single"
        assert os.environ["EMPTY"] == ""
        assert os.environ["EXISTING"] == "kept"

    def test_load_dotenv_uses_python_dotenv_when_available(
        self,
        tmp_path: Path,
    ) -> None:
        """If python-dotenv is importable, its loader should be used."""
        from vauban._pipeline._mode_remote import _load_dotenv

        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        env_path = tmp_path / ".env"
        env_path.write_text("KEY=value\n")

        fake_dotenv = ModuleType("dotenv")
        calls: list[Path] = []

        def _fake_load_dotenv(path: Path) -> None:
            calls.append(path)

        fake_dotenv.load_dotenv = _fake_load_dotenv  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            _load_dotenv(config_path)

        assert calls == [env_path]

    def test_print_summary_renders_errors_and_previews(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Summary output should include both error and response previews."""
        from vauban._pipeline._mode_remote import _print_summary

        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="KEY",
            models=["m1", "m2", "missing"],
            prompts=["hello"],
        )
        result: dict[str, object] = {
            "models": {
                "m1": {
                    "responses": [
                        {"prompt": "p1", "response": "hello\nworld"},
                        {"prompt": "p2", "error": "timeout"},
                        "skip-me",
                    ],
                    "activation_files": ["a.npy"],
                },
                "m2": {
                    "responses": [],
                    "activation_files": [],
                },
            },
        }

        _print_summary(cfg, result)

        output = capsys.readouterr().err
        assert "Model" in output
        assert "m1" in output
        assert "ERROR: timeout" in output
        assert "hello world" in output

    def test_print_summary_ignores_non_mapping_results(self) -> None:
        """Non-dict model payloads should be ignored cleanly."""
        from vauban._pipeline._mode_remote import _print_summary

        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="KEY",
            models=["m1"],
            prompts=["hello"],
        )

        _print_summary(cfg, {"models": []})

    def test_print_summary_skips_models_without_response_lists(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Preview output should skip dict entries that lack a list of responses."""
        from vauban._pipeline._mode_remote import _print_summary

        cfg = RemoteConfig(
            backend="jsinfer",
            api_key_env="KEY",
            models=["m1"],
            prompts=["hello"],
        )

        _print_summary(cfg, {"models": {"m1": {"responses": "not-a-list"}}})

        output = capsys.readouterr().err
        assert "Model" in output
        assert "── m1 ──" not in output


class TestJsinferBackend:
    """Tests for the lazy-import jsinfer backend wrapper."""

    def test_factory_returns_backend(self) -> None:
        from vauban.remote._jsinfer import JsinferBackend, create_jsinfer_backend

        backend = create_jsinfer_backend("secret-key")

        assert isinstance(backend, JsinferBackend)
        assert backend._api_key == "secret-key"

    def test_chat_builds_requests_and_handles_missing_responses(self) -> None:
        from vauban.remote._jsinfer import JsinferBackend

        _FakeBatchInferenceClient.instances.clear()
        backend = JsinferBackend("secret-key")

        with patch.dict(sys.modules, {"jsinfer": _make_fake_jsinfer_module()}):
            results = asyncio.run(
                backend.chat(
                    model_id="demo-model",
                    prompts=["hello", "bye"],
                    max_tokens=32,
                ),
            )

        client = _FakeBatchInferenceClient.instances[0]
        assert client.api_key == "secret-key"
        assert client.chat_model == "demo-model"
        assert [
            request.custom_id for request in client.chat_requests
        ] == ["p000", "p001"]
        assert client.chat_requests[0].messages[0].content == "hello"
        assert results == [
            RemoteChatResult(prompt="hello", response="reply-0"),
            RemoteChatResult(prompt="bye", error="no response"),
        ]

    def test_activations_convert_arrays_for_serialization(self) -> None:
        from vauban.remote._jsinfer import JsinferBackend

        _FakeBatchInferenceClient.instances.clear()
        backend = JsinferBackend("secret-key")

        with patch.dict(sys.modules, {"jsinfer": _make_fake_jsinfer_module()}):
            results = asyncio.run(
                backend.activations(
                    model_id="demo-model",
                    prompts=["hello", "bye"],
                    modules=["layer.0", "layer.1"],
                ),
            )

        client = _FakeBatchInferenceClient.instances[0]
        assert client.activation_model == "demo-model"
        assert [request.custom_id for request in client.activation_requests] == [
            "act_000",
            "act_001",
        ]
        assert client.activation_requests[0].module_names == ["layer.0", "layer.1"]
        assert results == [
            RemoteActivationResult(
                prompt_index=0,
                module_name="layer.0",
                data=[[1.0, 2.0]],
            ),
            RemoteActivationResult(
                prompt_index=0,
                module_name="layer.1",
                data=[[3.0, 4.0]],
            ),
        ]


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
        models_dict = cast("dict[str, object]", models)
        entry = models_dict["test-model"]
        assert isinstance(entry, dict)
        entry_dict = cast("dict[str, object]", entry)
        responses = entry_dict["responses"]
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
        models_dict = cast("dict[str, object]", models)
        entry = models_dict["test/model"]
        assert isinstance(entry, dict)
        entry_dict = cast("dict[str, object]", entry)
        act_files = entry_dict["activation_files"]
        assert isinstance(act_files, list)
        assert len(act_files) == 1

        # File exists on disk
        filepath = tmp_path / cast("str", act_files[0])
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
        models_dict = cast("dict[str, object]", models)
        entry = models_dict["m"]
        assert isinstance(entry, dict)
        entry_dict = cast("dict[str, object]", entry)
        responses = entry_dict["responses"]
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

        register_backend("test_custom", _factory)
        assert "test_custom" in _REGISTRY

        # Cleanup
        del _REGISTRY["test_custom"]
