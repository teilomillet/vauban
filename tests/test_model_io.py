# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban._model_io: backend-dispatched model loading."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import vauban._model_io as model_io
from vauban._backend import get_backend


class FakeMlxLmModule(ModuleType):
    """Typed fake module for ``mlx_lm``."""

    load: object


class FakeTorchModule(ModuleType):
    """Typed fake module for ``torch``."""

    float16: object


class FakeTransformersModule(ModuleType):
    """Typed fake module for ``transformers``."""

    AutoModelForCausalLM: object
    AutoTokenizer: object


class FakeTorchWrapperModule(ModuleType):
    """Typed fake module for ``vauban._model_torch``."""

    TorchCausalLMWrapper: object

# ===================================================================
# MLX backend tests
# ===================================================================


class TestLoadModelMLX:
    """load_model delegates to mlx_lm.load on the MLX backend."""

    def test_helper_calls_mlx_lm_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mlx_lm = FakeMlxLmModule("mlx_lm")
        mock_load = MagicMock(return_value=(mock_model, mock_tokenizer, object()))
        mlx_lm.load = mock_load
        monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)

        model, tokenizer = model_io._load_model_mlx("test-model-path")

        mock_load.assert_called_once_with("test-model-path")
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    @pytest.mark.skipif(
        get_backend() != "mlx",
        reason="MLX-specific test",
    )
    def test_calls_mlx_lm_load(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch(
            "mlx_lm.load",
            return_value=(mock_model, mock_tokenizer),
        ) as mock_load:
            from vauban._model_io import load_model

            model, tokenizer = load_model("test-model-path")
            mock_load.assert_called_once_with("test-model-path")
            assert model is mock_model
            assert tokenizer is mock_tokenizer

    @pytest.mark.skipif(
        get_backend() != "mlx",
        reason="MLX-specific test",
    )
    def test_returns_tuple(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            from vauban._model_io import load_model

            result = load_model("test-model")
            assert isinstance(result, tuple)
            assert len(result) == 2

    @pytest.mark.skipif(
        get_backend() != "mlx",
        reason="MLX-specific test",
    )
    def test_handles_extra_return_values(self) -> None:
        """mlx_lm.load may return extra values (config, etc.)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_config = MagicMock()
        with patch(
            "mlx_lm.load",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            from vauban._model_io import load_model

            model, tokenizer = load_model("test-model")
            assert model is mock_model
            assert tokenizer is mock_tokenizer


# ===================================================================
# Torch backend tests (gated)
# ===================================================================


class TestLoadModelTorch:
    """Torch backend load_model tests (skipped if torch unavailable)."""

    def test_helper_calls_hf_from_pretrained(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_torch = FakeTorchModule("torch")
        float16 = object()
        fake_torch.float16 = float16

        mock_hf_model = MagicMock()
        mock_hf_model.eval.return_value = None
        mock_tokenizer = MagicMock()
        auto_model = MagicMock()
        auto_model.from_pretrained.return_value = mock_hf_model
        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        transformers = FakeTransformersModule("transformers")
        transformers.AutoModelForCausalLM = auto_model
        transformers.AutoTokenizer = auto_tokenizer

        wrapper_module = FakeTorchWrapperModule("vauban._model_torch")
        mock_wrapper = MagicMock(return_value="wrapped-model")
        wrapper_module.TorchCausalLMWrapper = mock_wrapper

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.setitem(sys.modules, "vauban._model_torch", wrapper_module)

        model, tokenizer = model_io._load_model_torch("test-model")

        auto_model.from_pretrained.assert_called_once_with(
            "test-model",
            dtype=float16,
            device_map="auto",
        )
        mock_hf_model.eval.assert_called_once_with()
        auto_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_wrapper.assert_called_once_with(mock_hf_model)
        assert model == "wrapped-model"
        assert tokenizer is mock_tokenizer

    @pytest.mark.skipif(
        get_backend() != "torch",
        reason="Torch-specific test",
    )
    def test_calls_hf_from_pretrained(self) -> None:
        pytest.importorskip("torch")
        mock_hf_model = MagicMock()
        mock_hf_model.eval.return_value = mock_hf_model
        mock_tokenizer = MagicMock()

        with (
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=mock_hf_model,
            ) as mock_load,
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            from vauban._model_io import load_model

            _model, tokenizer = load_model("test-model")
            mock_load.assert_called_once()
            assert tokenizer is mock_tokenizer


# ===================================================================
# Backend detection
# ===================================================================


class TestBackendDetection:
    """The model_io module resolves the correct backend."""

    def test_backend_is_known(self) -> None:
        backend = get_backend()
        assert backend in ("mlx", "torch")

    def test_load_model_importable(self) -> None:
        from vauban._model_io import load_model

        assert callable(load_model)

    def test_load_model_dispatches_to_mlx_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        monkeypatch.setattr(model_io, "_BACKEND", "mlx")
        monkeypatch.setattr(
            model_io,
            "_load_model_mlx",
            lambda model_path: (mock_model, mock_tokenizer),
        )

        model, tokenizer = model_io.load_model("model-id")

        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_model_dispatches_to_torch_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        monkeypatch.setattr(model_io, "_BACKEND", "torch")
        monkeypatch.setattr(
            model_io,
            "_load_model_torch",
            lambda model_path: (mock_model, mock_tokenizer),
        )

        model, tokenizer = model_io.load_model("model-id")

        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_model_rejects_unknown_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(model_io, "_BACKEND", "unknown")

        with pytest.raises(ValueError, match="Unknown backend"):
            model_io.load_model("model-id")
