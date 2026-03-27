# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban._model_io: backend-dispatched model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vauban._backend import get_backend

# ===================================================================
# MLX backend tests
# ===================================================================


class TestLoadModelMLX:
    """load_model delegates to mlx_lm.load on the MLX backend."""

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
