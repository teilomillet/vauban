# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for ``vauban.linear_probe`` coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import MockTokenizer
from vauban import _ops as ops
from vauban.linear_probe import _bce_loss, _sigmoid, _train_single_probe, train_probe
from vauban.types import LinearProbeConfig

if TYPE_CHECKING:
    from vauban.types import CausalLM


class _ModelWithPath:
    """Simple model stub exposing a configurable model path."""

    def __init__(self, model_path: object) -> None:
        self._model_path = model_path


class TestTrainProbeExtra:
    """Cover train_probe branches not hit by the base tests."""

    def test_validates_prompt_inputs(self) -> None:
        config = LinearProbeConfig(layers=[0])
        model = cast("CausalLM", _ModelWithPath("model"))
        tokenizer = MockTokenizer(32)

        with pytest.raises(ValueError, match="harmful_prompts must be non-empty"):
            train_probe(model, tokenizer, [], ["safe"], config)

        with pytest.raises(ValueError, match="harmless_prompts must be non-empty"):
            train_probe(model, tokenizer, ["harm"], [], config)

    def test_rejects_out_of_range_layers(self) -> None:
        activations = [ops.zeros((2, 4))]
        config = LinearProbeConfig(layers=[1], n_epochs=0)

        with (
            patch(
                "vauban.linear_probe._collect_per_prompt_activations",
                return_value=activations,
            ),
            pytest.raises(ValueError, match="out of range"),
        ):
            train_probe(
                cast("CausalLM", _ModelWithPath("model")),
                MockTokenizer(32),
                ["harm"],
                ["safe"],
                config,
            )

    def test_falls_back_to_unknown_model_path_and_zero_epoch_loss(self) -> None:
        activations = [
            ops.array([
                [2.0, 2.0, 2.0, 2.0],
                [-2.0, -2.0, -2.0, -2.0],
            ]),
        ]
        config = LinearProbeConfig(layers=[0], n_epochs=0)

        with patch(
            "vauban.linear_probe._collect_per_prompt_activations",
            return_value=activations,
        ):
            result = train_probe(
                cast("CausalLM", _ModelWithPath(object())),
                MockTokenizer(32),
                ["harm"],
                ["safe"],
                config,
            )

        assert result.model_path == "unknown"
        assert result.d_model == 4
        assert len(result.layers) == 1
        assert result.layers[0].loss == 0.0
        assert result.layers[0].loss_history == []


class TestLinearProbeHelpers:
    """Small helper coverage for the probe math."""

    def test_train_single_probe_zero_epochs_uses_empty_history_branch(self) -> None:
        activations = ops.array([[1.0, 0.0], [0.0, 1.0]])
        labels = ops.array([1.0, 0.0])

        accuracy, final_loss, history = _train_single_probe(
            activations,
            labels,
            n_epochs=0,
            learning_rate=0.1,
            batch_size=1,
            regularization=0.0,
        )

        assert accuracy == pytest.approx(0.5)
        assert final_loss == 0.0
        assert history == []

    def test_sigmoid_and_bce_loss_return_finite_values(self) -> None:
        logits = ops.array([-2.0, 0.0, 2.0])
        probs = _sigmoid(logits)
        targets = ops.array([0.0, 1.0, 1.0])
        loss = _bce_loss(probs, targets)
        ops.eval(probs, loss)

        prob_list = [round(float(probs[i].item()), 3) for i in range(3)]
        assert prob_list == [0.119, 0.5, 0.881]
        assert float(loss.item()) > 0.0

    def test_train_single_probe_accepts_cuda_activations(self) -> None:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        activations = torch.randn((8, 4), device="cuda")
        labels = torch.tensor(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            device="cuda",
        )

        accuracy, _final_loss, history = _train_single_probe(
            activations,
            labels,
            n_epochs=1,
            learning_rate=0.01,
            batch_size=4,
            regularization=0.0,
        )

        assert 0.0 <= accuracy <= 1.0
        assert len(history) == 1
