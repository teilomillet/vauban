# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for ``vauban.circuit`` coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from vauban import _ops as ops
from vauban.circuit import (
    _compute_effect,
    _forward_cache_components,
    _match_seq_len,
    _patched_forward_component,
    _trace_components,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM


@dataclass(frozen=True, slots=True)
class _FakeTransformer:
    """Minimal transformer with layers and identity norm."""

    layers: list[object]

    @staticmethod
    def norm(h: object) -> object:
        return h


class _LinearLayer:
    """Fake linear/SSM layer used to hit special-case branches."""

    is_linear = True

    def __call__(self, h: Array, _mask: object) -> Array:
        return h + 2.0


class _Model:
    """Model stub exposing a transformer."""

    def __init__(self) -> None:
        self.transformer = _FakeTransformer([_LinearLayer()])


class TestCircuitExtra:
    """Cover helper branches left out of the base circuit tests."""

    def test_match_seq_len_identity_branch(self) -> None:
        tensor = ops.ones((1, 3, 2))
        matched = _match_seq_len(tensor, 3)

        assert matched is tensor

    def test_forward_cache_components_linear_layer_records_delta(self) -> None:
        model = _Model()
        token_ids = ops.array([[1, 2]])
        hidden = ops.ones((1, 2, 3))
        logits = ops.zeros((1, 2, 4))

        with (
            patch("vauban.circuit.get_transformer", return_value=model.transformer),
            patch("vauban.circuit.embed_and_mask", return_value=(hidden, None)),
            patch("vauban.circuit.make_ssm_mask", return_value=None),
            patch("vauban.circuit.select_mask", return_value=None),
            patch("vauban.circuit.lm_head_forward", return_value=logits),
        ):
            out_logits, components = _forward_cache_components(
                cast("CausalLM", model),
                token_ids,
            )

        assert out_logits is logits
        assert len(components) == 1
        delta, mlp = components[0]
        assert delta.tolist() == (hidden + 2.0 - hidden).tolist()
        assert mlp.tolist() == ops.zeros_like(delta).tolist()

    def test_patched_forward_component_linear_attn_branch(self) -> None:
        model = _Model()
        token_ids = ops.array([[1, 2]])
        hidden = ops.ones((1, 2, 3))
        logits = ops.zeros((1, 2, 4))
        clean_component = ops.ones((1, 1, 3))

        with (
            patch("vauban.circuit.get_transformer", return_value=model.transformer),
            patch("vauban.circuit.embed_and_mask", return_value=(hidden, None)),
            patch("vauban.circuit.make_ssm_mask", return_value=None),
            patch("vauban.circuit.select_mask", return_value=None),
            patch("vauban.circuit.lm_head_forward", return_value=logits),
        ):
            out = _patched_forward_component(
                cast("CausalLM", model),
                token_ids,
                0,
                "attn",
                clean_component,
            )

        assert out is logits

    def test_trace_components_records_direction_attributions(self) -> None:
        corrupt_logits = ops.zeros((1, 1, 2))
        attn = ops.ones((1, 1, 3))
        mlp = ops.ones((1, 1, 3)) * 2.0
        direction = ops.array([1.0, 0.0, 0.0])

        with (
            patch(
                "vauban.circuit._forward_cache_components",
                side_effect=[
                    (corrupt_logits, [(attn, mlp)]),
                    (corrupt_logits, [(attn, mlp)]),
                ],
            ),
            patch(
                "vauban.circuit._patched_forward_component",
                return_value=corrupt_logits,
            ),
        ):
            effects, attributions = _trace_components(
                cast("CausalLM", object()),
                ops.array([[1]]),
                ops.array([[2]]),
                [0],
                "kl",
                -1,
                direction,
                True,
                None,
            )

        assert effects[(0, "attn")] == pytest.approx(0.0)
        assert effects[(0, "mlp")] == pytest.approx(0.0)
        assert attributions[(0, "attn")] == pytest.approx(1.0)
        assert attributions[(0, "mlp")] == pytest.approx(2.0)

    def test_compute_effect_requires_logit_diff_tokens(self) -> None:
        logits = ops.zeros((1, 1, 2))

        with pytest.raises(ValueError, match="logit_diff_tokens"):
            _compute_effect(logits, logits, "logit_diff", -1, None)
