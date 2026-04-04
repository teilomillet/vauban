# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for ``vauban.measure._diff`` coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from vauban import _ops as ops
from vauban.measure._diff import (
    _find_sub,
    _get_proj_weight,
    _get_weight,
    measure_diff,
)

if TYPE_CHECKING:
    from vauban.types import CausalLM


class _Node:
    """Simple namespace-style object for test graphs."""

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True, slots=True)
class _WeightHolder:
    """Holder exposing a weight attribute."""

    weight: object


class TestMeasureDiffHelpers:
    """Helper coverage for submodule and weight probing."""

    def test_find_sub_and_get_proj_weight_handle_missing_values(self) -> None:
        attention = _Node(out_proj=_WeightHolder(weight=ops.eye(2)))
        layer = _Node(attention=attention)

        assert _find_sub(layer, ("self_attn", "attention")) is attention
        assert _find_sub(layer, ("missing",)) is None
        assert _get_proj_weight(attention, ("missing", "out_proj")) is not None
        assert _get_proj_weight(attention, ("missing",)) is None

    def test_get_weight_rejects_missing_or_shapeless_weights(self) -> None:
        module = _Node(
            present=_WeightHolder(weight=ops.eye(2)),
            shapeless=_WeightHolder(weight="bad"),
        )

        assert _get_weight(module, "present") is not None
        assert _get_weight(module, "missing") is None
        assert _get_weight(module, "shapeless") is None


class TestMeasureDiffExtra:
    """Cover the unusual measurement branches."""

    def test_handles_layers_with_no_matching_projections(self) -> None:
        base = _Node(layers=[_Node()])
        aligned = _Node(layers=[_Node()])

        with patch(
            "vauban.measure._diff.get_transformer",
            side_effect=lambda model: model,
        ):
            result = measure_diff(
                cast("CausalLM", base),
                cast("CausalLM", aligned),
                top_k=2,
            )

        assert result.best_layer == 0
        assert result.basis.shape == (2, 1)
        assert result.singular_values == [0.0, 0.0]
        assert result.explained_variance == [0.0]

    def test_handles_moe_shape_and_padding_without_attention_projection(self) -> None:
        base_layer = _Node(
            mlp=_Node(down_proj=_WeightHolder(weight=ops.zeros((2, 3, 4)))),
        )
        aligned_layer = _Node(
            mlp=_Node(down_proj=_WeightHolder(weight=ops.ones((2, 3, 4)))),
        )
        base = _Node(layers=[base_layer])
        aligned = _Node(layers=[aligned_layer])

        with patch(
            "vauban.measure._diff.get_transformer",
            side_effect=lambda model: model,
        ):
            result = measure_diff(
                cast("CausalLM", base),
                cast("CausalLM", aligned),
                top_k=5,
            )

        assert result.d_model == 6
        assert result.basis.shape == (5, 6)
        assert len(result.singular_values) == 5
        assert result.singular_values[-1] == 0.0

    def test_zero_norm_vectors_are_preserved_without_normalization(self) -> None:
        weight = ops.eye(2)
        base_layer = _Node(
            self_attn=_Node(o_proj=_WeightHolder(weight=weight)),
        )
        aligned_layer = _Node(
            self_attn=_Node(o_proj=_WeightHolder(weight=weight + 1.0)),
        )
        base = _Node(layers=[base_layer])
        aligned = _Node(layers=[aligned_layer])

        fake_u = ops.zeros((2, 2))
        fake_s = ops.array([2.0, 1.0])
        fake_vt = ops.zeros((2, 2))

        with (
            patch(
                "vauban.measure._diff.get_transformer",
                side_effect=lambda model: model,
            ),
            patch(
                "vauban.measure._diff.svd_stable",
                return_value=(fake_u, fake_s, fake_vt),
            ),
        ):
            result = measure_diff(
                cast("CausalLM", base),
                cast("CausalLM", aligned),
                top_k=2,
            )

        assert result.basis.shape == (2, 2)
        assert result.basis.tolist() == [[0.0, 0.0], [0.0, 0.0]]
