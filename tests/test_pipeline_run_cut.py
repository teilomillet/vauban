# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline cut phase."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import (
    make_direction_result,
    make_mock_transformer,
    make_pipeline_config,
)
from vauban import _ops as ops
from vauban._pipeline._run_cut import run_cut_phase
from vauban._pipeline._run_state import RunState
from vauban.types import (
    CutConfig,
    DBDIResult,
    DirectionResult,
    EvalConfig,
    MeasureConfig,
    SubspaceResult,
    SurfaceResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


class _ModelStub:
    """Minimal model stub for cut-phase orchestration tests."""

    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] | None = None

    def parameters(self) -> dict[str, object]:
        """Return a small weight tree placeholder."""
        return {"weights": _make_array(1.0)}

    def load_weights(self, weights: list[tuple[str, object]]) -> None:
        """Capture loaded weights for assertions."""
        self.loaded_weights = weights


def _make_array(value: float) -> Array:
    """Build a tiny backend array."""
    return ops.array([value])


def _make_weights(value: float) -> dict[str, Array]:
    """Build a minimal flat weight mapping."""
    return {
        "model.layers.0.self_attn.o_proj.weight": _make_array(value),
    }


def _make_direction_with_types(layer_types: list[str] | None) -> DirectionResult:
    """Build a direction result with optional layer-type metadata."""
    direction = make_direction_result()
    return DirectionResult(
        direction=direction.direction,
        layer_index=direction.layer_index,
        cosine_scores=direction.cosine_scores,
        d_model=direction.d_model,
        model_path=direction.model_path,
        layer_types=layer_types,
    )


def _make_subspace_result(layer_types: list[str] | None = None) -> SubspaceResult:
    """Build a minimal subspace result."""
    direction = make_direction_result()
    basis = direction.direction[None, :]
    return SubspaceResult(
        basis=basis,
        singular_values=[1.0],
        explained_variance=[1.0],
        layer_index=1,
        d_model=16,
        model_path="test-model",
        per_layer_bases=[basis],
        layer_types=layer_types,
    )


def _make_dbdi_result() -> DBDIResult:
    """Build a minimal DBDI result."""
    return DBDIResult(
        hdd=_make_array(2.0),
        red=_make_array(3.0),
        hdd_layer_index=0,
        red_layer_index=1,
        hdd_cosine_scores=[0.3],
        red_cosine_scores=[0.4],
        d_model=16,
        model_path="test-model",
    )


def _make_state(
    tmp_path: Path,
    *,
    cut: CutConfig | None = None,
    measure: MeasureConfig | None = None,
    eval_config: EvalConfig | None = None,
    direction_result: DirectionResult | None = None,
    subspace_result: SubspaceResult | None = None,
    dbdi_result: DBDIResult | None = None,
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
    cosine_scores: list[float] | None = None,
    borderline_path: object | None = None,
) -> RunState:
    """Build a ``RunState`` tailored for cut-phase tests."""
    config = make_pipeline_config(
        tmp_path,
        cut=cut if cut is not None else CutConfig(),
        measure=measure if measure is not None else MeasureConfig(),
        eval=eval_config if eval_config is not None else EvalConfig(),
        borderline_path=borderline_path,
    )
    return RunState(
        config_path="test.toml",
        config=config,
        model=cast("CausalLM", _ModelStub()),
        tokenizer=cast("Tokenizer", object()),
        t0=0.0,
        verbose=False,
        harmful=harmful,
        harmless=harmless,
        direction_result=direction_result,
        subspace_result=subspace_result,
        dbdi_result=dbdi_result,
        cosine_scores=list(cosine_scores or []),
    )


class TestRunCutPhase:
    """Tests for ``run_cut_phase``."""

    def test_strategy_requires_cosine_scores(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            cut=CutConfig(layer_strategy="top_k"),
            direction_result=make_direction_result(),
        )

        with pytest.raises(
            ValueError,
            match="Probe-guided layer selection requires 'direction' mode",
        ):
            run_cut_phase(state)

    def test_explicit_layers_sparsify_false_refusal_and_standard_cut(
        self,
        tmp_path: Path,
    ) -> None:
        flat_weights = _make_weights(1.0)
        modified_weights = _make_weights(4.0)
        sparse_direction = _make_array(8.0)
        orthogonal_direction = _make_array(9.0)
        false_refusal_result = make_direction_result()
        state = _make_state(
            tmp_path,
            cut=CutConfig(
                layers=[1],
                alpha=1.5,
                sparsity=0.25,
                false_refusal_ortho=True,
            ),
            direction_result=_make_direction_with_types(["global", "sliding"]),
            harmful=["harm"],
            harmless=["safe"],
            borderline_path=tmp_path / "border.jsonl",
        )

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(flat_weights.items()),
            ),
            patch(
                "vauban.cut.sparsify_direction",
                return_value=sparse_direction,
            ),
            patch(
                "vauban.dataset.resolve_prompts",
                return_value=["borderline"],
            ),
            patch(
                "vauban.measure.measure",
                return_value=false_refusal_result,
            ),
            patch(
                "vauban.cut._biprojected_direction",
                return_value=orthogonal_direction,
            ),
            patch(
                "vauban.cut.cut",
                return_value=modified_weights,
            ) as mock_cut,
            patch("vauban.export.export_model") as mock_export,
        ):
            run_cut_phase(state)

        assert state.target_layers == [1]
        assert state.flat_weights == flat_weights
        assert state.modified_weights == modified_weights
        assert state.direction_result is not None
        assert state.direction_result.direction is orthogonal_direction
        assert mock_cut.call_args[0][1] is orthogonal_direction
        assert mock_export.call_args[0][0] == "test-model"
        assert mock_export.call_args[0][1] == modified_weights

    def test_strategy_selects_layers_and_hydrates_modified_model(
        self,
        tmp_path: Path,
    ) -> None:
        flat_weights = _make_weights(1.0)
        modified_weights = _make_weights(5.0)
        loaded_model = _ModelStub()
        direction_result = _make_direction_with_types(["global", "sliding"])
        state = _make_state(
            tmp_path,
            cut=CutConfig(
                layer_strategy="top_k",
                layer_top_k=1,
                layer_type_filter="sliding",
            ),
            eval_config=EvalConfig(prompts_path=tmp_path / "eval.jsonl"),
            direction_result=direction_result,
            cosine_scores=[0.1, 0.9],
        )

        with (
            patch(
                "vauban.measure.select_target_layers",
                return_value=[1],
            ) as mock_select,
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(flat_weights.items()),
            ),
            patch("vauban.cut.cut", return_value=modified_weights),
            patch("vauban.export.export_model"),
            patch(
                "vauban._model_io.load_model",
                return_value=(
                    cast("CausalLM", loaded_model),
                    cast("Tokenizer", object()),
                ),
            ),
            patch("vauban.dequantize.is_quantized", return_value=False),
        ):
            run_cut_phase(state)

        assert mock_select.call_args is not None
        assert mock_select.call_args.kwargs["layer_types"] == ["global", "sliding"]
        assert mock_select.call_args.kwargs["type_filter"] == "sliding"
        assert state.target_layers == [1]
        assert state.modified_model is loaded_model
        assert loaded_model.loaded_weights == list(modified_weights.items())

    def test_subspace_mode_requires_result(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            measure=MeasureConfig(mode="subspace"),
        )

        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=2),
            ),
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(_make_weights(1.0).items()),
            ),
            pytest.raises(
                ValueError,
                match="subspace_result is required for subspace cut",
            ),
        ):
            run_cut_phase(state)

    def test_subspace_mode_uses_basis_and_layer_types(
        self,
        tmp_path: Path,
    ) -> None:
        flat_weights = _make_weights(1.0)
        modified_weights = _make_weights(6.0)
        subspace_result = _make_subspace_result(["global", "sliding"])
        state = _make_state(
            tmp_path,
            cut=CutConfig(layers=[0], alpha=0.8, norm_preserve=True),
            measure=MeasureConfig(mode="subspace"),
            subspace_result=subspace_result,
        )

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(flat_weights.items()),
            ),
            patch(
                "vauban.cut.cut_subspace",
                return_value=modified_weights,
            ) as mock_cut_subspace,
            patch("vauban.export.export_model"),
        ):
            run_cut_phase(state)

        assert mock_cut_subspace.call_args[0][1] is subspace_result.basis
        assert state.modified_weights == modified_weights

    def test_biprojected_mode_requires_direction_and_prompts(
        self,
        tmp_path: Path,
    ) -> None:
        state = _make_state(
            tmp_path,
            cut=CutConfig(layers=[0], biprojected=True),
            direction_result=make_direction_result(),
            harmful=None,
            harmless=["safe"],
        )

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(_make_weights(1.0).items()),
            ),
            pytest.raises(
                ValueError,
                match="direction_result, harmful, and harmless are required",
            ),
        ):
            run_cut_phase(state)

    def test_biprojected_mode_measures_and_dequantizes_loaded_model(
        self,
        tmp_path: Path,
    ) -> None:
        flat_weights = _make_weights(1.0)
        modified_weights = _make_weights(7.0)
        loaded_model = _ModelStub()
        direction_result = make_direction_result()
        harmless_acts = make_direction_result()
        state = _make_state(
            tmp_path,
            cut=CutConfig(layers=[0], biprojected=True),
            eval_config=EvalConfig(prompts_path=tmp_path / "eval.jsonl"),
            direction_result=direction_result,
            harmful=["harm"],
            harmless=["safe"],
        )

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(flat_weights.items()),
            ),
            patch(
                "vauban.measure.measure",
                return_value=harmless_acts,
            ) as mock_measure,
            patch(
                "vauban.cut.cut_biprojected",
                return_value=modified_weights,
            ) as mock_cut_bi,
            patch("vauban.export.export_model"),
            patch(
                "vauban._model_io.load_model",
                return_value=(
                    cast("CausalLM", loaded_model),
                    cast("Tokenizer", object()),
                ),
            ),
            patch("vauban.dequantize.is_quantized", return_value=True),
            patch("vauban.dequantize.dequantize_model") as mock_dequantize,
        ):
            run_cut_phase(state)

        assert mock_measure.call_args[0][2] == ["safe"]
        assert mock_measure.call_args[0][3] == ["harm"]
        assert mock_cut_bi.call_args[0][1] is direction_result.direction
        assert mock_cut_bi.call_args[0][2] is harmless_acts.direction
        assert loaded_model.loaded_weights == list(modified_weights.items())
        mock_dequantize.assert_called_once_with(loaded_model)

    def test_standard_cut_requires_direction(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            cut=CutConfig(layers=[0]),
        )

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(_make_weights(1.0).items()),
            ),
            pytest.raises(ValueError, match="direction_result is required for cut"),
        ):
            run_cut_phase(state)

    def test_dbdi_both_requires_dbdi_result(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            cut=CutConfig(layers=[0], dbdi_target="both"),
            measure=MeasureConfig(mode="dbdi"),
            direction_result=make_direction_result(),
        )

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(_make_weights(1.0).items()),
            ),
            patch("vauban.cut.cut", return_value=_make_weights(8.0)),
            pytest.raises(
                ValueError,
                match="dbdi_result is required for DBDI both-mode cut",
            ),
        ):
            run_cut_phase(state)

    def test_dbdi_both_applies_second_cut_and_hydrates_from_surface(
        self,
        tmp_path: Path,
    ) -> None:
        first_cut = _make_weights(8.0)
        second_cut = _make_weights(9.0)
        loaded_model = _ModelStub()
        state = _make_state(
            tmp_path,
            cut=CutConfig(layers=[0], dbdi_target="both"),
            measure=MeasureConfig(mode="dbdi"),
            direction_result=make_direction_result(),
            dbdi_result=_make_dbdi_result(),
        )
        state.surface_before = cast("SurfaceResult", object())

        with (
            patch(
                "vauban._ops.tree_flatten",
                return_value=list(_make_weights(1.0).items()),
            ),
            patch(
                "vauban.cut.cut",
                side_effect=[first_cut, second_cut],
            ) as mock_cut,
            patch("vauban.export.export_model"),
            patch(
                "vauban._model_io.load_model",
                return_value=(
                    cast("CausalLM", loaded_model),
                    cast("Tokenizer", object()),
                ),
            ),
            patch("vauban.dequantize.is_quantized", return_value=False),
        ):
            run_cut_phase(state)

        assert mock_cut.call_count == 2
        assert state.dbdi_result is not None
        assert mock_cut.call_args_list[1][0][1] is state.dbdi_result.hdd
        assert state.modified_weights == second_cut
        assert loaded_model.loaded_weights == list(second_cut.items())
