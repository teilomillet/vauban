# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline surface phases."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import (
    make_direction_result,
    make_pipeline_config,
)
from vauban._pipeline._run_state import RunState
from vauban._pipeline._run_surface import (
    finalize_surface_phase,
    prepare_surface_phase,
)
from vauban.taxonomy import TaxonomyCoverage
from vauban.types import (
    DirectionResult,
    SubspaceResult,
    SurfaceComparison,
    SurfaceConfig,
    SurfaceGroup,
    SurfaceGroupDelta,
    SurfacePoint,
    SurfacePrompt,
    SurfaceResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.types import CausalLM, Tokenizer


def _make_run_state(
    tmp_path: Path,
    *,
    direction_result: DirectionResult | None = None,
    subspace_result: SubspaceResult | None = None,
    surface: SurfaceConfig | None = None,
) -> RunState:
    """Build a minimal ``RunState`` for surface-phase tests."""
    config = make_pipeline_config(tmp_path, surface=surface)
    return RunState(
        config_path="test.toml",
        config=config,
        model=cast("CausalLM", object()),
        tokenizer=cast("Tokenizer", object()),
        t0=0.0,
        verbose=False,
        direction_result=direction_result,
        subspace_result=subspace_result,
    )


def _make_surface_prompt(prompt: str = "prompt") -> SurfacePrompt:
    """Build a minimal surface prompt."""
    return SurfacePrompt(prompt=prompt, label="harmful", category="weapons")


def _make_surface_result(
    *,
    taxonomy_coverage: TaxonomyCoverage | None = None,
) -> SurfaceResult:
    """Build a minimal ``SurfaceResult``."""
    point = SurfacePoint(
        prompt="prompt",
        label="harmful",
        category="weapons",
        projections=[0.2],
        direction_projection=0.2,
        refused=False,
        response="ok",
    )
    group = SurfaceGroup(
        name="harmful",
        count=1,
        refusal_rate=0.0,
        mean_projection=0.2,
        min_projection=0.2,
        max_projection=0.2,
    )
    return SurfaceResult(
        points=[point],
        groups_by_label=[group],
        groups_by_category=[group],
        threshold=0.1,
        total_scanned=1,
        total_refused=0,
        coverage_score=0.5,
        taxonomy_coverage=taxonomy_coverage,
    )


def _make_surface_comparison() -> SurfaceComparison:
    """Build a minimal ``SurfaceComparison``."""
    before = _make_surface_result()
    after = _make_surface_result()
    delta = SurfaceGroupDelta(
        name="harmful",
        count=1,
        refusal_rate_before=0.5,
        refusal_rate_after=0.1,
        refusal_rate_delta=-0.4,
        mean_projection_before=0.3,
        mean_projection_after=0.2,
        mean_projection_delta=-0.1,
    )
    return SurfaceComparison(
        before=before,
        after=after,
        refusal_rate_before=0.5,
        refusal_rate_after=0.1,
        refusal_rate_delta=-0.4,
        threshold_before=0.2,
        threshold_after=0.1,
        threshold_delta=-0.1,
        category_deltas=[delta],
        label_deltas=[delta],
        coverage_score_before=0.5,
        coverage_score_after=0.6,
        coverage_score_delta=0.1,
        worst_cell_refusal_rate_before=0.4,
        worst_cell_refusal_rate_after=0.2,
        worst_cell_refusal_rate_delta=-0.2,
    )


def _make_subspace_result() -> SubspaceResult:
    """Build a minimal subspace result with one basis vector."""
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
    )


class TestPrepareSurfacePhase:
    """Tests for ``prepare_surface_phase``."""

    def test_returns_without_surface_config(self, tmp_path: Path) -> None:
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=None,
        )

        prepare_surface_phase(state)

        assert state.surface_prompts is None
        assert state.surface_before is None

    def test_returns_without_direction(self, tmp_path: Path) -> None:
        state = _make_run_state(
            tmp_path,
            surface=SurfaceConfig(prompts_path="default"),
        )

        prepare_surface_phase(state)

        assert state.surface_prompts is None
        assert state.surface_before is None

    @pytest.mark.parametrize(
        ("prompts_path", "resolved_path"),
        [
            ("default", "default.jsonl"),
            ("default_multilingual", "multilingual.jsonl"),
            ("default_full", "full.jsonl"),
            ("/tmp/custom.jsonl", "/tmp/custom.jsonl"),
        ],
    )
    def test_resolves_prompt_source_from_config(
        self,
        tmp_path: Path,
        prompts_path: str,
        resolved_path: str,
    ) -> None:
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=SurfaceConfig(prompts_path=prompts_path, generate=False),
        )

        with (
            patch(
                "vauban.surface.default_surface_path",
                return_value="default.jsonl",
            ),
            patch(
                "vauban.surface.default_multilingual_surface_path",
                return_value="multilingual.jsonl",
            ),
            patch(
                "vauban.surface.default_full_surface_path",
                return_value="full.jsonl",
            ),
            patch(
                "vauban.surface.load_surface_prompts",
                return_value=[_make_surface_prompt()],
            ) as mock_load,
            patch(
                "vauban.surface.map_surface",
                return_value=_make_surface_result(),
            ) as mock_map,
        ):
            prepare_surface_phase(state)

        assert mock_load.call_args == ((resolved_path,),)
        assert mock_map.call_args is not None
        assert state.surface_prompts == [_make_surface_prompt()]
        assert state.surface_before is not None

    def test_uses_subspace_result_and_logs_long_missing_list(
        self,
        tmp_path: Path,
    ) -> None:
        missing = frozenset({"a", "b", "c", "d", "e", "f"})
        coverage = TaxonomyCoverage(
            present=frozenset({"weapons"}),
            missing=missing,
            aliased={},
            coverage_ratio=1 / 7,
        )
        state = _make_run_state(
            tmp_path,
            subspace_result=_make_subspace_result(),
            surface=SurfaceConfig(prompts_path="default"),
        )

        with (
            patch(
                "vauban.surface.default_surface_path",
                return_value="default.jsonl",
            ),
            patch(
                "vauban.surface.load_surface_prompts",
                return_value=[_make_surface_prompt()],
            ),
            patch(
                "vauban.surface.map_surface",
                return_value=_make_surface_result(taxonomy_coverage=coverage),
            ),
            patch("vauban._pipeline._run_surface.log") as mock_log,
        ):
            prepare_surface_phase(state)

        log_messages = [call.args[0] for call in mock_log.call_args_list]
        assert any("(+1 more)" in msg for msg in log_messages)
        assert state.surface_layer == 1

    def test_logs_short_missing_category_list(self, tmp_path: Path) -> None:
        coverage = TaxonomyCoverage(
            present=frozenset({"weapons"}),
            missing=frozenset({"fraud", "malware"}),
            aliased={},
            coverage_ratio=1 / 3,
        )
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=SurfaceConfig(prompts_path="default"),
        )

        with (
            patch(
                "vauban.surface.default_surface_path",
                return_value="default.jsonl",
            ),
            patch(
                "vauban.surface.load_surface_prompts",
                return_value=[_make_surface_prompt()],
            ),
            patch(
                "vauban.surface.map_surface",
                return_value=_make_surface_result(taxonomy_coverage=coverage),
            ),
            patch("vauban._pipeline._run_surface.log") as mock_log,
        ):
            prepare_surface_phase(state)

        log_messages = [call.args[0] for call in mock_log.call_args_list]
        assert any("fraud, malware" in msg for msg in log_messages)


class TestFinalizeSurfacePhase:
    """Tests for ``finalize_surface_phase``."""

    def test_returns_without_surface_before(self, tmp_path: Path) -> None:
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=SurfaceConfig(prompts_path="default"),
        )

        finalize_surface_phase(state)

        assert state.report_files == []

    @pytest.mark.parametrize(
        ("field_name", "field_value", "surface", "message"),
        [
            (
                "modified_model",
                None,
                SurfaceConfig(prompts_path="default"),
                "modified_model is required",
            ),
            (
                "surface_prompts",
                None,
                SurfaceConfig(prompts_path="default"),
                "surface_prompts is required",
            ),
            (
                "surface_direction",
                None,
                SurfaceConfig(prompts_path="default"),
                "surface_direction is required",
            ),
            ("config", "drop_surface", None, "surface config is required"),
        ],
    )
    def test_validates_required_state(
        self,
        tmp_path: Path,
        field_name: str,
        field_value: object | None,
        surface: SurfaceConfig | None,
        message: str,
    ) -> None:
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=SurfaceConfig(prompts_path="default"),
        )
        state.surface_before = _make_surface_result()
        state.modified_model = cast("CausalLM", object())
        state.surface_prompts = [_make_surface_prompt()]
        state.surface_direction = make_direction_result().direction
        if field_name == "config":
            state.config = make_pipeline_config(tmp_path, surface=surface)
        else:
            setattr(state, field_name, field_value)

        with pytest.raises(ValueError, match=message):
            finalize_surface_phase(state)

    def test_writes_report_and_appends_file(self, tmp_path: Path) -> None:
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=SurfaceConfig(prompts_path="default"),
        )
        state.surface_before = _make_surface_result()
        state.modified_model = cast("CausalLM", object())
        state.surface_prompts = [_make_surface_prompt()]
        state.surface_direction = make_direction_result().direction
        comparison = _make_surface_comparison()

        with (
            patch(
                "vauban.surface.map_surface",
                return_value=_make_surface_result(),
            ),
            patch(
                "vauban.surface.compare_surfaces",
                return_value=comparison,
            ),
            patch(
                "vauban._pipeline._run_surface.surface_gate_failures",
                return_value=[],
            ),
        ):
            finalize_surface_phase(state)

        report_path = tmp_path / "surface_report.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text())
        assert payload["summary"]["refusal_rate_after"] == 0.1
        assert state.report_files == ["surface_report.json"]

    def test_raises_when_surface_gates_fail(self, tmp_path: Path) -> None:
        state = _make_run_state(
            tmp_path,
            direction_result=make_direction_result(),
            surface=SurfaceConfig(prompts_path="default"),
        )
        state.surface_before = _make_surface_result()
        state.modified_model = cast("CausalLM", object())
        state.surface_prompts = [_make_surface_prompt()]
        state.surface_direction = make_direction_result().direction

        with (
            patch(
                "vauban.surface.map_surface",
                return_value=_make_surface_result(),
            ),
            patch(
                "vauban.surface.compare_surfaces",
                return_value=_make_surface_comparison(),
            ),
            patch(
                "vauban._pipeline._run_surface.surface_gate_failures",
                return_value=["too much regression"],
            ),
            pytest.raises(RuntimeError, match="too much regression"),
        ):
            finalize_surface_phase(state)

        assert state.report_files == []
