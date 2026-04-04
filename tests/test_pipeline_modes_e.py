# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode runners E: guard."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import (
    make_direction_result,
    make_early_mode_context,
    make_mock_transformer,
)
from vauban._pipeline._mode_guard import _run_guard_mode
from vauban.types import (
    GuardConfig,
    GuardEvent,
    GuardResult,
    GuardTierSpec,
)

if TYPE_CHECKING:
    from pathlib import Path


class _HasShape(Protocol):
    """Protocol for objects exposing a tuple-like ``shape``."""

    shape: tuple[int, ...]


class _PromptTokenizer:
    """Tokenizer stub for defensive-prompt embedding tests."""

    def encode(self, text: str) -> list[int]:
        """Return a deterministic token sequence."""
        return [len(text), 7, 3]


def _make_guard_result(
    prompt: str,
    *,
    total_rewinds: int = 0,
    circuit_broken: bool = False,
) -> GuardResult:
    """Build a minimal GuardResult for pipeline runner tests."""
    return GuardResult(
        prompt=prompt,
        text="guarded output",
        events=[
            GuardEvent(
                token_index=0,
                token_id=1,
                token_str="A",
                projection=0.25,
                zone="yellow",
                action="steer",
                alpha_applied=0.5,
                rewind_count=total_rewinds,
                checkpoint_offset=0,
            ),
        ],
        total_rewinds=total_rewinds,
        circuit_broken=circuit_broken,
        tokens_generated=1,
        tokens_rewound=0,
        final_zone_counts={"green": 0, "yellow": 1, "orange": 0, "red": 0},
    )


def _shape_tuple(value: _HasShape) -> tuple[int, ...]:
    """Return an object's ``shape`` attribute as a tuple of ints."""
    return cast("tuple[int, ...]", tuple(value.shape))


class TestGuardMode:
    """Tests for ``_run_guard_mode``."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="guard config is required"):
            _run_guard_mode(ctx)

    def test_missing_direction_raises(self, tmp_path: Path) -> None:
        guard_cfg = GuardConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, guard=guard_cfg)

        with (
            pytest.raises(ValueError, match="direction_result is required"),
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=2),
            ),
        ):
            _run_guard_mode(ctx)

    def test_condition_direction_mismatch_raises(
        self,
        tmp_path: Path,
    ) -> None:
        guard_cfg = GuardConfig(
            prompts=["test"],
            condition_direction_path="condition.npy",
        )
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=make_direction_result(),
            guard=guard_cfg,
        )

        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=2, d_model=16),
            ),
            patch("numpy.load", return_value=np.zeros((8,), dtype=float)) as mock_load,
            pytest.raises(ValueError, match="d_model mismatch"),
        ):
            _run_guard_mode(ctx)

        assert mock_load.call_args == ((str(tmp_path.parent / "condition.npy"),),)

    def test_happy_path_with_loaded_arrays_and_harmless_calibration(
        self,
        tmp_path: Path,
    ) -> None:
        guard_cfg = GuardConfig(
            prompts=["p1", "p2"],
            condition_direction_path="condition.npy",
            defensive_embeddings_path="defensive.npy",
            calibrate=True,
        )
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=make_direction_result(),
            harmless=["safe1", "safe2"],
            guard=guard_cfg,
        )
        transformer = make_mock_transformer(n_layers=3, d_model=16)
        calibrated_tiers = [
            GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
            GuardTierSpec(threshold=0.2, zone="yellow", alpha=0.4),
            GuardTierSpec(threshold=0.5, zone="orange", alpha=1.2),
            GuardTierSpec(threshold=0.8, zone="red", alpha=2.5),
        ]
        seen_condition_shapes: list[tuple[int, ...]] = []
        seen_defensive_shapes: list[tuple[int, ...]] = []
        seen_guard_cfg_ids: list[int] = []

        def _fake_guard_generate(
            model: object,
            tokenizer: object,
            prompt: str,
            direction: object,
            guard_layers: list[int],
            guard_cfg_for_run: GuardConfig,
            *,
            condition_direction: object | None = None,
            defensive_embeds: object | None = None,
        ) -> GuardResult:
            del model, tokenizer, direction
            assert guard_layers == [0, 1, 2]
            assert condition_direction is not None
            assert defensive_embeds is not None
            seen_condition_shapes.append(
                _shape_tuple(cast("_HasShape", condition_direction)),
            )
            seen_defensive_shapes.append(
                _shape_tuple(cast("_HasShape", defensive_embeds)),
            )
            seen_guard_cfg_ids.append(id(guard_cfg_for_run))
            assert guard_cfg_for_run.tiers == calibrated_tiers
            return _make_guard_result(prompt, total_rewinds=1, circuit_broken=True)

        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=transformer,
            ),
            patch(
                "vauban.guard.calibrate_guard_thresholds",
                return_value=calibrated_tiers,
            ) as mock_calibrate,
            patch(
                "vauban.guard.guard_generate",
                side_effect=_fake_guard_generate,
            ),
            patch("numpy.load") as mock_load,
            patch("numpy.save") as mock_save,
            patch("vauban._pipeline._mode_guard.finish_mode_run") as mock_finish,
        ):
            mock_load.side_effect = [
                np.zeros((16,), dtype=float),
                np.ones((2, 16), dtype=float),
            ]
            _run_guard_mode(ctx)

        assert (tmp_path / "guard_report.json").exists()
        assert json.loads((tmp_path / "guard_tiers.json").read_text()) == [
            {"threshold": 0.0, "zone": "green", "alpha": 0.0},
            {"threshold": 0.2, "zone": "yellow", "alpha": 0.4},
            {"threshold": 0.5, "zone": "orange", "alpha": 1.2},
            {"threshold": 0.8, "zone": "red", "alpha": 2.5},
        ]
        assert mock_calibrate.call_args[0][2] == ["safe1", "safe2"]
        assert seen_condition_shapes == [(16,), (16,)]
        assert seen_defensive_shapes == [(1, 2, 16), (1, 2, 16)]
        assert seen_guard_cfg_ids == [seen_guard_cfg_ids[0], seen_guard_cfg_ids[0]]
        assert seen_guard_cfg_ids[0] != id(guard_cfg)
        assert mock_save.call_args is not None
        assert mock_save.call_args[0][0] == str(tmp_path / "guard_direction.npy")
        metadata = mock_finish.call_args[0][3]
        assert metadata["n_prompts"] == 2
        assert metadata["total_rewinds"] == 2
        assert metadata["circuit_broken"] == 2

    def test_happy_path_with_prompt_and_harmful_calibration(
        self,
        tmp_path: Path,
    ) -> None:
        guard_cfg = GuardConfig(
            prompts=["prompt"],
            defensive_prompt="stay aligned",
            calibrate=True,
            calibrate_prompts="harmful",
        )
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=make_direction_result(),
            harmful=["harm1", "harm2", "harm3"],
            guard=guard_cfg,
        )
        ctx.tokenizer = _PromptTokenizer()
        transformer = make_mock_transformer(n_layers=2, d_model=16)
        defensive_embedding_array = np.ones((1, 3, 16), dtype=float)
        transformer.embed_tokens.return_value = defensive_embedding_array
        calibrated_tiers = [
            GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
            GuardTierSpec(threshold=0.7, zone="red", alpha=3.0),
        ]

        def _fake_guard_generate(
            model: object,
            tokenizer: object,
            prompt: str,
            direction: object,
            guard_layers: list[int],
            guard_cfg_for_run: GuardConfig,
            *,
            condition_direction: object | None = None,
            defensive_embeds: object | None = None,
        ) -> GuardResult:
            del model, tokenizer, direction, guard_layers, condition_direction
            assert prompt == "prompt"
            assert guard_cfg_for_run.tiers == calibrated_tiers
            assert defensive_embeds is defensive_embedding_array
            return _make_guard_result(prompt, total_rewinds=2, circuit_broken=False)

        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=transformer,
            ),
            patch(
                "vauban.guard.calibrate_guard_thresholds",
                return_value=calibrated_tiers,
            ) as mock_calibrate,
            patch(
                "vauban.guard.guard_generate",
                side_effect=_fake_guard_generate,
            ),
            patch("numpy.save"),
            patch("vauban._pipeline._mode_guard.finish_mode_run") as mock_finish,
        ):
            _run_guard_mode(ctx)

        assert mock_calibrate.call_args[0][2] == ["harm1", "harm2", "harm3"]
        metadata = mock_finish.call_args[0][3]
        assert metadata["n_prompts"] == 1
        assert metadata["total_rewinds"] == 2
        assert metadata["circuit_broken"] == 0

    def test_happy_path_reuses_original_config_without_calibration(
        self,
        tmp_path: Path,
    ) -> None:
        guard_cfg = GuardConfig(prompts=["one"])
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=make_direction_result(),
            guard=guard_cfg,
        )

        def _fake_guard_generate(
            model: object,
            tokenizer: object,
            prompt: str,
            direction: object,
            guard_layers: list[int],
            guard_cfg_for_run: GuardConfig,
            *,
            condition_direction: object | None = None,
            defensive_embeds: object | None = None,
        ) -> GuardResult:
            del model, tokenizer, prompt, direction
            del condition_direction, defensive_embeds
            assert guard_layers == [0, 1]
            assert guard_cfg_for_run is guard_cfg
            return _make_guard_result("one")

        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=2, d_model=16),
            ),
            patch(
                "vauban.guard.guard_generate",
                side_effect=_fake_guard_generate,
            ),
            patch("numpy.save"),
            patch("vauban._pipeline._mode_guard.finish_mode_run"),
        ):
            _run_guard_mode(ctx)
