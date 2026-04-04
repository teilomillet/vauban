# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for CAST entry points, forward pass steering, and SVF-aware CAST."""

from typing import cast

import pytest

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban._array import Array
from vauban.cast import (
    _cast_forward,
    _cast_forward_svf,
    cast_generate,
    cast_generate_svf,
    cast_generate_with_messages,
)
from vauban.svf import SVFBoundary
from vauban.types import AlphaTier, LayerCache


def _make_layer_cache(model: MockCausalLM) -> list[LayerCache]:
    """Cast the mock model cache to the protocol used by CAST helpers."""
    return cast("list[LayerCache]", model.make_cache())


class TestCastGenerateWithMessages:
    """Tests for the multi-turn message-based entry point."""

    def test_single_user_message(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        messages = [{"role": "user", "content": "test prompt"}]
        result = cast_generate_with_messages(
            mock_model, mock_tokenizer, messages,
            direction, layers=[0], max_tokens=3,
        )
        assert result.prompt == "test prompt"
        assert len(result.text) > 0
        assert result.considered > 0

    def test_multi_turn_uses_last_user_message(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ]
        result = cast_generate_with_messages(
            mock_model, mock_tokenizer, messages,
            direction, layers=[0], max_tokens=2,
        )
        assert result.prompt == "follow-up"

    def test_no_user_message_empty_prompt(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        messages = [{"role": "assistant", "content": "hello"}]
        result = cast_generate_with_messages(
            mock_model, mock_tokenizer, messages,
            direction, layers=[0], max_tokens=2,
        )
        assert result.prompt == ""


class TestCastForward:
    """Tests for the _cast_forward steering mechanics."""

    def test_no_intervention_at_high_threshold(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = _make_layer_cache(mock_model)

        logits, proj_before, _proj_after, interventions, considered, _, _ = (
            _cast_forward(
                mock_model, token_ids, direction,
                cast_layers=[0], alpha=1.0, threshold=1e9,
                cache=cache,
            )
        )
        ops.eval(logits)
        assert interventions == 0
        assert considered == 1
        assert len(proj_before) == 1

    def test_intervention_at_low_threshold(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = _make_layer_cache(mock_model)

        logits, proj_before, proj_after, interventions, considered, _, _ = (
            _cast_forward(
                mock_model, token_ids, direction,
                cast_layers=[0, 1], alpha=1.0, threshold=-1e9,
                cache=cache,
            )
        )
        ops.eval(logits)
        assert interventions == 2
        assert considered == 2
        assert len(proj_before) == 2
        assert len(proj_after) == 2

    def test_non_cast_layers_skipped(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = _make_layer_cache(mock_model)

        _, _, _, _, considered, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[], alpha=1.0, threshold=0.0,
            cache=cache,
        )
        assert considered == 0

    def test_dual_direction_gating(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        """Condition direction gates, primary direction steers."""
        token_ids = ops.array([[1, 2]])
        cache = _make_layer_cache(mock_model)

        # Zero condition = never triggers
        zero_cond = ops.zeros((D_MODEL,))
        ops.eval(zero_cond)

        _, _, _, interventions, _, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=0.01,
            cache=cache,
            condition_direction=zero_cond,
        )
        assert interventions == 0

    def test_alpha_tiers_applied(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2]])
        cache = _make_layer_cache(mock_model)
        tiers = [
            AlphaTier(threshold=0.0, alpha=0.1),
            AlphaTier(threshold=100.0, alpha=99.0),
        ]

        logits, _, _, interventions, _, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=-1e9,
            cache=cache,
            alpha_tiers=tiers,
        )
        ops.eval(logits)
        # Should have intervened with the first tier alpha (0.1)
        assert interventions == 1

    def test_displacement_monitoring(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = _make_layer_cache(mock_model)

        # Create baseline activations (zeros → any activation will displace)
        baseline = {0: ops.zeros((D_MODEL,))}
        ops.eval(baseline[0])

        _, _, _, _, _, disp_interventions, max_disp = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=1e9,  # high threshold
            cache=cache,
            baseline_activations=baseline,
            displacement_threshold=0.001,  # very low → will trigger
        )
        # Displacement should trigger: baseline is zeros, any activation displaces
        assert disp_interventions > 0
        assert max_disp > 0.0

    def test_logits_shape(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = _make_layer_cache(mock_model)

        logits, _, _, _, _, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=0.0,
            cache=cache,
        )
        ops.eval(logits)
        assert logits.ndim == 3  # (batch, seq, vocab)
        assert logits.shape[0] == 1


class TestCastExternalities:
    """Tests for externality monitoring integration."""

    def test_externality_off_by_default(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        result = cast_generate(
            mock_model, mock_tokenizer, "test",
            direction, layers=[0], max_tokens=2,
        )
        assert result.displacement_interventions == 0
        assert result.max_displacement == pytest.approx(0.0)

    def test_externality_with_baseline(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        baseline = {0: ops.zeros((D_MODEL,))}
        ops.eval(baseline[0])

        result = cast_generate(
            mock_model, mock_tokenizer, "test",
            direction, layers=[0], max_tokens=2,
            baseline_activations=baseline,
            displacement_threshold=0.001,
        )
        # max_displacement should be tracked even if no disp interventions
        assert result.max_displacement >= 0.0


class TestCastSVF:
    """Tests for SVF-aware CAST generation and forward steps."""

    def test_cast_generate_svf_aggregates_scores(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import vauban.cast as cast_module

        logits_step_1 = ops.array([[[0.1, 0.9, 0.2]]])
        logits_step_2 = ops.array([[[0.1, 0.2, 0.9]]])
        steps = iter([
            (logits_step_1, [1.0], [0.5], 1, 1),
            (logits_step_2, [2.0], [1.5], 0, 1),
        ])

        def _fake_encode(
            tokenizer: MockTokenizer,
            messages: list[dict[str, str]],
        ) -> Array:
            del tokenizer, messages
            return ops.array([[1, 2]])

        def _fake_forward(
            model: MockCausalLM,
            token_ids: Array,
            boundary: SVFBoundary,
            layers: list[int],
            alpha: float,
            cache: object,
        ) -> tuple[Array, list[float], list[float], int, int]:
            del model, token_ids, boundary, layers, alpha, cache
            return next(steps)

        monkeypatch.setattr(cast_module, "encode_chat_prompt", _fake_encode)
        monkeypatch.setattr(cast_module, "_cast_forward_svf", _fake_forward)

        boundary = SVFBoundary(
            d_model=D_MODEL,
            projection_dim=4,
            hidden_dim=4,
            n_layers=len(mock_model.model.layers),
        )
        result = cast_generate_svf(
            mock_model,
            mock_tokenizer,
            "prompt",
            boundary,
            layers=[0],
            max_tokens=2,
        )

        assert result.prompt == "prompt"
        assert result.text == mock_tokenizer.decode([1, 2])
        assert result.projections_before == [1.0, 2.0]
        assert result.projections_after == [0.5, 1.5]
        assert result.interventions == 1
        assert result.considered == 2

    def test_cast_forward_svf_no_positive_scores(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        token_ids = ops.array([[1, 2]])
        cache = _make_layer_cache(mock_model)
        boundary = SVFBoundary(
            d_model=D_MODEL,
            projection_dim=4,
            hidden_dim=4,
            n_layers=len(mock_model.model.layers),
        )
        grad = ops.zeros((D_MODEL,))
        ops.eval(grad)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "vauban.svf.svf_gradient",
                lambda boundary_obj, last_token, layer_idx: (-0.5, grad),
            )
            logits, before, after, interventions, considered = _cast_forward_svf(
                mock_model,
                token_ids,
                boundary,
                cast_layers=[0],
                alpha=1.0,
                cache=cache,
            )

        ops.eval(logits)
        assert interventions == 0
        assert considered == 1
        assert before == [-0.5]
        assert after == [-0.5]

    def test_cast_forward_svf_positive_score_intervenes(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        token_ids = ops.array([[1, 2]])
        cache = _make_layer_cache(mock_model)
        boundary = SVFBoundary(
            d_model=D_MODEL,
            projection_dim=4,
            hidden_dim=4,
            n_layers=len(mock_model.model.layers),
        )
        grad = ops.ones((D_MODEL,))
        ops.eval(grad)
        scores = iter([(1.0, grad), (-0.25, grad)])

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "vauban.svf.svf_gradient",
                lambda boundary_obj, last_token, layer_idx: next(scores),
            )
            logits, before, after, interventions, considered = _cast_forward_svf(
                mock_model,
                token_ids,
                boundary,
                cast_layers=[0],
                alpha=1.0,
                cache=cache,
            )

        ops.eval(logits)
        assert interventions == 1
        assert considered == 1
        assert before == [1.0]
        assert after == [-0.25]
