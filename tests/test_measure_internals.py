"""Tests for measure module internals: clip, separation, direction."""

from __future__ import annotations

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.measure._activations import (
    _clip_activation,
    _collect_activations,
    _collect_per_prompt_activations,
    _forward_collect,
)
from vauban.measure._direction import _best_direction, _cosine_separation

# ---------------------------------------------------------------------------
# _clip_activation
# ---------------------------------------------------------------------------


class TestClipActivation:
    """Tests for activation winsorization by absolute-value quantile."""

    def test_zero_quantile_is_noop(self) -> None:
        """Quantile 0.0 clips at the maximum, leaving all values unchanged."""
        x = ops.array([1.0, -2.0, 3.0, -4.0, 5.0])
        clipped = _clip_activation(x, quantile=0.0)
        assert ops.allclose(clipped, x)

    def test_clips_extreme_values(self) -> None:
        """A non-zero quantile should reduce the magnitude of outliers."""
        # 10 values: sorted abs = [0,1,2,3,4,5,6,7,8,9]
        x = ops.array([float(i) for i in range(10)])
        clipped = _clip_activation(x, quantile=0.2)
        ops.eval(clipped)
        # Threshold at index int(10 * 0.8) = 8, sorted_abs[8] = 8
        # Values > 8 should be clipped to 8
        assert float(clipped[9].item()) <= 8.0 + 1e-6

    def test_result_is_symmetric(self) -> None:
        """Clipping uses [-threshold, threshold], so symmetric inputs stay symmetric."""
        x = ops.array([-5.0, -3.0, 0.0, 3.0, 5.0])
        clipped = _clip_activation(x, quantile=0.2)
        ops.eval(clipped)
        # clipped[0] should be -clipped[4]
        assert abs(float(clipped[0].item()) + float(clipped[4].item())) < 1e-6

    def test_shape_preserved(self) -> None:
        """Output shape must match input shape."""
        x = ops.random.normal((D_MODEL,))
        ops.eval(x)
        clipped = _clip_activation(x, quantile=0.1)
        assert clipped.shape == x.shape


# ---------------------------------------------------------------------------
# _cosine_separation
# ---------------------------------------------------------------------------


class TestCosineSeparation:
    """Tests for projection-based cosine separation scoring."""

    def test_positive_gap_for_distinct_means(self) -> None:
        """When harmful projects higher than harmless, the gap is positive."""
        direction = ops.array([1.0, 0.0, 0.0, 0.0])
        harmful_mean = ops.array([3.0, 0.0, 0.0, 0.0])
        harmless_mean = ops.array([1.0, 0.0, 0.0, 0.0])
        score = _cosine_separation(harmful_mean, harmless_mean, direction)
        ops.eval(score)
        # proj_harmful = 3.0, proj_harmless = 1.0 => gap = 2.0
        assert float(score.item()) > 0.0
        assert abs(float(score.item()) - 2.0) < 1e-5

    def test_zero_gap_for_identical_means(self) -> None:
        """When harmful and harmless means are the same, the gap is zero."""
        direction = ops.array([1.0, 0.0, 0.0, 0.0])
        same_mean = ops.array([2.0, 1.0, 0.5, 0.0])
        score = _cosine_separation(same_mean, same_mean, direction)
        ops.eval(score)
        assert abs(float(score.item())) < 1e-6

    def test_negative_gap_when_harmless_projects_higher(self) -> None:
        """When harmless projects higher, the gap is negative."""
        direction = ops.array([0.0, 1.0])
        harmful_mean = ops.array([0.0, 1.0])
        harmless_mean = ops.array([0.0, 5.0])
        score = _cosine_separation(harmful_mean, harmless_mean, direction)
        ops.eval(score)
        assert float(score.item()) < 0.0


# ---------------------------------------------------------------------------
# _best_direction
# ---------------------------------------------------------------------------


class TestBestDirection:
    """Tests for best refusal direction selection across layers."""

    def test_returns_unit_vector(self) -> None:
        """The returned direction must be a unit vector."""
        harmful = [ops.array([3.0, 0.0, 0.0, 0.0]) for _ in range(2)]
        harmless = [ops.array([1.0, 0.0, 0.0, 0.0]) for _ in range(2)]
        direction, _layer, _scores = _best_direction(harmful, harmless)
        ops.eval(direction)
        norm = float(ops.linalg.norm(direction).item())
        assert abs(norm - 1.0) < 1e-5

    def test_selects_layer_with_highest_score(self) -> None:
        """The best_layer index should correspond to the maximum cosine score."""
        # Layer 0: small diff, layer 1: large diff
        harmful = [
            ops.array([1.1, 0.0, 0.0, 0.0]),
            ops.array([5.0, 0.0, 0.0, 0.0]),
        ]
        harmless = [
            ops.array([1.0, 0.0, 0.0, 0.0]),
            ops.array([0.0, 0.0, 0.0, 0.0]),
        ]
        _direction, best_layer, cosine_scores = _best_direction(harmful, harmless)
        assert len(cosine_scores) == 2
        assert best_layer == cosine_scores.index(max(cosine_scores))

    def test_handles_two_layers(self) -> None:
        """Basic structural test: works with exactly 2 layers."""
        harmful = [
            ops.random.normal((D_MODEL,)),
            ops.random.normal((D_MODEL,)),
        ]
        harmless = [
            ops.random.normal((D_MODEL,)),
            ops.random.normal((D_MODEL,)),
        ]
        ops.eval(*harmful, *harmless)
        direction, best_layer, scores = _best_direction(harmful, harmless)
        assert direction.shape == (D_MODEL,)
        assert best_layer in (0, 1)
        assert len(scores) == 2

    def test_direction_shape_matches_activations(self) -> None:
        """Direction dimensionality must match activation dimensionality."""
        dim = 8
        harmful = [ops.random.normal((dim,)) for _ in range(3)]
        harmless = [ops.random.normal((dim,)) for _ in range(3)]
        ops.eval(*harmful, *harmless)
        direction, _layer, _scores = _best_direction(harmful, harmless)
        assert direction.shape == (dim,)


# ---------------------------------------------------------------------------
# _forward_collect (additional coverage beyond test_measure.py)
# ---------------------------------------------------------------------------


class TestForwardCollectAdditional:
    """Additional coverage for layer-by-layer forward collection."""

    def test_position_zero_gives_first_token_activations(
        self, mock_model: MockCausalLM,
    ) -> None:
        """Explicit position=0 should return activations from the first token."""
        token_ids = ops.array([[10, 20, 30]])
        residuals = _forward_collect(mock_model, token_ids, token_position=0)
        assert len(residuals) == NUM_LAYERS
        for r in residuals:
            assert r.shape == (D_MODEL,)

    def test_single_token_sequence(
        self, mock_model: MockCausalLM,
    ) -> None:
        """A single-token input should produce valid activations."""
        token_ids = ops.array([[5]])
        residuals = _forward_collect(mock_model, token_ids, token_position=-1)
        assert len(residuals) == NUM_LAYERS
        for r in residuals:
            assert r.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# _collect_activations
# ---------------------------------------------------------------------------


class TestCollectActivations:
    """Tests for Welford streaming mean activation collection."""

    def test_returns_correct_number_of_layers(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Output list length must match the number of model layers."""
        prompts = ["hello world", "how are you"]
        means = _collect_activations(mock_model, mock_tokenizer, prompts)
        assert len(means) == NUM_LAYERS

    def test_welford_stable_with_many_prompts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Welford mean of repeated prompts equals single-prompt result."""
        prompt = "stable test"
        single = _collect_activations(
            mock_model, mock_tokenizer, [prompt],
        )
        # 8 copies of the same prompt -- Welford mean should converge to same value
        multi = _collect_activations(
            mock_model, mock_tokenizer, [prompt] * 8,
        )
        for s, m in zip(single, multi, strict=True):
            assert ops.allclose(s, m, atol=1e-5)

    def test_handles_single_prompt(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Single prompt produces d_model-shaped output without error."""
        means = _collect_activations(
            mock_model, mock_tokenizer, ["only one"],
        )
        assert len(means) == NUM_LAYERS
        for m in means:
            assert m.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# _collect_per_prompt_activations
# ---------------------------------------------------------------------------


class TestCollectPerPromptActivations:
    """Tests for per-prompt (non-averaged) activation collection."""

    def test_correct_shape(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Each layer should have shape (n_prompts, d_model)."""
        prompts = ["prompt one", "prompt two", "prompt three"]
        per_layer = _collect_per_prompt_activations(
            mock_model, mock_tokenizer, prompts,
        )
        assert len(per_layer) == NUM_LAYERS
        for layer_acts in per_layer:
            assert layer_acts.shape == (3, D_MODEL)

    def test_single_prompt_shape(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """A single prompt should produce shape (1, d_model) per layer."""
        per_layer = _collect_per_prompt_activations(
            mock_model, mock_tokenizer, ["solo"],
        )
        assert len(per_layer) == NUM_LAYERS
        for layer_acts in per_layer:
            assert layer_acts.shape == (1, D_MODEL)
