"""Tests for EGD/COLD gradient starvation fix.

Covers random init, temperature annealing, and entropy bonus.
"""

from __future__ import annotations

import random

import pytest
from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

from vauban import _ops as ops
from vauban.softprompt._cold import _cold_attack
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._runtime import (
    _build_one_hot,
    _build_peaked_probs,
    _compute_temperature,
    _resolve_init_ids,
    _sample_random_init_ids,
)
from vauban.types import SoftPromptConfig, SoftPromptResult

# ---------------------------------------------------------------------------
# _compute_temperature
# ---------------------------------------------------------------------------


class TestComputeTemperature:
    """Tests for temperature schedule computation."""

    def test_constant_returns_base(self) -> None:
        assert _compute_temperature(0.5, 0, 10, "constant") == 0.5
        assert _compute_temperature(0.5, 5, 10, "constant") == 0.5
        assert _compute_temperature(0.5, 9, 10, "constant") == 0.5

    def test_linear_endpoints(self) -> None:
        t0 = _compute_temperature(0.5, 0, 10, "linear")
        t_end = _compute_temperature(0.5, 9, 10, "linear")
        assert abs(t0 - 2.0) < 1e-6
        assert abs(t_end - 0.5) < 1e-6

    def test_cosine_endpoints(self) -> None:
        t0 = _compute_temperature(0.5, 0, 10, "cosine")
        t_end = _compute_temperature(0.5, 9, 10, "cosine")
        assert abs(t0 - 2.0) < 1e-6
        assert abs(t_end - 0.5) < 1e-6

    def test_cosine_midpoint_between_endpoints(self) -> None:
        t_mid = _compute_temperature(0.5, 4, 10, "cosine")
        assert 0.5 < t_mid < 2.0

    def test_high_base_no_anneal(self) -> None:
        """When base >= 2.0, high == base so annealing is a flat line."""
        t0 = _compute_temperature(3.0, 0, 10, "linear")
        t_end = _compute_temperature(3.0, 9, 10, "linear")
        assert abs(t0 - 3.0) < 1e-6
        assert abs(t_end - 3.0) < 1e-6

    def test_n_steps_one_returns_base(self) -> None:
        """Single step always returns base regardless of schedule."""
        assert _compute_temperature(0.5, 0, 1, "linear") == 0.5
        assert _compute_temperature(0.5, 0, 1, "cosine") == 0.5

    def test_linear_monotonically_decreasing(self) -> None:
        temps = [_compute_temperature(0.5, s, 20, "linear") for s in range(20)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-9

    def test_cosine_monotonically_decreasing(self) -> None:
        temps = [_compute_temperature(0.5, s, 20, "cosine") for s in range(20)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# _sample_random_init_ids
# ---------------------------------------------------------------------------


class TestSampleRandomInitIds:
    """Tests for random token ID sampling."""

    def test_correct_length(self) -> None:
        random.seed(0)
        ids = _sample_random_init_ids(8, [1, 2, 3])
        assert len(ids) == 8

    def test_respects_allowed_indices(self) -> None:
        random.seed(0)
        allowed = [10, 20, 30]
        ids = _sample_random_init_ids(100, allowed)
        assert all(i in allowed for i in ids)

    def test_single_allowed_index(self) -> None:
        ids = _sample_random_init_ids(5, [42])
        assert ids == [42, 42, 42, 42, 42]

    def test_empty_allowed_raises(self) -> None:
        with pytest.raises(ValueError, match="No allowed token indices"):
            _sample_random_init_ids(3, [])


# ---------------------------------------------------------------------------
# _build_one_hot / _build_peaked_probs / _resolve_init_ids
# ---------------------------------------------------------------------------


class TestBuildOneHot:
    """Tests for one-hot matrix construction."""

    def test_shape(self) -> None:
        oh = _build_one_hot([0, 3, 7], 10)
        assert oh.shape == (3, 10)

    def test_values(self) -> None:
        oh = _build_one_hot([2], 5)
        row = [float(oh[0, i].item()) for i in range(5)]
        assert row == [0.0, 0.0, 1.0, 0.0, 0.0]


class TestBuildPeakedProbs:
    """Tests for peaked probability distribution."""

    def test_shape(self) -> None:
        p = _build_peaked_probs([0, 1], 10)
        assert p.shape == (2, 10)

    def test_rows_sum_to_one(self) -> None:
        p = _build_peaked_probs([3, 7], 20)
        for row_idx in range(2):
            row_sum = float(ops.sum(p[row_idx]).item())
            assert abs(row_sum - 1.0) < 1e-5

    def test_peak_mass(self) -> None:
        p = _build_peaked_probs([5], 100, peak_mass=0.9)
        peak_val = float(p[0, 5].item())
        assert peak_val > 0.85  # close to 0.9


class TestResolveInitIds:
    """Tests for init ID resolution."""

    def test_with_init_tokens(self) -> None:
        ids = _resolve_init_ids([1, 2, 3], 5, None, 100)
        assert ids == [1, 2, 3, 0, 0]  # padded

    def test_truncation(self) -> None:
        ids = _resolve_init_ids([1, 2, 3, 4, 5], 3, None, 100)
        assert ids == [1, 2, 3]

    def test_random_fallback(self) -> None:
        random.seed(42)
        ids = _resolve_init_ids(None, 4, None, 100)
        assert len(ids) == 4
        assert all(0 <= i < 100 for i in ids)


# ---------------------------------------------------------------------------
# Config parsing: temperature_schedule and entropy_weight
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Tests for new config fields."""

    def test_temperature_schedule_default(self) -> None:
        cfg = SoftPromptConfig(mode="egd")
        assert cfg.temperature_schedule == "constant"

    def test_entropy_weight_default(self) -> None:
        cfg = SoftPromptConfig(mode="egd")
        assert cfg.entropy_weight == 0.0

    def test_temperature_schedule_valid(self) -> None:
        for sched in ("constant", "linear", "cosine"):
            cfg = SoftPromptConfig(mode="egd", temperature_schedule=sched)
            assert cfg.temperature_schedule == sched

    def test_toml_parse_temperature_schedule(self) -> None:
        from vauban.config._parse_softprompt import _parse_softprompt

        raw: dict[str, object] = {
            "softprompt": {
                "mode": "egd",
                "temperature_schedule": "cosine",
                "target_prefixes": ["Sure"],
            },
        }
        cfg = _parse_softprompt(raw)
        assert cfg is not None
        assert cfg.temperature_schedule == "cosine"

    def test_toml_parse_temperature_schedule_invalid(self) -> None:
        from vauban.config._parse_softprompt import _parse_softprompt

        raw: dict[str, object] = {
            "softprompt": {
                "mode": "egd",
                "temperature_schedule": "exponential",
                "target_prefixes": ["Sure"],
            },
        }
        with pytest.raises(ValueError, match="temperature_schedule"):
            _parse_softprompt(raw)

    def test_toml_parse_entropy_weight(self) -> None:
        from vauban.config._parse_softprompt import _parse_softprompt

        raw: dict[str, object] = {
            "softprompt": {
                "mode": "egd",
                "entropy_weight": 0.05,
                "target_prefixes": ["Sure"],
            },
        }
        cfg = _parse_softprompt(raw)
        assert cfg is not None
        assert cfg.entropy_weight == 0.05

    def test_toml_parse_entropy_weight_negative(self) -> None:
        from vauban.config._parse_softprompt import _parse_softprompt

        raw: dict[str, object] = {
            "softprompt": {
                "mode": "egd",
                "entropy_weight": -0.1,
                "target_prefixes": ["Sure"],
            },
        }
        with pytest.raises(ValueError, match="entropy_weight"):
            _parse_softprompt(raw)


# ---------------------------------------------------------------------------
# EGD/COLD smoke tests with new features
# ---------------------------------------------------------------------------


class TestEgdSmoke:
    """Smoke tests for EGD with temperature annealing and entropy bonus."""

    def test_cosine_schedule(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            temperature_schedule="cosine",
            egd_temperature=0.5,
        )

        result = _egd_attack(model, tokenizer, ["test"], config, None)
        assert isinstance(result, SoftPromptResult)
        assert result.mode == "egd"
        assert len(result.loss_history) == 5
        assert result.token_ids is not None
        assert all(0 <= t < VOCAB_SIZE for t in result.token_ids)

    def test_entropy_bonus(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            entropy_weight=0.01,
        )

        result = _egd_attack(model, tokenizer, ["test"], config, None)
        assert isinstance(result, SoftPromptResult)
        assert result.mode == "egd"
        assert len(result.loss_history) == 5

    def test_combined(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            temperature_schedule="linear",
            egd_temperature=0.5,
            entropy_weight=0.01,
        )

        result = _egd_attack(model, tokenizer, ["test"], config, None)
        assert isinstance(result, SoftPromptResult)
        assert len(result.loss_history) == 5


class TestColdSmoke:
    """Smoke tests for COLD with temperature annealing and entropy bonus."""

    def test_cosine_schedule(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            temperature_schedule="cosine",
            cold_temperature=0.5,
            cold_noise_scale=0.0,
        )

        result = _cold_attack(model, tokenizer, ["test"], config, None)
        assert isinstance(result, SoftPromptResult)
        assert result.mode == "cold"
        assert len(result.loss_history) == 5
        assert result.token_ids is not None
        assert all(0 <= t < VOCAB_SIZE for t in result.token_ids)

    def test_entropy_bonus(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            cold_noise_scale=0.0,
            entropy_weight=0.01,
        )

        result = _cold_attack(model, tokenizer, ["test"], config, None)
        assert isinstance(result, SoftPromptResult)
        assert result.mode == "cold"
        assert len(result.loss_history) == 5

    def test_combined(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            temperature_schedule="linear",
            cold_temperature=0.5,
            cold_noise_scale=0.0,
            entropy_weight=0.01,
        )

        result = _cold_attack(model, tokenizer, ["test"], config, None)
        assert isinstance(result, SoftPromptResult)
        assert len(result.loss_history) == 5


# ---------------------------------------------------------------------------
# MODE_DESCRIPTIONS validation
# ---------------------------------------------------------------------------


class TestModeDescriptions:
    """Tests for init mode description coverage."""

    def test_all_modes_have_descriptions(self) -> None:
        from vauban._init import KNOWN_MODES, MODE_DESCRIPTIONS

        missing = KNOWN_MODES - set(MODE_DESCRIPTIONS)
        assert not missing, f"Missing descriptions: {missing}"

    def test_no_extra_descriptions(self) -> None:
        from vauban._init import KNOWN_MODES, MODE_DESCRIPTIONS

        extra = set(MODE_DESCRIPTIONS) - KNOWN_MODES
        assert not extra, f"Extra descriptions: {extra}"


# ---------------------------------------------------------------------------
# Manual topic index
# ---------------------------------------------------------------------------


class TestManualTopicIndex:
    """Tests for the topic index rendering."""

    def test_topic_index_rendered_when_no_topic(self) -> None:
        from vauban.manual import render_manual

        output = render_manual(None)
        assert "Topic Index" in output
        assert "GENERAL TOPICS" in output
        assert "PIPELINE MODES" in output
        assert "CONFIG SECTIONS" in output
        # Should NOT contain full manual content
        assert "QUICKSTART" not in output

    def test_all_renders_full_manual(self) -> None:
        from vauban.manual import render_manual

        output = render_manual("all")
        assert "QUICKSTART" in output
        assert "CONFIG SECTIONS" in output

    def test_specific_topic_shows_new_fields(self) -> None:
        from vauban.manual import render_manual

        output = render_manual("softprompt")
        assert "temperature_schedule" in output
        assert "entropy_weight" in output
