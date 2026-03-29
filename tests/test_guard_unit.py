# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for guard zone classification and cache checkpointing."""

import pytest

from tests.conftest import MockCausalLM, MockTokenizer
from vauban._array import Array
from vauban.guard import (
    GuardSession,
    _classify_zone,
    _restore_cache,
    _snapshot_cache,
    guard_generate,
)
from vauban.types import GuardConfig, GuardTierSpec

# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------


class TestClassifyZone:
    """Tests for the _classify_zone threshold resolution."""

    @pytest.fixture
    def tiers(self) -> list[GuardTierSpec]:
        return [
            GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
            GuardTierSpec(threshold=0.3, zone="yellow", alpha=0.5),
            GuardTierSpec(threshold=0.6, zone="orange", alpha=1.5),
            GuardTierSpec(threshold=0.9, zone="red", alpha=3.0),
        ]

    def test_below_all_thresholds(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(-0.5, tiers)
        assert zone == "green"
        assert alpha == 0.0

    def test_at_green_boundary(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(0.0, tiers)
        assert zone == "green"
        assert alpha == 0.0

    def test_in_yellow_range(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(0.4, tiers)
        assert zone == "yellow"
        assert alpha == 0.5

    def test_at_yellow_boundary(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(0.3, tiers)
        assert zone == "yellow"
        assert alpha == 0.5

    def test_in_orange_range(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(0.7, tiers)
        assert zone == "orange"
        assert alpha == 1.5

    def test_in_red_range(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(1.0, tiers)
        assert zone == "red"
        assert alpha == 3.0

    def test_at_red_boundary(self, tiers: list[GuardTierSpec]) -> None:
        zone, alpha = _classify_zone(0.9, tiers)
        assert zone == "red"
        assert alpha == 3.0

    def test_single_tier(self) -> None:
        tiers = [GuardTierSpec(threshold=0.5, zone="yellow", alpha=1.0)]
        zone, alpha = _classify_zone(0.6, tiers)
        assert zone == "yellow"
        assert alpha == 1.0

    def test_single_tier_below(self) -> None:
        tiers = [GuardTierSpec(threshold=0.5, zone="yellow", alpha=1.0)]
        zone, alpha = _classify_zone(0.3, tiers)
        assert zone == "yellow"
        assert alpha == 1.0


# ---------------------------------------------------------------------------
# KV cache checkpoint
# ---------------------------------------------------------------------------


class TestCacheCheckpoint:
    """Tests for cache snapshot and restore."""

    def test_snapshot_captures_offset(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        from vauban import _ops as ops

        cache = mock_model.make_cache()
        dummy_tok = ops.array([[0]])

        checkpoint = _snapshot_cache(cache, [1, 2, 3], dummy_tok)
        assert checkpoint.token_count == 3
        assert checkpoint.generated_ids == [1, 2, 3]
        assert len(checkpoint.layer_states) == len(cache)
        for _k, _v, offset in checkpoint.layer_states:
            assert offset == 0

    def test_restore_resets_offset(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        from vauban import _ops as ops

        cache = mock_model.make_cache()
        dummy_tok = ops.array([[0]])
        # Take snapshot at offset=0
        checkpoint = _snapshot_cache(cache, [], dummy_tok)

        # Advance cache
        for lc in cache:
            lc.offset = 10

        # Restore
        restored = _restore_cache(mock_model, checkpoint)
        for lc in restored:
            assert lc.offset == 0

    def test_generated_ids_are_copied(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        """Ensure checkpoint holds a copy, not a reference."""
        from vauban import _ops as ops

        cache = mock_model.make_cache()
        dummy_tok = ops.array([[0]])
        original = [1, 2, 3]
        checkpoint = _snapshot_cache(cache, original, dummy_tok)
        original.append(4)
        assert checkpoint.generated_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Guard generate — basic integration with mock model
# ---------------------------------------------------------------------------


class TestGuardGenerate:
    """Integration tests for the guard generation loop."""

    def test_all_green_no_rewinds(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """With high thresholds, everything stays green."""
        config = GuardConfig(
            prompts=["test"],
            tiers=[
                GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
                GuardTierSpec(threshold=999.0, zone="yellow", alpha=0.5),
                GuardTierSpec(threshold=999.0, zone="orange", alpha=1.5),
                GuardTierSpec(threshold=999.0, zone="red", alpha=3.0),
            ],
            max_tokens=5,
        )
        result = guard_generate(
            mock_model, mock_tokenizer, "test", direction,
            layers=[0], config=config,
        )
        assert result.total_rewinds == 0
        assert not result.circuit_broken
        assert result.tokens_generated == 5
        assert all(e.zone == "green" for e in result.events)

    def test_red_triggers_circuit_break(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """With very low red threshold, circuit breaks immediately."""
        config = GuardConfig(
            prompts=["test"],
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="red", alpha=3.0),
            ],
            max_tokens=10,
        )
        result = guard_generate(
            mock_model, mock_tokenizer, "test", direction,
            layers=[0], config=config,
        )
        assert result.circuit_broken
        assert result.tokens_generated < 10

    def test_orange_triggers_rewind(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """With low orange threshold, rewinds are triggered."""
        config = GuardConfig(
            prompts=["test"],
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="orange", alpha=1.5),
            ],
            max_tokens=10,
            max_rewinds=2,
        )
        result = guard_generate(
            mock_model, mock_tokenizer, "test", direction,
            layers=[0], config=config,
        )
        # Should exhaust rewinds and circuit break
        assert result.total_rewinds > 0
        assert result.circuit_broken

    def test_result_has_zone_counts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = GuardConfig(
            prompts=["test"],
            tiers=[
                GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
                GuardTierSpec(threshold=999.0, zone="red", alpha=3.0),
            ],
            max_tokens=3,
        )
        result = guard_generate(
            mock_model, mock_tokenizer, "test", direction,
            layers=[0], config=config,
        )
        assert "green" in result.final_zone_counts
        assert sum(result.final_zone_counts.values()) >= result.tokens_generated

    def test_max_rewinds_zero_means_no_rewind(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """With max_rewinds=0, orange immediately circuit breaks."""
        config = GuardConfig(
            prompts=["test"],
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="orange", alpha=1.5),
            ],
            max_tokens=10,
            max_rewinds=0,
        )
        result = guard_generate(
            mock_model, mock_tokenizer, "test", direction,
            layers=[0], config=config,
        )
        assert result.circuit_broken
        assert result.total_rewinds == 1  # tried once, exceeded max


# ---------------------------------------------------------------------------
# GuardSession — integration API
# ---------------------------------------------------------------------------


class TestGuardSession:
    """Tests for the stateful GuardSession integration API."""

    def test_all_green_session(self, direction: Array) -> None:
        """With high thresholds, all checks return green/pass."""
        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
                GuardTierSpec(threshold=999.0, zone="red", alpha=3.0),
            ],
        )
        # Feed small random activations (should stay green)
        from vauban import _ops as ops

        for _ in range(5):
            act = ops.zeros_like(direction)
            verdict = session.check(act)
            assert verdict.zone == "green"
            assert verdict.action == "pass"

        assert session.rewind_count == 0
        assert not session.circuit_broken
        assert session.step == 5
        assert len(session.events) == 5

    def test_red_triggers_break(self, direction: Array) -> None:
        """With very low threshold, immediately breaks."""
        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="red", alpha=3.0),
            ],
        )
        from vauban import _ops as ops

        verdict = session.check(ops.zeros_like(direction))
        assert verdict.zone == "red"
        assert verdict.action == "break"
        assert session.circuit_broken

    def test_orange_rewind_cycle(self, direction: Array) -> None:
        """Orange zone triggers rewind; calling rewind() resets step."""
        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="orange", alpha=1.5),
            ],
            max_rewinds=2,
        )
        from vauban import _ops as ops

        act = ops.zeros_like(direction)

        verdict = session.check(act)
        assert verdict.action == "rewind"
        assert session.rewind_count == 1

        offset = session.rewind()
        assert offset == 0
        assert session.step == 0

        # Second rewind
        verdict = session.check(act)
        assert verdict.action == "rewind"
        assert session.rewind_count == 2

        session.rewind()

        # Third attempt exceeds max_rewinds → break
        verdict = session.check(act)
        assert verdict.action == "break"
        assert session.circuit_broken

    def test_reset_clears_state(self, direction: Array) -> None:
        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="red", alpha=3.0),
            ],
        )
        from vauban import _ops as ops

        session.check(ops.zeros_like(direction))
        assert session.circuit_broken

        session.reset()
        assert not session.circuit_broken
        assert session.step == 0
        assert session.rewind_count == 0
        assert len(session.events) == 0

    def test_checkpoint_advances_on_green(self, direction: Array) -> None:
        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
                GuardTierSpec(threshold=999.0, zone="red", alpha=3.0),
            ],
            checkpoint_interval=2,
        )
        from vauban import _ops as ops

        act = ops.zeros_like(direction)

        # Step 0: green, streak=1, no checkpoint yet
        session.check(act)
        assert session.checkpoint_offset == 0

        # Step 1: green, streak=2 → checkpoint at step 1
        session.check(act)
        assert session.checkpoint_offset == 1

    def test_circuit_broken_returns_break(self, direction: Array) -> None:
        """After circuit break, all subsequent checks return break."""
        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=-999.0, zone="red", alpha=3.0),
            ],
        )
        from vauban import _ops as ops

        act = ops.zeros_like(direction)
        session.check(act)  # breaks
        assert session.circuit_broken

        # Subsequent check still returns break
        verdict = session.check(act)
        assert verdict.action == "break"
        assert verdict.zone == "red"

    def test_from_file(
        self, direction: Array, tmp_path: "Path",  # noqa: F821
    ) -> None:
        """Test loading direction from .npy file."""
        import json

        import numpy as np

        # Save direction
        dir_path = tmp_path / "direction.npy"
        np.save(str(dir_path), np.array(direction))

        # Save tiers
        tiers_data = [
            {"threshold": 0.0, "zone": "green", "alpha": 0.0},
            {"threshold": 0.5, "zone": "red", "alpha": 3.0},
        ]
        tiers_path = tmp_path / "tiers.json"
        tiers_path.write_text(json.dumps(tiers_data))

        session = GuardSession.from_file(
            str(dir_path), str(tiers_path), max_rewinds=5,
        )
        assert len(session.tiers) == 2
        assert session.tiers[1].zone == "red"
