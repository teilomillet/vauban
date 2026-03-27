# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the intent alignment module."""

import pytest

from vauban.types import IntentCheckResult, IntentConfig, IntentState


class TestIntentConfig:
    """Tests for IntentConfig defaults and parsing."""

    def test_defaults(self) -> None:
        config = IntentConfig()
        assert config.mode == "embedding"
        assert config.similarity_threshold == 0.7
        assert config.max_tokens == 10

    def test_judge_mode(self) -> None:
        config = IntentConfig(mode="judge")
        assert config.mode == "judge"


class TestIntentState:
    """Tests for IntentState dataclass."""

    def test_without_activation(self) -> None:
        state = IntentState(user_request="Send me a summary", activation=None)
        assert state.user_request == "Send me a summary"
        assert state.activation is None


class TestIntentCheckResult:
    """Tests for IntentCheckResult dataclass."""

    def test_aligned(self) -> None:
        result = IntentCheckResult(aligned=True, score=0.85, mode="embedding")
        assert result.aligned is True

    def test_misaligned(self) -> None:
        result = IntentCheckResult(aligned=False, score=0.3, mode="embedding")
        assert result.aligned is False


class TestIntentConfigParsing:
    """Tests for [intent] config parsing."""

    def test_parse_valid(self) -> None:
        from vauban.config._parse_intent import _parse_intent

        raw = {
            "intent": {
                "mode": "judge",
                "target_layer": 10,
                "similarity_threshold": 0.8,
                "max_tokens": 5,
            },
        }
        config = _parse_intent(raw)
        assert config is not None
        assert config.mode == "judge"
        assert config.target_layer == 10
        assert config.similarity_threshold == pytest.approx(0.8)

    def test_invalid_mode(self) -> None:
        from vauban.config._parse_intent import _parse_intent

        raw = {"intent": {"mode": "invalid"}}
        with pytest.raises(ValueError, match="invalid"):
            _parse_intent(raw)

    def test_missing_returns_none(self) -> None:
        from vauban.config._parse_intent import _parse_intent

        assert _parse_intent({}) is None
