# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the defense stack composition module."""

import pytest

from vauban.types import (
    DefenseStackConfig,
    DefenseStackResult,
    SICPromptResult,
)


class TestDefenseStackConfig:
    """Tests for DefenseStackConfig defaults."""

    def test_defaults(self) -> None:
        config = DefenseStackConfig()
        assert config.scan is None
        assert config.sic is None
        assert config.policy is None
        assert config.intent is None
        assert config.fail_fast is True


class TestDefenseStackResult:
    """Tests for DefenseStackResult."""

    def test_not_blocked(self) -> None:
        result = DefenseStackResult(
            blocked=False,
            layer_that_blocked=None,
        )
        assert result.blocked is False
        assert result.reasons == []

    def test_blocked_by_scan(self) -> None:
        result = DefenseStackResult(
            blocked=True,
            layer_that_blocked="scan",
            reasons=["Injection detected"],
        )
        assert result.blocked is True
        assert result.layer_that_blocked == "scan"
        assert result.sic_result is None

    def test_blocked_by_sic(self) -> None:
        sic = SICPromptResult(
            clean_prompt="safe content",
            blocked=True,
            iterations=3,
            initial_score=-0.5,
            final_score=-0.3,
        )
        result = DefenseStackResult(
            blocked=True,
            layer_that_blocked="sic",
            sic_result=sic,
            reasons=["SIC: content blocked after sanitization failed"],
        )
        assert result.blocked is True
        assert result.layer_that_blocked == "sic"
        assert result.sic_result is sic
        assert result.sic_result.iterations == 3


class TestDefendConfigParsing:
    """Tests for [defend] config parsing."""

    def test_parse_valid(self) -> None:
        from vauban.config._parse_defend import _parse_defend

        raw = {
            "defend": {"fail_fast": False},
            "scan": {"threshold": 0.5},
        }
        config = _parse_defend(raw)
        assert config is not None
        assert config.fail_fast is False
        assert config.scan is not None
        assert config.scan.threshold == pytest.approx(0.5)

    def test_missing_returns_none(self) -> None:
        from vauban.config._parse_defend import _parse_defend

        assert _parse_defend({}) is None

    def test_with_policy(self) -> None:
        from vauban.config._parse_defend import _parse_defend

        raw = {
            "defend": {"fail_fast": True},
            "policy": {
                "default_action": "block",
                "rules": [
                    {
                        "name": "test",
                        "action": "block",
                        "tool_pattern": "*",
                    },
                ],
            },
        }
        config = _parse_defend(raw)
        assert config is not None
        assert config.policy is not None
        assert config.policy.default_action == "block"
