# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for ``vauban.flywheel._attack`` coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from tests.conftest import MockTokenizer
from vauban.flywheel._attack import warmstart_gcg_payloads
from vauban.types import (
    EnvironmentConfig,
    EnvironmentTarget,
    EnvironmentTask,
    FlywheelConfig,
    Payload,
)

if TYPE_CHECKING:
    from vauban.types import CausalLM, SoftPromptConfig


@dataclass(frozen=True, slots=True)
class _FakeAttackResult:
    """Minimal GCG result carrying token ids only."""

    token_ids: list[int] | None


def _make_world(task: str) -> EnvironmentConfig:
    """Build a minimal environment config with task text."""
    return EnvironmentConfig(
        system_prompt="system",
        tools=[],
        target=EnvironmentTarget(function="tool"),
        task=EnvironmentTask(content=task),
        injection_surface="surface",
    )


class TestWarmstartGcgPayloads:
    """Branch coverage for the flywheel GCG warm-start helper."""

    def test_returns_empty_when_no_world_prompts(self) -> None:
        tokenizer = MockTokenizer(32)
        result = warmstart_gcg_payloads(
            cast("CausalLM", object()),
            tokenizer,
            [],
            [Payload(text="seed", source="lib", cycle_discovered=0)],
            FlywheelConfig(),
            None,
        )

        assert result == []

    def test_uses_infix_position_and_returns_stripped_payload(self) -> None:
        tokenizer = MockTokenizer(64)
        captured: dict[str, object] = {}

        def _stub_gcg_attack(*args: object, **kwargs: object) -> _FakeAttackResult:
            captured["config"] = args[3]
            captured["direction"] = args[4]
            captured["environment_config"] = kwargs["environment_config"]
            return _FakeAttackResult([0, 1, 2])

        config = FlywheelConfig(
            positions=["suffix", "infix"],
            gcg_steps=7,
            gcg_n_tokens=4,
        )
        worlds = [_make_world("task one"), _make_world("task two")]

        with patch("vauban.softprompt._gcg._gcg_attack", side_effect=_stub_gcg_attack):
            result = warmstart_gcg_payloads(
                cast("CausalLM", object()),
                tokenizer,
                worlds,
                [Payload(text="seed", source="lib", cycle_discovered=0)],
                config,
                direction=None,
            )

        sp_config = cast("SoftPromptConfig", captured["config"])
        env_config = captured["environment_config"]
        assert env_config == worlds[0]
        assert sp_config.token_position == "infix"
        assert sp_config.n_steps == 7
        assert sp_config.n_tokens == 4
        assert result == [tokenizer.decode([0, 1, 2]).strip()]

    def test_returns_empty_when_decoded_payload_is_blank(self) -> None:
        class _BlankTokenizer(MockTokenizer):
            def decode(self, token_ids: list[int]) -> str:
                return "   "

        with patch(
            "vauban.softprompt._gcg._gcg_attack",
            return_value=_FakeAttackResult([1, 2, 3]),
        ):
            result = warmstart_gcg_payloads(
                cast("CausalLM", object()),
                _BlankTokenizer(64),
                [_make_world("task")],
                [],
                FlywheelConfig(),
                None,
            )

        assert result == []

    def test_returns_empty_when_token_ids_missing_or_attack_raises(self) -> None:
        tokenizer = MockTokenizer(64)

        with patch(
            "vauban.softprompt._gcg._gcg_attack",
            return_value=_FakeAttackResult(None),
        ):
            assert warmstart_gcg_payloads(
                cast("CausalLM", object()),
                tokenizer,
                [_make_world("task")],
                [],
                FlywheelConfig(),
                None,
            ) == []

        with patch(
            "vauban.softprompt._gcg._gcg_attack",
            side_effect=RuntimeError("boom"),
        ):
            assert warmstart_gcg_payloads(
                cast("CausalLM", object()),
                tokenizer,
                [_make_world("task")],
                [],
                FlywheelConfig(),
                None,
            ) == []
