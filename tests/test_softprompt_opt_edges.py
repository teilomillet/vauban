# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Edge-case coverage for softprompt optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from conftest import D_MODEL, VOCAB_SIZE, MockCausalLM, MockTokenizer

from vauban import _ops as ops
from vauban.softprompt._attack_init import AttackInitState
from vauban.softprompt._cold import _cold_attack
from vauban.softprompt._constraints import (
    _build_vocab_mask,
    _is_emoji_char,
    _is_invisible_char,
    _matches_constraint,
)
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._gcg_objective import GCGDefenseConfig, GCGSharedState
from vauban.softprompt._generation import (
    _evaluate_attack,
    _evaluate_attack_with_history,
)
from vauban.types import (
    EnvironmentConfig,
    EnvironmentResult,
    EnvironmentTarget,
    EnvironmentTask,
    SoftPromptConfig,
    ToolSchema,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer, TransformerModel


@dataclass(frozen=True, slots=True)
class _FakeTransformer:
    """Minimal transformer stub for attack-init fixtures."""

    d_model: int

    def embed_tokens(self, token_ids: Array) -> Array:
        batch, n_tokens = token_ids.shape
        return ops.ones((batch, n_tokens, self.d_model))


@dataclass(frozen=True, slots=True)
class _FixedTokenizer:
    """Tokenizer stub with a configurable decode payload."""

    eos_token_id: int
    decode_text: str

    def decode(self, token_ids: list[int]) -> str:
        return self.decode_text


@dataclass(frozen=True, slots=True)
class _DummyEnvResult:
    """Minimal environment result for post-hoc scoring."""

    reward: float


def _fake_value_and_grad(
    fn: Callable[[Array], Array],
) -> Callable[[Array], tuple[Array, Array]]:
    """Return a deterministic value-and-grad wrapper."""

    def wrapped(x: Array) -> tuple[Array, Array]:
        loss = fn(x)
        return loss, ops.zeros_like(x)

    return wrapped


def _make_shared_state(
    model: MockCausalLM,
    *,
    loss_mode: str = "targeted",
    perplexity_weight: float = 0.0,
    direction_weight: float = 0.0,
    token_position: str = "prefix",
) -> GCGSharedState:
    """Build a compact objective state for softprompt tests."""
    target_ids = ops.array([1, 2])
    direction = ops.ones((D_MODEL,)) / float(D_MODEL)
    ops.eval(target_ids, direction)
    return GCGSharedState(
        model=model,
        target_ids=target_ids,
        n_tokens=2,
        loss_mode=loss_mode,
        direction=direction,
        direction_weight=direction_weight,
        direction_mode="last",
        direction_layers=None,
        eos_token_id=VOCAB_SIZE - 1,
        eos_loss_mode="none",
        eos_loss_weight=0.0,
        ref_model=None,
        kl_ref_weight=0.0,
        refusal_ids=None,
        defense=GCGDefenseConfig(
            weight=0.0,
            sic_layer=None,
            sic_threshold=0.0,
            cast_layers=None,
            cast_threshold=0.0,
        ),
        perplexity_weight=perplexity_weight,
        token_position=token_position,
        infix_map=None,
        svf_boundary=None,
    )


def _make_attack_init(
    model: MockCausalLM,
    tokenizer: MockTokenizer,
    *,
    transfer_data: list[tuple[MockCausalLM, MockTokenizer, list[Array], Array]]
    | None = None,
    objective_state: GCGSharedState | None = None,
) -> AttackInitState:
    """Build a compact AttackInitState for softprompt optimizer tests."""
    transformer = _FakeTransformer(d_model=D_MODEL)
    embed_matrix = ops.eye(D_MODEL)
    target_ids = ops.array([1, 2])
    prompt_ids = [ops.array([0, 1]), ops.array([1, 2])]
    ops.eval(embed_matrix, target_ids)
    prepared_transfer = cast(
        "list[tuple[CausalLM, Tokenizer, list[Array], Array]]",
        transfer_data if transfer_data is not None else [],
    )
    return AttackInitState(
        transformer=cast("TransformerModel", transformer),
        vocab_size=D_MODEL,
        d_model=D_MODEL,
        embed_matrix=embed_matrix,
        target_ids=target_ids,
        all_prompt_ids=prompt_ids,
        objective_state=(
            objective_state
            if objective_state is not None
            else _make_shared_state(model)
        ),
        transfer_data=prepared_transfer,
    )


class TestConstraintHelpers:
    """Direct branch coverage for token-constraint helpers."""

    @pytest.mark.parametrize(
        ("text", "constraint", "expected"),
        [
            ("ABC", "ascii", True),
            ("abc", "alpha", True),
            ("abc 123", "alphanumeric", True),
            ("你好", "non_latin", True),
            ("汉字", "chinese", True),
            ("123!?", "non_alphabetic", True),
            ("\u200b", "invisible", True),
            ("a\u0301", "zalgo", True),
            ("😀", "emoji", True),
            ("", "ascii", False),
        ],
    )
    def test_matches_constraint_branches(
        self,
        text: str,
        constraint: str,
        expected: bool,
    ) -> None:
        assert _matches_constraint(text, constraint) is expected

    def test_unknown_constraint_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown token constraint"):
            _matches_constraint("test", "bogus")

    def test_invisible_and_emoji_helpers(self) -> None:
        assert _is_invisible_char("\u200b") is True
        assert _is_invisible_char(" ") is False
        assert _is_emoji_char("😀") is True
        assert _is_emoji_char("A") is False

    def test_build_vocab_mask_missing_glitch_inputs_raises(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)

        with pytest.raises(ValueError, match="exclude_glitch"):
            _build_vocab_mask(
                tokenizer,
                VOCAB_SIZE,
                "exclude_glitch",
            )


class TestGenerationHelpers:
    """Direct coverage for generation helper branches."""

    def test_evaluate_attack_defensive_eos_and_system_prompt(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        tokenizer = _FixedTokenizer(eos_token_id=VOCAB_SIZE - 1, decode_text="I refuse")
        soft_embeds = ops.ones((1, 2, D_MODEL))
        logits_prefix = ops.array([[[0.0, 1.0, 0.0]]])
        logits_eos = ops.array([[[0.0, 0.0, 1.0]]])
        captured: list[list[dict[str, str]]] = []

        def fake_encode(_tokenizer: object, messages: list[dict[str, str]]) -> Array:
            captured.append(messages)
            return ops.array([1, 2, 3])

        with (
            patch("vauban.softprompt._generation.DEFAULT_REFUSAL_PHRASES", ["refuse"]),
            patch(
                "vauban.softprompt._generation.get_transformer", return_value=object()
            ),
            patch("vauban.softprompt._generation.make_cache", return_value=[object()]),
            patch(
                "vauban.softprompt._generation.make_ssm_mask",
                return_value=ops.zeros((1, 1, 1)),
            ),
            patch(
                "vauban.softprompt._generation.encode_chat_prompt",
                side_effect=fake_encode,
            ),
            patch(
                "vauban.softprompt._generation._prefill_with_cache",
                return_value=logits_prefix,
            ),
            patch(
                "vauban.softprompt._generation._decode_step", return_value=logits_eos
            ),
        ):
            from vauban.types import SoftPromptConfig

            config = SoftPromptConfig(
                mode="continuous",
                max_gen_tokens=2,
                loss_mode="defensive",
                system_prompt="system",
            )
            success_rate, responses = _evaluate_attack(
                mock_model,
                cast("Tokenizer", tokenizer),
                ["hello"],
                soft_embeds,
                config,
            )

        assert success_rate == 1.0
        assert responses == ["I refuse"]
        assert captured[0][0]["role"] == "system"
        assert captured[0][-1]["role"] == "user"

    def test_evaluate_attack_with_history_honors_system_prompt_and_eos(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        tokenizer = _FixedTokenizer(
            eos_token_id=VOCAB_SIZE - 1, decode_text="safe output"
        )
        soft_embeds = ops.ones((1, 2, D_MODEL))
        logits_prefix = ops.array([[[0.0, 1.0, 0.0]]])
        logits_eos = ops.array([[[0.0, 0.0, 1.0]]])
        captured: list[list[dict[str, str]]] = []

        def fake_encode(_tokenizer: object, messages: list[dict[str, str]]) -> Array:
            captured.append(messages)
            return ops.array([1, 2, 3])

        with (
            patch("vauban.softprompt._generation.DEFAULT_REFUSAL_PHRASES", ["refuse"]),
            patch(
                "vauban.softprompt._generation.get_transformer", return_value=object()
            ),
            patch("vauban.softprompt._generation.make_cache", return_value=[object()]),
            patch(
                "vauban.softprompt._generation.make_ssm_mask",
                return_value=ops.zeros((1, 1, 1)),
            ),
            patch(
                "vauban.softprompt._generation.encode_chat_prompt",
                side_effect=fake_encode,
            ),
            patch(
                "vauban.softprompt._generation._prefill_with_cache",
                return_value=logits_prefix,
            ),
            patch(
                "vauban.softprompt._generation._decode_step", return_value=logits_eos
            ),
        ):
            from vauban.types import SoftPromptConfig

            config = SoftPromptConfig(
                mode="continuous",
                max_gen_tokens=2,
                loss_mode="targeted",
                system_prompt="system",
                token_position="suffix",
            )
            success_rate, responses = _evaluate_attack_with_history(
                mock_model,
                cast("Tokenizer", tokenizer),
                ["hello"],
                soft_embeds,
                config,
                [{"role": "assistant", "content": "previous"}],
            )

        assert success_rate == 1.0
        assert responses == ["safe output"]
        assert captured[0][0]["role"] == "system"
        assert captured[0][1]["role"] == "assistant"
        assert captured[0][-1]["role"] == "user"


class TestEgdAndColdBranches:
    """Branch coverage for EGD and COLD optimization loops."""

    def test_egd_sample_strategy_with_entropy_and_transfer(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        init = _make_attack_init(
            mock_model,
            mock_tokenizer,
            transfer_data=[
                (mock_model, mock_tokenizer, [ops.array([1])], ops.array([1])),
            ],
            objective_state=_make_shared_state(
                mock_model,
                perplexity_weight=0.2,
                direction_weight=0.1,
                token_position="suffix",
            ),
        )
        selected = [ops.array([0, 1]), ops.array([1, 2])]
        env_config = EnvironmentConfig(
            system_prompt="system",
            tools=[ToolSchema(name="tool", description="d", parameters={})],
            target=EnvironmentTarget(function="tool"),
            task=EnvironmentTask(content="task"),
            injection_surface="surface",
        )
        soft_embeds = ops.ones((1, 2, D_MODEL))
        ops.eval(soft_embeds)

        with (
            patch("vauban.softprompt._egd.prepare_attack", return_value=init),
            patch("vauban.softprompt._egd._sample_prompt_ids", return_value=selected),
            patch(
                "vauban.softprompt._egd._compute_average_objective_loss",
                return_value=ops.array(1.0),
            ),
            patch(
                "vauban.softprompt._egd._compute_per_prompt_losses",
                return_value=[1.0, 1.0],
            ),
            patch("vauban.softprompt._egd._score_transfer_loss", return_value=0.25),
            patch(
                "vauban.softprompt._egd._evaluate_attack", return_value=(0.5, ["ok"])
            ),
            patch(
                "vauban.softprompt._egd.ops.value_and_grad",
                side_effect=_fake_value_and_grad,
            ),
            patch(
                "vauban.environment.run_agent_loop",
                return_value=EnvironmentResult(
                    reward=0.3,
                    target_called=True,
                    target_args_match=True,
                    turns=[],
                    tool_calls_made=[],
                    injection_payload="payload",
                ),
            ),
        ):
            config = SoftPromptConfig(
                mode="egd",
                n_tokens=2,
                n_steps=2,
                learning_rate=0.1,
                batch_size=2,
                top_k=4,
                prompt_strategy="sample",
                worst_k=1,
                temperature_schedule="linear",
                egd_temperature=0.5,
                entropy_weight=0.5,
                transfer_loss_weight=0.5,
                transfer_rerank_count=2,
                seed=7,
                max_gen_tokens=2,
            )
            result = _egd_attack(
                mock_model,
                mock_tokenizer,
                ["alpha", "beta"],
                config,
                None,
                transfer_models=[("transfer", mock_model, mock_tokenizer)],
                environment_config=env_config,
            )

        assert result.mode == "egd"
        assert result.token_ids is not None
        assert len(result.loss_history) == 2
        assert result.accessibility_score > 0.0
        assert result.eval_responses == ["ok"]

    def test_egd_worst_k_early_stops(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        init = _make_attack_init(
            mock_model,
            mock_tokenizer,
            objective_state=_make_shared_state(mock_model),
        )
        soft_embeds = ops.ones((1, 2, D_MODEL))
        ops.eval(soft_embeds)

        with (
            patch("vauban.softprompt._egd.prepare_attack", return_value=init),
            patch(
                "vauban.softprompt._egd._select_worst_k_prompt_ids",
                return_value=[ops.array([0, 1])],
            ),
            patch(
                "vauban.softprompt._egd._compute_average_objective_loss",
                return_value=ops.array(1.0),
            ),
            patch(
                "vauban.softprompt._egd._compute_per_prompt_losses", return_value=[1.0]
            ),
            patch("vauban.softprompt._egd._evaluate_attack", return_value=(0.0, [""])),
            patch(
                "vauban.softprompt._egd.ops.value_and_grad",
                side_effect=_fake_value_and_grad,
            ),
        ):
            config = SoftPromptConfig(
                mode="egd",
                n_tokens=2,
                n_steps=2,
                learning_rate=0.1,
                prompt_strategy="worst_k",
                worst_k=1,
                patience=1,
                max_gen_tokens=2,
            )
            result = _egd_attack(
                mock_model,
                mock_tokenizer,
                ["alpha"],
                config,
                None,
            )

        assert result.early_stopped is True
        assert len(result.loss_history) == 2

    def test_cold_sample_strategy_and_posthoc_branches(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        init = _make_attack_init(
            mock_model,
            mock_tokenizer,
            transfer_data=[
                (mock_model, mock_tokenizer, [ops.array([1])], ops.array([1])),
            ],
            objective_state=_make_shared_state(
                mock_model,
                perplexity_weight=0.2,
                direction_weight=0.1,
            ),
        )
        selected = [ops.array([0, 1]), ops.array([1, 2])]
        env_config = EnvironmentConfig(
            system_prompt="system",
            tools=[ToolSchema(name="tool", description="d", parameters={})],
            target=EnvironmentTarget(function="tool"),
            task=EnvironmentTask(content="task"),
            injection_surface="surface",
        )

        with (
            patch("vauban.softprompt._cold.prepare_attack", return_value=init),
            patch(
                "vauban.softprompt._cold._build_vocab_mask",
                return_value=ops.ones((D_MODEL,)),
            ),
            patch("vauban.softprompt._cold._resolve_init_ids", return_value=[0, 1]),
            patch("vauban.softprompt._cold._sample_prompt_ids", return_value=selected),
            patch(
                "vauban.softprompt._cold._compute_average_objective_loss",
                return_value=ops.array(1.0),
            ),
            patch(
                "vauban.softprompt._cold._compute_per_prompt_losses",
                return_value=[1.0, 1.0],
            ),
            patch("vauban.softprompt._cold._score_transfer_loss", return_value=0.25),
            patch(
                "vauban.softprompt._cold._evaluate_attack", return_value=(0.5, ["ok"])
            ),
            patch(
                "vauban.softprompt._cold.ops.value_and_grad",
                side_effect=_fake_value_and_grad,
            ),
            patch(
                "vauban.environment.run_agent_loop",
                return_value=EnvironmentResult(
                    reward=0.3,
                    target_called=True,
                    target_args_match=True,
                    turns=[],
                    tool_calls_made=[],
                    injection_payload="payload",
                ),
            ),
        ):
            config = SoftPromptConfig(
                mode="cold",
                n_tokens=2,
                n_steps=2,
                learning_rate=0.1,
                prompt_strategy="sample",
                worst_k=1,
                init_tokens=[1],
                token_constraint="ascii",
                temperature_schedule="linear",
                cold_temperature=0.5,
                cold_noise_scale=0.0,
                entropy_weight=0.5,
                transfer_loss_weight=0.5,
                transfer_rerank_count=2,
                seed=7,
                max_gen_tokens=2,
            )
            result = _cold_attack(
                mock_model,
                mock_tokenizer,
                ["alpha", "beta"],
                config,
                None,
                transfer_models=[("transfer", mock_model, mock_tokenizer)],
                environment_config=env_config,
            )

        assert result.mode == "cold"
        assert result.token_ids is not None
        assert len(result.loss_history) == 2
        assert result.accessibility_score > 0.0
        assert result.eval_responses == ["ok"]

    def test_cold_worst_k_early_stops(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        init = _make_attack_init(
            mock_model,
            mock_tokenizer,
            objective_state=_make_shared_state(mock_model),
        )

        with (
            patch("vauban.softprompt._cold.prepare_attack", return_value=init),
            patch("vauban.softprompt._cold._build_vocab_mask", return_value=None),
            patch("vauban.softprompt._cold._resolve_init_ids", return_value=[0, 1]),
            patch(
                "vauban.softprompt._cold._select_worst_k_prompt_ids",
                return_value=[ops.array([0, 1])],
            ),
            patch(
                "vauban.softprompt._cold._compute_average_objective_loss",
                return_value=ops.array(1.0),
            ),
            patch(
                "vauban.softprompt._cold._compute_per_prompt_losses", return_value=[1.0]
            ),
            patch("vauban.softprompt._cold._evaluate_attack", return_value=(0.0, [""])),
            patch(
                "vauban.softprompt._cold.ops.value_and_grad",
                side_effect=_fake_value_and_grad,
            ),
        ):
            config = SoftPromptConfig(
                mode="cold",
                n_tokens=2,
                n_steps=2,
                learning_rate=0.1,
                prompt_strategy="worst_k",
                worst_k=1,
                patience=1,
                cold_noise_scale=0.0,
                max_gen_tokens=2,
            )
            result = _cold_attack(
                mock_model,
                mock_tokenizer,
                ["alpha"],
                config,
                None,
            )

        assert result.early_stopped is True
        assert len(result.loss_history) == 2
