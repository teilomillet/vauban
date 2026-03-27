# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.softprompt GAN orchestration and escalation behavior."""

from __future__ import annotations

import pytest
from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

from vauban.types import (
    DefenseEvalResult,
    GanRoundResult,
    SoftPromptConfig,
    SoftPromptResult,
    TransferEvalResult,
)


class TestGanRoundResult:
    """Tests for GanRoundResult dataclass."""

    def test_gan_round_result_construction(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        de = DefenseEvalResult(
            sic_blocked=0, sic_sanitized=1, sic_clean=0,
            sic_bypass_rate=1.0, cast_interventions=0,
            cast_refusal_rate=0.0, cast_responses=["ok"],
        )
        rr = GanRoundResult(
            round_index=0,
            attack_result=attack,
            defense_result=de,
            attacker_won=True,
            config_snapshot={"n_tokens": 4, "n_steps": 10},
        )
        assert rr.round_index == 0
        assert rr.attacker_won is True
        assert rr.defense_result is not None

    def test_gan_round_result_to_dict(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        rr = GanRoundResult(
            round_index=1,
            attack_result=attack,
            defense_result=None,
            attacker_won=False,
            config_snapshot={"n_tokens": 8},
        )
        d = rr.to_dict()
        assert d["round_index"] == 1
        assert d["attacker_won"] is False
        assert d["defense_result"] is None

    def test_gan_round_result_frozen(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={},
        )
        with pytest.raises(AttributeError):
            rr.round_index = 5  # type: ignore[misc]


class TestGanConfigDefaults:
    """Tests for GAN-related config defaults."""

    def test_gan_rounds_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_rounds == 0

    def test_gan_step_multiplier_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_step_multiplier == 1.5

    def test_gan_direction_escalation_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_direction_escalation == 0.25

    def test_gan_token_escalation_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_token_escalation == 4

    def test_init_tokens_default(self) -> None:
        c = SoftPromptConfig()
        assert c.init_tokens is None

    def test_defense_eval_sic_mode_default(self) -> None:
        c = SoftPromptConfig()
        assert c.defense_eval_sic_mode == "direction"

    def test_defense_eval_sic_max_iterations_default(self) -> None:
        c = SoftPromptConfig()
        assert c.defense_eval_sic_max_iterations == 3

    def test_defense_eval_cast_layers_default(self) -> None:
        c = SoftPromptConfig()
        assert c.defense_eval_cast_layers is None


class TestBuildAdvPrompts:
    """Tests for _build_adv_prompts position-aware prompt construction."""

    def test_suffix_position_appends(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        result = _build_adv_prompts(["Hello world"], "TOKENS", "suffix")
        assert result == ["Hello world TOKENS"]

    def test_prefix_position_appends(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        result = _build_adv_prompts(["Hello world"], "TOKENS", "prefix")
        assert result == ["Hello world TOKENS"]

    def test_infix_replaces_marker(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        prompts = ["Write a story where {suffix} a character travels"]
        result = _build_adv_prompts(prompts, "TOKENS", "infix")
        assert result == ["Write a story where TOKENS a character travels"]

    def test_infix_no_marker_falls_back(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        result = _build_adv_prompts(["No marker here"], "TOKENS", "infix")
        assert result == ["No marker here TOKENS"]

    def test_none_token_text_returns_copy(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        prompts = ["Hello"]
        result = _build_adv_prompts(prompts, None, "infix")
        assert result == ["Hello"]
        assert result is not prompts

    def test_infix_multiple_prompts(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        prompts = [
            "Story where {suffix} character travels",
            "Document how {suffix} technique is used",
        ]
        result = _build_adv_prompts(prompts, "ADV", "infix")
        assert result == [
            "Story where ADV character travels",
            "Document how ADV technique is used",
        ]


    def test_gan_history_default(self) -> None:
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        assert result.gan_history == []

    def test_gan_history_to_dict(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={"n_tokens": 4},
        )
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
            gan_history=[rr],
        )
        d = result.to_dict()
        assert len(d["gan_history"]) == 1  # type: ignore[arg-type]


class TestMultiturnConfig:
    """Tests for multi-turn GAN config fields."""

    def test_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.gan_multiturn is False
        assert cfg.gan_multiturn_max_turns == 10

    def test_custom(self) -> None:
        cfg = SoftPromptConfig(gan_multiturn=True, gan_multiturn_max_turns=5)
        assert cfg.gan_multiturn is True
        assert cfg.gan_multiturn_max_turns == 5

    def test_frozen(self) -> None:
        cfg = SoftPromptConfig(gan_multiturn=True)
        with pytest.raises(AttributeError):
            cfg.gan_multiturn = False  # type: ignore[misc]

    def test_continuous_mode_rejected(self) -> None:
        """Multi-turn requires hard tokens — continuous mode is rejected."""
        from vauban.softprompt import gan_loop

        model = MockCausalLM(D_MODEL, VOCAB_SIZE, NUM_LAYERS, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        cfg = SoftPromptConfig(
            mode="continuous",
            gan_rounds=2,
            gan_multiturn=True,
            n_tokens=2,
            n_steps=1,
        )
        with pytest.raises(ValueError, match="hard tokens"):
            gan_loop(model, tok, ["test"], cfg, direction=None)


class TestDefenderEscalationConfig:
    """Tests for GAN defender escalation config defaults."""

    def test_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.gan_defense_escalation is False
        assert cfg.gan_defense_alpha_multiplier == 1.5
        assert cfg.gan_defense_threshold_escalation == 0.5
        assert cfg.gan_defense_sic_iteration_escalation == 1

    def test_custom(self) -> None:
        cfg = SoftPromptConfig(
            gan_defense_escalation=True,
            gan_defense_alpha_multiplier=2.0,
            gan_defense_threshold_escalation=0.3,
            gan_defense_sic_iteration_escalation=2,
        )
        assert cfg.gan_defense_escalation is True
        assert cfg.gan_defense_alpha_multiplier == 2.0
        assert cfg.gan_defense_threshold_escalation == 0.3
        assert cfg.gan_defense_sic_iteration_escalation == 2

    def test_frozen(self) -> None:
        cfg = SoftPromptConfig(gan_defense_escalation=True)
        with pytest.raises(AttributeError):
            cfg.gan_defense_escalation = False  # type: ignore[misc]


class TestEscalateDefense:
    """Tests for _escalate_defense() function."""

    def test_single_escalation(self) -> None:
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
            gan_defense_alpha_multiplier=1.5,
            gan_defense_threshold_escalation=0.5,
            gan_defense_sic_iteration_escalation=1,
        )
        esc = _escalate_defense(cfg)
        assert esc.defense_eval_alpha == 1.5
        assert esc.defense_eval_threshold == -0.5
        assert esc.defense_eval_sic_max_iterations == 4

    def test_cumulative_escalation(self) -> None:
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
            gan_defense_alpha_multiplier=1.5,
            gan_defense_threshold_escalation=0.5,
            gan_defense_sic_iteration_escalation=1,
        )
        esc1 = _escalate_defense(cfg)
        esc2 = _escalate_defense(esc1)
        assert esc2.defense_eval_alpha == pytest.approx(2.25)
        assert esc2.defense_eval_threshold == pytest.approx(-1.0)
        assert esc2.defense_eval_sic_max_iterations == 5

    def test_preserves_other_fields(self) -> None:
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            mode="gcg",
            n_tokens=8,
            n_steps=100,
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
        )
        esc = _escalate_defense(cfg)
        assert esc.mode == "gcg"
        assert esc.n_tokens == 8
        assert esc.n_steps == 100


class TestGanRoundTransferResults:
    def test_gan_round_result_transfer_results_default(self) -> None:
        """GanRoundResult.transfer_results defaults to empty list."""
        from vauban.types import GanRoundResult

        # Create a minimal attack result
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2, 3, 4],
            token_text="test", eval_responses=["resp"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={},
        )
        assert rr.transfer_results == []

    def test_gan_round_result_to_dict_includes_transfer(self) -> None:
        """to_dict should include transfer_results."""
        from vauban.types import GanRoundResult

        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2, 3, 4],
            token_text="test", eval_responses=["resp"],
        )
        tr = TransferEvalResult(
            model_id="test-model", success_rate=0.8,
            eval_responses=["ok"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={}, transfer_results=[tr],
        )
        d = rr.to_dict()
        assert "transfer_results" in d
        assert len(d["transfer_results"]) == 1  # type: ignore[arg-type]


class TestEscalateDefenseAlphaTiers:
    def test_escalate_scales_existing_tiers(self) -> None:
        """Existing tiers should have alphas scaled by multiplier."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.5,
            defense_eval_alpha_tiers=[(0.3, 1.0), (0.6, 2.0)],
            gan_defense_alpha_multiplier=2.0,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is not None
        assert len(new.defense_eval_alpha_tiers) == 2
        assert new.defense_eval_alpha_tiers[0] == (0.3, 2.0)
        assert new.defense_eval_alpha_tiers[1] == (0.6, 4.0)

    def test_escalate_auto_generates_tiers_from_flat(self) -> None:
        """Without tiers, escalation auto-generates 3 TRYLOCK-style tiers."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=2.0,
            defense_eval_threshold=1.0,
            defense_eval_alpha_tiers=None,
            gan_defense_alpha_multiplier=1.5,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is not None
        assert len(new.defense_eval_alpha_tiers) == 3
        # Thresholds: 0.5, 1.0, 1.5  Alphas: 1.0, 2.0, 3.0
        assert new.defense_eval_alpha_tiers[0] == (0.5, 1.0)
        assert new.defense_eval_alpha_tiers[1] == (1.0, 2.0)
        assert new.defense_eval_alpha_tiers[2] == (1.5, 3.0)

    def test_escalate_no_auto_gen_when_threshold_zero(self) -> None:
        """No auto-generation when threshold=0 (would produce degenerate tiers)."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=2.0,
            defense_eval_threshold=0.0,
            defense_eval_alpha_tiers=None,
            gan_defense_alpha_multiplier=1.5,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is None

    def test_escalate_no_auto_gen_when_alpha_zero(self) -> None:
        """No auto-generation when alpha=0."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=0.0,
            defense_eval_threshold=0.5,
            defense_eval_alpha_tiers=None,
            gan_defense_alpha_multiplier=1.5,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is None


class TestInjectionContextConfigDefaults:
    """Tests for SoftPromptConfig injection context field defaults."""

    def test_injection_context_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.injection_context is None

    def test_injection_context_template_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.injection_context_template is None

    def test_injection_context_custom(self) -> None:
        cfg = SoftPromptConfig(injection_context="tool_output")
        assert cfg.injection_context == "tool_output"

    def test_injection_context_template_custom(self) -> None:
        cfg = SoftPromptConfig(
            injection_context_template="X {payload} Y",
        )
        assert cfg.injection_context_template == "X {payload} Y"
