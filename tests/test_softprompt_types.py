# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.softprompt dataclasses, defaults, and public API."""

from __future__ import annotations

import pytest

from vauban import _ops as ops
from vauban.softprompt._encoding import _compute_infix_split
from vauban.softprompt._loss import _compute_loss
from vauban.types import (
    DefenseEvalResult,
    SoftPromptConfig,
    SoftPromptResult,
    TransferEvalResult,
)


class TestSoftPromptConfig:
    def test_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.mode == "continuous"
        assert cfg.n_tokens == 16
        assert cfg.n_steps == 200
        assert cfg.learning_rate == 0.01
        assert cfg.init_scale == 0.1
        assert cfg.batch_size == 64
        assert cfg.top_k == 256
        assert cfg.direction_weight == 0.0
        assert cfg.target_prefixes == ["Sure", "Here"]
        assert cfg.max_gen_tokens == 100
        assert cfg.seed is None
        assert cfg.embed_reg_weight == 0.0
        assert cfg.patience == 0
        assert cfg.lr_schedule == "constant"
        assert cfg.n_restarts == 1
        assert cfg.prompt_strategy == "all"
        assert cfg.gan_multiturn is False
        assert cfg.gan_multiturn_max_turns == 10

    def test_frozen(self) -> None:
        cfg = SoftPromptConfig()
        with pytest.raises(AttributeError):
            cfg.n_tokens = 10  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = SoftPromptConfig(
            mode="gcg",
            n_tokens=32,
            n_steps=100,
            learning_rate=0.001,
            batch_size=128,
            top_k=512,
            direction_weight=0.5,
            target_prefixes=["OK"],
            seed=42,
        )
        assert cfg.mode == "gcg"
        assert cfg.n_tokens == 32
        assert cfg.seed == 42


class TestSoftPromptResult:
    def test_construction(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.8,
            final_loss=1.5,
            loss_history=[3.0, 2.0, 1.5],
            n_steps=3,
            n_tokens=16,
            embeddings=ops.zeros((1, 16, 8)),
            token_ids=None,
            token_text=None,
            eval_responses=["response1"],
        )
        assert result.mode == "continuous"
        assert result.success_rate == 0.8
        assert result.final_loss == 1.5
        assert len(result.loss_history) == 3
        assert result.embeddings is not None
        assert result.token_ids is None
        # Default new fields
        assert result.accessibility_score == 0.0
        assert result.per_prompt_losses == []
        assert result.early_stopped is False

    def test_frozen(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.0,
            final_loss=0.0,
            loss_history=[],
            n_steps=0,
            n_tokens=1,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        with pytest.raises(AttributeError):
            result.success_rate = 1.0  # type: ignore[misc]


class TestSoftPromptResultNewFields:
    def test_defaults(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=2.0,
            loss_history=[2.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        assert result.accessibility_score == 0.0
        assert result.per_prompt_losses == []
        assert result.early_stopped is False

    def test_explicit_values(self) -> None:
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.9,
            final_loss=0.5,
            loss_history=[1.0, 0.5],
            n_steps=2,
            n_tokens=8,
            embeddings=None,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            token_text="test",
            eval_responses=["response"],
            accessibility_score=0.6,
            per_prompt_losses=[0.4, 0.6],
            early_stopped=True,
        )
        assert result.accessibility_score == 0.6
        assert result.per_prompt_losses == [0.4, 0.6]
        assert result.early_stopped is True


class TestNewConfigDefaults:
    def test_new_fields_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.direction_mode == "last"
        assert cfg.direction_layers is None
        assert cfg.loss_mode == "targeted"
        assert cfg.egd_temperature == 1.0

    def test_egd_mode_accepted(self) -> None:
        cfg = SoftPromptConfig(mode="egd")
        assert cfg.mode == "egd"

    def test_custom_direction_mode(self) -> None:
        cfg = SoftPromptConfig(direction_mode="raid")
        assert cfg.direction_mode == "raid"

    def test_custom_direction_layers(self) -> None:
        cfg = SoftPromptConfig(direction_layers=[0, 1])
        assert cfg.direction_layers == [0, 1]


class TestTransferEvalResult:
    def test_construction(self) -> None:
        result = TransferEvalResult(
            model_id="test-model",
            success_rate=0.5,
            eval_responses=["resp1", "resp2"],
        )
        assert result.model_id == "test-model"
        assert result.success_rate == 0.5
        assert len(result.eval_responses) == 2

    def test_frozen(self) -> None:
        result = TransferEvalResult(
            model_id="test", success_rate=0.0, eval_responses=[],
        )
        with pytest.raises(AttributeError):
            result.success_rate = 1.0  # type: ignore[misc]


class TestSoftPromptResultTransfer:
    def test_default_empty(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=2.0,
            loss_history=[2.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        assert result.transfer_results == []

    def test_with_transfer_results(self) -> None:
        tr = TransferEvalResult(
            model_id="other-model",
            success_rate=0.3,
            eval_responses=["r1"],
        )
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.9,
            final_loss=0.5,
            loss_history=[0.5],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=[1, 2, 3, 4],
            token_text="test",
            eval_responses=["resp"],
            transfer_results=[tr],
        )
        assert len(result.transfer_results) == 1
        assert result.transfer_results[0].model_id == "other-model"


class TestSoftPromptSerialization:
    def test_softprompt_to_dict_includes_transfer(self) -> None:
        from vauban._serializers import _softprompt_to_dict

        tr = TransferEvalResult(
            model_id="m1", success_rate=0.4, eval_responses=["r"],
        )
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.8,
            final_loss=1.0,
            loss_history=[1.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=[1, 2, 3, 4],
            token_text="test",
            eval_responses=["resp"],
            transfer_results=[tr],
        )
        d = _softprompt_to_dict(result)
        assert "transfer_results" in d
        trs = d["transfer_results"]
        assert isinstance(trs, list)
        assert len(trs) == 1
        assert trs[0]["model_id"] == "m1"  # type: ignore[index]
        assert trs[0]["success_rate"] == 0.4  # type: ignore[index]

    def test_softprompt_to_dict_empty_transfer(self) -> None:
        from vauban._serializers import _softprompt_to_dict

        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=2.0,
            loss_history=[2.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        d = _softprompt_to_dict(result)
        assert d["transfer_results"] == []


class TestNewConfigDefaults2:
    def test_constraint_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.token_constraint is None

    def test_eos_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.eos_loss_mode == "none"
        assert cfg.eos_loss_weight == 0.0

    def test_kl_ref_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.kl_ref_weight == 0.0


class TestNewConfigDefaults3:
    def test_worst_k_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.worst_k == 5

    def test_grad_accum_steps_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.grad_accum_steps == 1

    def test_transfer_models_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.transfer_models == []

    def test_custom_worst_k(self) -> None:
        cfg = SoftPromptConfig(worst_k=10)
        assert cfg.worst_k == 10

    def test_custom_grad_accum_steps(self) -> None:
        cfg = SoftPromptConfig(grad_accum_steps=4)
        assert cfg.grad_accum_steps == 4

    def test_custom_transfer_models(self) -> None:
        cfg = SoftPromptConfig(transfer_models=["model-a"])
        assert cfg.transfer_models == ["model-a"]


class TestNewConfigDefaults4:
    """Tests for target_repeat_count and system_prompt config defaults."""

    def test_target_repeat_count_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.target_repeat_count == 0

    def test_system_prompt_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.system_prompt is None

    def test_custom_target_repeat_count(self) -> None:
        cfg = SoftPromptConfig(target_repeat_count=50)
        assert cfg.target_repeat_count == 50

    def test_custom_system_prompt(self) -> None:
        cfg = SoftPromptConfig(system_prompt="You are helpful.")
        assert cfg.system_prompt == "You are helpful."


class TestDefenseEvalResult:
    """Tests for DefenseEvalResult dataclass."""

    def test_construction(self) -> None:
        result = DefenseEvalResult(
            sic_blocked=1,
            sic_sanitized=2,
            sic_clean=3,
            sic_bypass_rate=0.83,
            cast_interventions=10,
            cast_refusal_rate=0.5,
            cast_responses=["resp1", "resp2"],
        )
        assert result.sic_blocked == 1
        assert result.cast_refusal_rate == 0.5
        assert len(result.cast_responses) == 2

    def test_to_dict(self) -> None:
        result = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=0,
            sic_clean=1,
            sic_bypass_rate=1.0,
            cast_interventions=5,
            cast_refusal_rate=0.0,
            cast_responses=["ok"],
        )
        d = result.to_dict()
        assert d["sic_bypass_rate"] == 1.0
        assert d["cast_interventions"] == 5

    def test_frozen(self) -> None:
        result = DefenseEvalResult(
            sic_blocked=0, sic_sanitized=0, sic_clean=0,
            sic_bypass_rate=0.0, cast_interventions=0,
            cast_refusal_rate=0.0, cast_responses=[],
        )
        with pytest.raises(AttributeError):
            result.sic_blocked = 1  # type: ignore[misc]


class TestNewConfigDefaults5:
    """Tests for defense_eval config defaults."""

    def test_defense_eval_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval is None

    def test_defense_eval_layer_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval_layer is None

    def test_defense_eval_alpha_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval_alpha == 1.0

    def test_defense_eval_threshold_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval_threshold == 0.0

    def test_custom_defense_eval(self) -> None:
        cfg = SoftPromptConfig(defense_eval="both")
        assert cfg.defense_eval == "both"

    def test_softprompt_result_defense_eval_default(self) -> None:
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        assert result.defense_eval is None

    def test_softprompt_result_to_dict_defense_eval(self) -> None:
        de = DefenseEvalResult(
            sic_blocked=1, sic_sanitized=0, sic_clean=0,
            sic_bypass_rate=0.0, cast_interventions=5,
            cast_refusal_rate=1.0, cast_responses=["refused"],
        )
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
            defense_eval=de,
        )
        d = result.to_dict()
        assert d["defense_eval"] is not None
        assert d["defense_eval"]["sic_blocked"] == 1  # type: ignore[index]


class TestSoftPromptConfigNewFields:
    def test_prompt_pool_size_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.prompt_pool_size is None

    def test_beam_width_default_one(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.beam_width == 1

    def test_prompt_pool_size_custom(self) -> None:
        cfg = SoftPromptConfig(prompt_pool_size=200)
        assert cfg.prompt_pool_size == 200

    def test_beam_width_custom(self) -> None:
        cfg = SoftPromptConfig(beam_width=4)
        assert cfg.beam_width == 4


class TestPublicApiSurface:
    """Explicit checks for the reduced package-level public surface."""

    def test_high_level_exports_only(self) -> None:
        import vauban.softprompt as softprompt

        assert softprompt.__all__ == [
            "evaluate_against_defenses",
            "evaluate_against_defenses_multiturn",
            "gan_loop",
            "largo_loop",
            "paraphrase_prompts",
            "softprompt_attack",
        ]
        assert callable(softprompt.softprompt_attack)
        assert callable(softprompt.gan_loop)
        assert callable(softprompt.largo_loop)
        assert callable(softprompt.paraphrase_prompts)
        assert not hasattr(softprompt, "_compute_loss")
        assert not hasattr(softprompt, "_compute_infix_split")

    def test_private_helper_facades_still_exist(self) -> None:
        from vauban.softprompt._utils import _encode_targets, _resolve_injection_ids

        assert callable(_compute_infix_split)
        assert callable(_compute_loss)
        assert callable(_encode_targets)
        assert callable(_resolve_injection_ids)
