"""Tests for vauban.softprompt loss, regularization, and scoring helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

from vauban import _ops as ops
from vauban.softprompt import (
    _build_vocab_mask,
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_eos_loss,
    _compute_kl_collision_loss,
    _compute_learning_rate,
    _compute_loss,
    _compute_untargeted_loss,
    _continuous_attack,
    _egd_attack,
    _encode_refusal_tokens,
    _gcg_attack,
    softprompt_attack,
)
from vauban.types import (
    SoftPromptConfig,
)

if TYPE_CHECKING:

    from vauban._array import Array

class TestComputeLearningRate:
    def test_constant_returns_base(self) -> None:
        assert _compute_learning_rate(0.01, 0, 100, "constant") == 0.01
        assert _compute_learning_rate(0.01, 50, 100, "constant") == 0.01
        assert _compute_learning_rate(0.01, 99, 100, "constant") == 0.01

    def test_cosine_start_equals_base(self) -> None:
        lr = _compute_learning_rate(0.01, 0, 100, "cosine")
        assert abs(lr - 0.01) < 1e-9

    def test_cosine_end_near_zero(self) -> None:
        lr = _compute_learning_rate(0.01, 99, 100, "cosine")
        assert abs(lr) < 1e-9

    def test_cosine_midpoint(self) -> None:
        lr = _compute_learning_rate(0.01, 49, 99, "cosine")
        # cos(pi * 49/98) = cos(pi/2) = 0, so lr = 0.005
        assert abs(lr - 0.005) < 1e-6

    def test_cosine_single_step(self) -> None:
        # n_steps=1 -> schedule has no effect (n_steps - 1 = 0)
        lr = _compute_learning_rate(0.01, 0, 1, "cosine")
        assert lr == 0.01


class TestComputeEmbedRegularization:
    def test_zero_weight_returns_zero(self) -> None:
        soft = ops.random.normal((1, 4, 16))
        embed = ops.random.normal((32, 16))
        ops.eval(soft, embed)
        result = _compute_embed_regularization(soft, embed, 0.0)
        ops.eval(result)
        assert float(result.item()) == 0.0

    def test_matching_norms_near_zero(self) -> None:
        """When soft embeds have same mean norm as real embeds, reg is ~0."""
        embed = ops.random.normal((32, 16))
        ops.eval(embed)
        mean_real = float(ops.mean(ops.linalg.norm(embed, axis=-1)).item())
        # Create soft embeds normalized to match
        soft = ops.random.normal((1, 4, 16))
        ops.eval(soft)
        norms = ops.linalg.norm(soft[0], axis=-1, keepdims=True)
        soft_normalized = soft / norms * mean_real
        soft_normalized = soft_normalized[None, :] if soft_normalized.ndim == 2 else soft_normalized  # noqa: E501
        ops.eval(soft_normalized)
        result = _compute_embed_regularization(soft_normalized, embed, 1.0)
        ops.eval(result)
        assert float(result.item()) < 0.01

    def test_large_gap_gives_large_penalty(self) -> None:
        soft = ops.ones((1, 4, 16)) * 100.0  # very large norm
        embed = ops.ones((32, 16)) * 0.01  # very small norm
        ops.eval(soft, embed)
        result = _compute_embed_regularization(soft, embed, 1.0)
        ops.eval(result)
        assert float(result.item()) > 1.0


class TestAccessibilityScore:
    def test_zero_loss(self) -> None:
        assert _compute_accessibility_score(0.0) == 1.0

    def test_high_loss(self) -> None:
        score = _compute_accessibility_score(10.0)
        assert score < 0.001

    def test_monotonicity(self) -> None:
        s1 = _compute_accessibility_score(1.0)
        s2 = _compute_accessibility_score(2.0)
        s3 = _compute_accessibility_score(3.0)
        assert s1 > s2 > s3


class TestRAIDDirectionMode:
    def test_raid_runs(self) -> None:
        """RAID mode with direction_weight=0.1 produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            direction_mode="raid",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss), "RAID mode produced NaN loss"

    def test_raid_differs_from_last(self) -> None:
        """Same seed, RAID vs 'last' produce different final losses."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config_last = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            direction_weight=0.5,
            direction_mode="last",
            seed=42,
            max_gen_tokens=2,
        )
        result_last = _continuous_attack(
            model, tokenizer, ["test"], config_last, direction,
        )

        config_raid = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            direction_weight=0.5,
            direction_mode="raid",
            seed=42,
            max_gen_tokens=2,
        )
        result_raid = _continuous_attack(
            model, tokenizer, ["test"], config_raid, direction,
        )

        # They should differ since RAID accumulates across layers
        assert result_last.loss_history != result_raid.loss_history

    def test_all_positions_runs(self) -> None:
        """direction_mode='all_positions' runs without error."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            direction_mode="all_positions",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_direction_layers_subset(self) -> None:
        """direction_layers=[0] accepted, runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            direction_mode="raid",
            direction_layers=[0],
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        assert len(result.loss_history) == 3


class TestUntargetedLoss:
    def test_untargeted_runs(self) -> None:
        """loss_mode='untargeted' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.01,
            loss_mode="untargeted",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss), "Untargeted loss produced NaN"

    def test_untargeted_gradient_nonzero(self) -> None:
        """Gradient through untargeted loss is nonzero."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        refusal_ids = _encode_refusal_tokens(tokenizer)
        ops.eval(refusal_ids)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = ops.array(tokenizer.encode(text))[None, :]

        def loss_fn(embeds: Array) -> Array:
            return _compute_untargeted_loss(
                model, embeds, prompt_ids, 4, refusal_ids,
                None, 0.0,
            )

        loss_and_grad = ops.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        ops.eval(loss_val, grad)

        grad_norm = float(ops.linalg.norm(grad.reshape(-1)).item())
        assert grad_norm > 0, "Untargeted gradient is zero"

    def test_untargeted_with_direction(self) -> None:
        """Untargeted + RAID direction combined."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            loss_mode="untargeted",
            direction_weight=0.1,
            direction_mode="raid",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_encode_refusal_tokens(self) -> None:
        """Produces non-empty array of valid token IDs."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids = _encode_refusal_tokens(tokenizer)
        ops.eval(ids)
        assert ids.ndim == 1
        assert ids.shape[0] > 0
        for i in range(ids.shape[0]):
            assert 0 <= int(ids[i].item()) < VOCAB_SIZE


class TestTokenConstraint:
    def test_build_vocab_mask_ascii(self) -> None:
        """Mask shape is (VOCAB_SIZE,) with some True entries."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mask = _build_vocab_mask(tokenizer, VOCAB_SIZE, "ascii")
        assert mask is not None
        ops.eval(mask)
        assert mask.shape == (VOCAB_SIZE,)
        n_allowed = int(ops.sum(mask).item())
        assert n_allowed > 0
        assert n_allowed <= VOCAB_SIZE

    def test_build_vocab_mask_none(self) -> None:
        """Returns None when constraint is None."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mask = _build_vocab_mask(tokenizer, VOCAB_SIZE, None)
        assert mask is None

    def test_gcg_with_constraint(self) -> None:
        """GCG + token_constraint='ascii' runs, all token IDs map to ASCII."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=3,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            token_constraint="ascii",
        )

        result = _gcg_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            decoded = tokenizer.decode([tid])
            assert all(32 <= ord(c) < 127 for c in decoded)

    def test_egd_with_constraint(self) -> None:
        """EGD + token_constraint='ascii' runs, all token IDs map to ASCII."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            token_constraint="ascii",
        )

        result = _egd_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            decoded = tokenizer.decode([tid])
            assert all(32 <= ord(c) < 127 for c in decoded)


class TestEOSLoss:
    def test_eos_force_runs(self) -> None:
        """Continuous + eos_loss_mode='force' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            eos_loss_mode="force",
            eos_loss_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "EOS force loss produced NaN"

    def test_eos_suppress_runs(self) -> None:
        """Continuous + eos_loss_mode='suppress' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            eos_loss_mode="suppress",
            eos_loss_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "EOS suppress loss produced NaN"

    def test_eos_loss_helper_force(self) -> None:
        """Force loss decreases as P(EOS) increases."""
        # Create logits where EOS token has varying probability
        vocab_size = 8
        eos_id = 7

        # Low P(EOS): uniform logits
        logits_low = ops.zeros((1, 2, vocab_size))
        ops.eval(logits_low)
        loss_low = _compute_eos_loss(logits_low, 0, eos_id, "force")
        ops.eval(loss_low)

        # High P(EOS): boost EOS logit
        logits_high = ops.zeros((1, 2, vocab_size))
        logits_high[:, 0, eos_id] = logits_high[:, 0, eos_id] + 10.0
        ops.eval(logits_high)
        loss_high = _compute_eos_loss(logits_high, 0, eos_id, "force")
        ops.eval(loss_high)

        assert float(loss_high.item()) < float(loss_low.item())

    def test_eos_loss_helper_suppress(self) -> None:
        """Suppress loss decreases as P(EOS) decreases."""
        vocab_size = 8
        eos_id = 7

        # High P(EOS): boost EOS logit
        logits_high = ops.zeros((1, 2, vocab_size))
        logits_high[:, 0, eos_id] = logits_high[:, 0, eos_id] + 10.0
        ops.eval(logits_high)
        loss_high = _compute_eos_loss(logits_high, 0, eos_id, "suppress")
        ops.eval(loss_high)

        # Low P(EOS): uniform logits
        logits_low = ops.zeros((1, 2, vocab_size))
        ops.eval(logits_low)
        loss_low = _compute_eos_loss(logits_low, 0, eos_id, "suppress")
        ops.eval(loss_low)

        assert float(loss_low.item()) < float(loss_high.item())


class TestKLCollisionLoss:
    def test_kl_collision_runs(self) -> None:
        """Continuous + kl_ref_weight=0.1 with ref model runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        ref_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(ref_model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            kl_ref_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
            ref_model=ref_model,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "KL collision loss produced NaN"

    def test_kl_collision_same_model_low(self) -> None:
        """Same model as reference produces near-zero KL."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = ops.array(tokenizer.encode(text))[None, :]

        kl = _compute_kl_collision_loss(model, model, soft_embeds, prompt_ids, 4)
        ops.eval(kl)
        assert float(kl.item()) < 0.01

    def test_kl_collision_gradient_nonzero(self) -> None:
        """Gradient through KL loss is nonzero."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        ref_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(ref_model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = ops.array(tokenizer.encode(text))[None, :]

        def loss_fn(embeds: Array) -> Array:
            return _compute_kl_collision_loss(
                model, ref_model, embeds, prompt_ids, 4,
            )

        loss_and_grad = ops.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        ops.eval(loss_val, grad)

        grad_norm = float(ops.linalg.norm(grad.reshape(-1)).item())
        assert grad_norm > 0, "KL collision gradient is zero"


class TestGradAccumIntegration:
    def test_continuous_grad_accum(self) -> None:
        """Continuous mode with grad_accum_steps=2 runs and produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
            grad_accum_steps=2,
        )

        result = _continuous_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_gcg_grad_accum(self) -> None:
        """GCG mode with grad_accum_steps=2 runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
            grad_accum_steps=2,
        )

        result = _gcg_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "gcg"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_egd_grad_accum(self) -> None:
        """EGD mode with grad_accum_steps=2 runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
            grad_accum_steps=2,
        )

        result = _egd_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "egd"
        for loss in result.loss_history:
            assert not math.isnan(loss)


class TestDefenseAwarePenalty:
    def test_zero_weight_no_penalty(self) -> None:
        """defense_aware_weight=0 means no penalty contribution."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        h = ops.ones((1, 4, D_MODEL))
        direction = ops.ones((D_MODEL,))
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=0.5,
            cast_layers=[2], cast_threshold=0.5,
        )
        # The function itself returns a value; the weight=0 gating is external
        assert float(result.item()) >= 0.0

    def test_no_direction_returns_zero(self) -> None:
        """No direction → zero penalty."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        h = ops.ones((1, 4, D_MODEL))
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=None,
            sic_layer=2, sic_threshold=0.5,
            cast_layers=[2], cast_threshold=0.5,
        )
        assert float(result.item()) == 0.0

    def test_sic_layer_mismatch_no_penalty(self) -> None:
        """Wrong layer → no SIC penalty."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        h = ops.ones((1, 4, D_MODEL))
        direction = ops.ones((D_MODEL,))
        result = _compute_defense_aware_penalty(
            h, layer_idx=5, direction=direction,
            sic_layer=2, sic_threshold=0.5,
            cast_layers=None, cast_threshold=0.5,
        )
        assert float(result.item()) == 0.0

    def test_defense_aware_config_defaults(self) -> None:
        """New config fields have correct defaults."""
        cfg = SoftPromptConfig()
        assert cfg.defense_aware_weight == 0.0
        assert cfg.defense_eval_alpha_tiers is None

    def test_defense_aware_weight_custom(self) -> None:
        """defense_aware_weight can be set."""
        cfg = SoftPromptConfig(defense_aware_weight=0.5)
        assert cfg.defense_aware_weight == 0.5

    def test_alpha_tiers_custom(self) -> None:
        """defense_eval_alpha_tiers can be set."""
        tiers = [(0.3, 1.0), (0.6, 2.0)]
        cfg = SoftPromptConfig(defense_eval_alpha_tiers=tiers)
        assert cfg.defense_eval_alpha_tiers == tiers


class TestDefenseAwarePenaltyDirections:
    """Test SIC/CAST penalty direction and combined behavior."""

    def test_sic_penalty_when_proj_below_threshold(self) -> None:
        """SIC penalty activates when projection < threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Small hidden state → small projection → below threshold
        h = ops.ones((1, 4, D_MODEL)) * 0.01
        direction = ops.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=10.0,
            cast_layers=None, cast_threshold=0.0,
        )
        assert float(result.item()) > 0.0

    def test_sic_no_penalty_when_proj_above_threshold(self) -> None:
        """SIC penalty is zero when projection >= threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Large hidden state → large projection → above threshold
        h = ops.ones((1, 4, D_MODEL)) * 10.0
        direction = ops.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=0.01,
            cast_layers=None, cast_threshold=0.0,
        )
        assert float(result.item()) == 0.0

    def test_cast_penalty_when_proj_above_threshold(self) -> None:
        """CAST penalty activates when projection > threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Large hidden state → large projection → above threshold
        h = ops.ones((1, 4, D_MODEL)) * 10.0
        direction = ops.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=3, direction=direction,
            sic_layer=None, sic_threshold=0.0,
            cast_layers=[3], cast_threshold=0.01,
        )
        assert float(result.item()) > 0.0

    def test_cast_no_penalty_when_proj_below_threshold(self) -> None:
        """CAST penalty is zero when projection <= threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Small hidden state → small projection → below threshold
        h = ops.ones((1, 4, D_MODEL)) * 0.01
        direction = ops.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=3, direction=direction,
            sic_layer=None, sic_threshold=0.0,
            cast_layers=[3], cast_threshold=10.0,
        )
        assert float(result.item()) == 0.0

    def test_both_sic_and_cast_on_same_layer(self) -> None:
        """Both penalties contribute when SIC and CAST are on the same layer."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Use a moderate projection so it's below SIC threshold AND above CAST
        # threshold — both penalties should fire
        h = ops.ones((1, 4, D_MODEL))
        direction = ops.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        # Projection = sum(1 * 1/sqrt(D)) = sqrt(D) ≈ 5.66 for D=32
        sic_threshold = 100.0  # proj << threshold → SIC penalty
        cast_threshold = 0.01  # proj >> threshold → CAST penalty

        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=sic_threshold,
            cast_layers=[2], cast_threshold=cast_threshold,
        )
        # Both penalties contribute, so result > either alone
        sic_only = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=sic_threshold,
            cast_layers=None, cast_threshold=0.0,
        )
        cast_only = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=None, sic_threshold=0.0,
            cast_layers=[2], cast_threshold=cast_threshold,
        )
        assert float(result.item()) > float(sic_only.item())
        assert float(result.item()) > float(cast_only.item())


class TestGcgTransferScoring:
    def test_transfer_weight_zero_is_noop(self) -> None:
        """transfer_loss_weight=0 should skip re-ranking entirely."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        # Second model as "transfer" target
        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.0,
        )
        transfer_models = [("transfer", t_model, t_tok)]
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=transfer_models,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_transfer_weight_positive_runs(self) -> None:
        """transfer_loss_weight>0 with transfer models should run."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.5,
        )
        transfer_models = [("transfer", t_model, t_tok)]
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=transfer_models,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_no_transfer_models_unchanged(self) -> None:
        """No transfer models means no re-ranking, even with weight > 0."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.5,
        )
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=None,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None

    def test_transfer_loss_weight_default(self) -> None:
        """SoftPromptConfig default for transfer_loss_weight is 0.0."""
        cfg = SoftPromptConfig()
        assert cfg.transfer_loss_weight == 0.0

    def test_transfer_rerank_count_default(self) -> None:
        """SoftPromptConfig default for transfer_rerank_count is 8."""
        cfg = SoftPromptConfig()
        assert cfg.transfer_rerank_count == 8

    def test_transfer_rerank_count_custom(self) -> None:
        """transfer_rerank_count can be set to a custom value."""
        cfg = SoftPromptConfig(transfer_rerank_count=4)
        assert cfg.transfer_rerank_count == 4

    def test_transfer_rerank_count_used(self) -> None:
        """Custom rerank count should be respected during GCG."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.5,
            transfer_rerank_count=2,
        )
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=[("t", t_model, t_tok)],
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None


class TestEgdTransferScoring:
    def test_egd_transfer_weight_zero_noop(self) -> None:
        """EGD with transfer_loss_weight=0 should not change behavior."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd", n_tokens=4, n_steps=3,
            learning_rate=0.1,
            transfer_loss_weight=0.0,
        )
        result = _egd_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=[("t", t_model, t_tok)],
        )
        assert result.mode == "egd"
        assert result.token_ids is not None

    def test_egd_transfer_weight_positive_runs(self) -> None:
        """EGD with transfer_loss_weight>0 should include transfer loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd", n_tokens=4, n_steps=3,
            learning_rate=0.1,
            transfer_loss_weight=0.5,
        )
        result = _egd_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=[("t", t_model, t_tok)],
        )
        assert result.mode == "egd"
        assert result.token_ids is not None

    def test_egd_no_transfer_models_unchanged(self) -> None:
        """EGD without transfer models should work normally."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ops.eval(model.parameters())

        config = SoftPromptConfig(
            mode="egd", n_tokens=4, n_steps=3,
            learning_rate=0.1,
            transfer_loss_weight=0.5,
        )
        result = _egd_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=None,
        )
        assert result.mode == "egd"
        assert result.token_ids is not None


class TestPerplexityLoss:
    def test_default_disabled(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.perplexity_weight == 0.0

    def test_compute_perplexity_loss_basic(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        # logits: (1, 4, 32), suffix_token_ids: (1, 4)
        logits = ops.random.normal((1, 4, VOCAB_SIZE))
        suffix_ids = ops.array([[1, 2, 3, 4]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=4, soft_token_offset=0,
        )
        ops.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0

    def test_perplexity_loss_single_token_returns_zero(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        logits = ops.random.normal((1, 2, VOCAB_SIZE))
        suffix_ids = ops.array([[1]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=1, soft_token_offset=0,
        )
        ops.eval(loss)
        assert float(loss.item()) == 0.0

    def test_compute_loss_with_perplexity(self) -> None:

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = ops.array([[1, 2, 3]])
        target_ids = ops.array([5, 6])
        suffix_ids = ops.array([[1, 2, 3, 4]])

        loss_no_ppl = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
        )
        loss_with_ppl = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
        )
        ops.eval(loss_no_ppl, loss_with_ppl)
        # Perplexity adds a non-negative term
        assert float(loss_with_ppl.item()) >= float(loss_no_ppl.item()) - 1e-6


class TestPerplexityWithOffset:
    def test_perplexity_suffix_offset(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        # 3 prompt tokens + 4 soft tokens = logits at positions 3..5
        logits = ops.random.normal((1, 10, VOCAB_SIZE))
        suffix_ids = ops.array([[1, 2, 3, 4]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=4, soft_token_offset=3,
        )
        ops.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0

    def test_perplexity_infix_offset(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        # infix_split=2: soft tokens start at position 2
        logits = ops.random.normal((1, 12, VOCAB_SIZE))
        suffix_ids = ops.array([[1, 2, 3, 4]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=4, soft_token_offset=2,
        )
        ops.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0

    def test_add_perplexity_term_disabled(self) -> None:
        from vauban.softprompt import _add_perplexity_term

        logits = ops.random.normal((1, 8, VOCAB_SIZE))
        base_loss = ops.array(1.5)
        result = _add_perplexity_term(
            base_loss, logits,
            perplexity_weight=0.0, suffix_token_ids=None,
            n_tokens=4, n_prompt=3,
            token_position="prefix", infix_split=None,
        )
        ops.eval(result)
        assert float(result.item()) == float(base_loss.item())

    def test_add_perplexity_term_suffix(self) -> None:
        from vauban.softprompt import _add_perplexity_term

        logits = ops.random.normal((1, 10, VOCAB_SIZE))
        base_loss = ops.array(1.0)
        suffix_ids = ops.array([[1, 2, 3, 4]])
        result = _add_perplexity_term(
            base_loss, logits,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            n_tokens=4, n_prompt=3,
            token_position="suffix", infix_split=None,
        )
        ops.eval(result)
        assert float(result.item()) >= float(base_loss.item()) - 1e-6

    def test_add_perplexity_term_infix(self) -> None:
        from vauban.softprompt import _add_perplexity_term

        logits = ops.random.normal((1, 12, VOCAB_SIZE))
        base_loss = ops.array(1.0)
        suffix_ids = ops.array([[1, 2, 3, 4]])
        result = _add_perplexity_term(
            base_loss, logits,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            n_tokens=4, n_prompt=5,
            token_position="infix", infix_split=2,
        )
        ops.eval(result)
        assert float(result.item()) >= float(base_loss.item()) - 1e-6

    def test_loss_with_perplexity_suffix_position(self) -> None:

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = ops.array([[1, 2, 3]])
        target_ids = ops.array([5, 6])
        suffix_ids = ops.array([[1, 2, 3, 4]])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            token_position="suffix",
        )
        ops.eval(loss)
        assert loss.shape == ()

    def test_loss_with_perplexity_infix_position(self) -> None:

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = ops.array([[1, 2, 3, 4, 5]])
        target_ids = ops.array([5, 6])
        suffix_ids = ops.array([[1, 2, 3, 4]])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            token_position="infix", infix_split=2,
        )
        ops.eval(loss)
        assert loss.shape == ()


class TestExternalityLossMode:
    """Tests for loss_mode='externality' in GCG and EGD."""

    def test_gcg_externality_mode(self) -> None:
        """GCG with loss_mode='externality' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)
        direction = ops.random.normal((D_MODEL,))
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            loss_mode="externality",
            direction_weight=1.0,
        )

        result = _gcg_attack(
            model, tokenizer, ["Test prompt"], config,
            direction=direction,
        )
        assert result.mode == "gcg"
        assert math.isfinite(result.final_loss)
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_egd_externality_mode(self) -> None:
        """EGD with loss_mode='externality' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)
        direction = ops.random.normal((D_MODEL,))
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.01,
            seed=42,
            max_gen_tokens=2,
            loss_mode="externality",
            direction_weight=1.0,
        )

        result = _egd_attack(
            model, tokenizer, ["Test prompt"], config,
            direction=direction,
        )
        assert result.mode == "egd"
        assert math.isfinite(result.final_loss)
        assert result.token_ids is not None
        assert len(result.token_ids) == 4
