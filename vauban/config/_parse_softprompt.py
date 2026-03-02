"""Parse the [softprompt] section of a TOML config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.config._parse_softprompt_context import _parse_softprompt_context
from vauban.config._parse_softprompt_core import _parse_softprompt_core
from vauban.config._parse_softprompt_defense import _parse_softprompt_defense
from vauban.config._parse_softprompt_gan import _parse_softprompt_gan
from vauban.config._parse_softprompt_loss import _parse_softprompt_loss
from vauban.types import SoftPromptConfig

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.config._types import TomlDict


def _parse_softprompt(
    raw: TomlDict,
    base_dir: Path | None = None,
) -> SoftPromptConfig | None:
    """Parse the optional [softprompt] section into a SoftPromptConfig."""
    sec = raw.get("softprompt")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[softprompt] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)
    sec_table: TomlDict = {str(key): value for key, value in sec.items()}

    core = _parse_softprompt_core(sec_table)
    loss = _parse_softprompt_loss(sec_table, base_dir)
    defense = _parse_softprompt_defense(sec_table)
    gan = _parse_softprompt_gan(sec_table)
    context = _parse_softprompt_context(sec_table)

    if loss.kl_ref_weight > 0.0 and loss.ref_model is None:
        msg = (
            "[softprompt].kl_ref_weight > 0 requires"
            " [softprompt].ref_model to be set"
        )
        raise ValueError(msg)

    if (
        context.injection_context_template is not None
        and "{payload}" not in context.injection_context_template
    ):
        msg = (
            "[softprompt].injection_context_template must contain"
            " '{payload}' placeholder"
        )
        raise ValueError(msg)

    has_injection = (
        context.injection_context is not None
        or context.injection_context_template is not None
    )
    if has_injection and core.mode == "continuous":
        msg = (
            "[softprompt].injection_context requires mode 'gcg' or 'egd'"
            " (discrete tokens); continuous mode produces soft embeddings"
            " that cannot represent wrapped context"
        )
        raise ValueError(msg)

    if has_injection and gan.gan_multiturn:
        msg = (
            "[softprompt].injection_context cannot be combined with"
            " gan_multiturn; injection context wrapping and conversation"
            " history encoding are mutually exclusive"
        )
        raise ValueError(msg)

    if context.token_position == "infix" and core.mode == "continuous":
        msg = (
            "[softprompt].token_position = 'infix' requires mode 'gcg' or"
            " 'egd' (discrete tokens); continuous mode cannot resolve infix"
            " split positions"
        )
        raise ValueError(msg)

    if (
        loss.loss_mode == "externality"
        and loss.externality_target is None
    ):
        msg = (
            "[softprompt].externality_target is required when loss_mode ="
            " 'externality'"
        )
        raise ValueError(msg)

    if loss.largo_reflection_rounds > 0 and gan.gan_rounds > 0:
        msg = (
            "[softprompt].largo_reflection_rounds and gan_rounds are"
            " mutually exclusive — use one or the other"
        )
        raise ValueError(msg)

    if loss.largo_reflection_rounds > 0 and core.mode != "continuous":
        msg = (
            "[softprompt].largo_reflection_rounds requires mode"
            " 'continuous' (continuous embedding optimization)"
        )
        raise ValueError(msg)

    alpha_tiers = defense.defense_eval_alpha_tiers
    if alpha_tiers is not None:
        for i in range(1, len(alpha_tiers)):
            if alpha_tiers[i][0] < alpha_tiers[i - 1][0]:
                msg = (
                    "[softprompt].defense_eval_alpha_tiers must be sorted by"
                    " ascending threshold"
                )
                raise ValueError(msg)

    return SoftPromptConfig(
        mode=core.mode,
        n_tokens=core.n_tokens,
        n_steps=core.n_steps,
        learning_rate=core.learning_rate,
        init_scale=core.init_scale,
        batch_size=core.batch_size,
        top_k=core.top_k,
        direction_weight=loss.direction_weight,
        target_prefixes=core.target_prefixes,
        max_gen_tokens=core.max_gen_tokens,
        seed=core.seed,
        embed_reg_weight=loss.embed_reg_weight,
        patience=core.patience,
        lr_schedule=core.lr_schedule,
        n_restarts=core.n_restarts,
        prompt_strategy=core.prompt_strategy,
        direction_mode=core.direction_mode,
        direction_layers=core.direction_layers,
        loss_mode=loss.loss_mode,
        egd_temperature=loss.egd_temperature,
        token_constraint=loss.token_constraint,
        eos_loss_mode=loss.eos_loss_mode,
        eos_loss_weight=loss.eos_loss_weight,
        kl_ref_weight=loss.kl_ref_weight,
        ref_model=loss.ref_model,
        worst_k=loss.worst_k,
        grad_accum_steps=loss.grad_accum_steps,
        transfer_models=context.transfer_models,
        target_repeat_count=loss.target_repeat_count,
        system_prompt=context.system_prompt,
        defense_eval=defense.defense_eval,
        defense_eval_layer=defense.defense_eval_layer,
        defense_eval_alpha=defense.defense_eval_alpha,
        defense_eval_threshold=defense.defense_eval_threshold,
        defense_eval_sic_threshold=defense.defense_eval_sic_threshold,
        defense_eval_sic_mode=defense.defense_eval_sic_mode,
        defense_eval_sic_max_iterations=(
            defense.defense_eval_sic_max_iterations
        ),
        defense_eval_cast_layers=defense.defense_eval_cast_layers,
        gan_rounds=gan.gan_rounds,
        gan_step_multiplier=gan.gan_step_multiplier,
        gan_direction_escalation=gan.gan_direction_escalation,
        gan_token_escalation=gan.gan_token_escalation,
        gan_defense_escalation=gan.gan_defense_escalation,
        gan_defense_alpha_multiplier=gan.gan_defense_alpha_multiplier,
        gan_defense_threshold_escalation=gan.gan_defense_threshold_escalation,
        gan_defense_sic_iteration_escalation=(
            gan.gan_defense_sic_iteration_escalation
        ),
        gan_multiturn=gan.gan_multiturn,
        gan_multiturn_max_turns=gan.gan_multiturn_max_turns,
        prompt_pool_size=gan.prompt_pool_size,
        beam_width=gan.beam_width,
        defense_aware_weight=loss.defense_aware_weight,
        transfer_loss_weight=loss.transfer_loss_weight,
        transfer_rerank_count=loss.transfer_rerank_count,
        defense_eval_alpha_tiers=alpha_tiers,
        init_tokens=core.init_tokens,
        injection_context=context.injection_context,
        injection_context_template=context.injection_context_template,
        perplexity_weight=loss.perplexity_weight,
        token_position=context.token_position,
        paraphrase_strategies=context.paraphrase_strategies,
        externality_target=loss.externality_target,
        cold_temperature=loss.cold_temperature,
        cold_noise_scale=loss.cold_noise_scale,
        svf_boundary_path=loss.svf_boundary_path,
        largo_reflection_rounds=loss.largo_reflection_rounds,
        largo_max_reflection_tokens=loss.largo_max_reflection_tokens,
        largo_objective=loss.largo_objective,
        largo_embed_warmstart=loss.largo_embed_warmstart,
    )
