"""Entry point dispatcher for soft prompt attacks."""

import random
from dataclasses import replace

from vauban import _ops as ops
from vauban._array import Array
from vauban.softprompt._amplecgc import _amplecgc_attack
from vauban.softprompt._cold import _cold_attack
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._defense_eval import evaluate_against_defenses
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._encoding import (
    _resolve_infix_overrides,
    _resolve_injection_ids,
)
from vauban.softprompt._gan import gan_loop
from vauban.softprompt._gcg import _gcg_attack
from vauban.softprompt._paraphrase import paraphrase_prompts
from vauban.types import (
    ApiEvalConfig,
    CausalLM,
    EnvironmentConfig,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
)


def _run_single_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None = None,
    environment_config: EnvironmentConfig | None = None,
    svf_boundary: object | None = None,
) -> SoftPromptResult:
    """Dispatch to the appropriate attack mode.

    Does NOT run defense evaluation or GAN loop — just the raw attack.

    Args:
        model: The causal language model to attack.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration.
        direction: Optional refusal direction for direction-guided mode.
        ref_model: Optional reference model for KL collision loss.
        transfer_models: Optional list of (name, model, tokenizer) for
            multi-model candidate re-ranking during GCG.
        environment_config: Optional environment config for rollout scoring.
    """
    # Expand prompts with paraphrase strategies
    if config.paraphrase_strategies:
        prompts = paraphrase_prompts(prompts, config.paraphrase_strategies)

    injection_ids = _resolve_injection_ids(config, tokenizer, prompts)

    # Resolve infix split positions when token_position is "infix"
    infix_map: dict[int, int] | None = None
    if config.token_position == "infix" and injection_ids is None:
        infix_ids, infix_map = _resolve_infix_overrides(
            tokenizer, prompts, config.system_prompt,
        )
        injection_ids = infix_ids

    if config.mode == "continuous":
        return _continuous_attack(
            model, tokenizer, prompts, config, direction, ref_model,
        )
    if config.mode == "gcg":
        return _gcg_attack(
            model, tokenizer, prompts, config, direction, ref_model,
            all_prompt_ids_override=injection_ids,
            transfer_models=transfer_models,
            infix_map=infix_map,
            environment_config=environment_config,
        )
    if config.mode == "egd":
        return _egd_attack(
            model, tokenizer, prompts, config, direction, ref_model,
            all_prompt_ids_override=injection_ids,
            transfer_models=transfer_models,
            infix_map=infix_map,
            environment_config=environment_config,
        )
    if config.mode == "cold":
        return _cold_attack(
            model, tokenizer, prompts, config, direction, ref_model,
            all_prompt_ids_override=injection_ids,
            transfer_models=transfer_models,
            infix_map=infix_map,
            environment_config=environment_config,
        )
    if config.mode == "amplecgc":
        return _amplecgc_attack(
            model, tokenizer, prompts, config, direction, ref_model,
            all_prompt_ids_override=injection_ids,
            transfer_models=transfer_models,
            infix_map=infix_map,
            environment_config=environment_config,
        )
    msg = (
        f"Unknown soft prompt mode: {config.mode!r},"
        " must be 'continuous', 'gcg', 'egd', 'cold', or 'amplecgc'"
    )
    raise ValueError(msg)


def softprompt_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None = None,
    ref_model: CausalLM | None = None,
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None = None,
    api_eval_config: ApiEvalConfig | None = None,
    environment_config: EnvironmentConfig | None = None,
    svf_boundary: object | None = None,
) -> SoftPromptResult:
    """Run a soft prompt attack against a model.

    Optimizes a learnable prefix in embedding space that steers generation
    away from refusal. Supports continuous (gradient-based), GCG
    (discrete token search), and EGD (exponentiated gradient descent) modes.

    When ``config.gan_rounds > 0``, runs an iterative GAN-style loop where
    each round attacks, evaluates defenses, and escalates attack parameters.

    When ``config.defense_eval`` is set (without GAN), the found suffix is
    tested against SIC and/or CAST defense modules after optimization.

    Args:
        model: The causal language model to attack.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration.
        direction: Optional refusal direction for direction-guided mode.
        ref_model: Optional reference model for KL collision loss.
        transfer_models: Optional list of (name, model, tokenizer) for
            multi-model candidate re-ranking during GCG.
        api_eval_config: Optional API evaluation config for remote
            endpoint testing (passed to GAN loop).
    """
    if config.seed is not None:
        ops.random.seed(config.seed)
        random.seed(config.seed)

    # LARGO reflection loop: continuous + self-reflective decoding
    if config.largo_reflection_rounds > 0:
        from vauban.softprompt._largo import largo_loop

        return largo_loop(
            model, tokenizer, prompts, config, direction, ref_model,
            svf_boundary=svf_boundary,
        )

    # GAN loop: iterative attack-defense rounds
    if config.gan_rounds > 0:
        return gan_loop(
            model, tokenizer, prompts, config, direction, ref_model,
            transfer_models=transfer_models,
            api_eval_config=api_eval_config,
            environment_config=environment_config,
        )

    # Single attack
    result = _run_single_attack(
        model, tokenizer, prompts, config, direction, ref_model,
        transfer_models=transfer_models,
        environment_config=environment_config,
        svf_boundary=svf_boundary,
    )

    # Post-attack environment rollout evaluation (continuous mode only —
    # GCG and EGD handle rollout internally during/after optimisation)
    if (
        environment_config is not None
        and result.token_text
        and config.mode == "continuous"
    ):
        from vauban.environment import run_agent_loop

        env_result = run_agent_loop(
            model, tokenizer, environment_config, result.token_text,
        )
        if env_result.reward > 0.0:
            result = replace(
                result,
                success_rate=max(result.success_rate, env_result.reward),
            )

    # Post-attack defense evaluation
    if config.defense_eval is not None and direction is not None:
        layer_index = (
            config.defense_eval_layer
            if config.defense_eval_layer is not None
            else 0
        )
        defense_result = evaluate_against_defenses(
            model, tokenizer, prompts, config,
            direction, layer_index, result.token_text,
        )
        result = replace(result, defense_eval=defense_result)

    return result
