"""Iterative GAN-style attack-defense loop.

Each round: attacker optimizes a suffix → defender (SIC + CAST) tries to
block it → attacker gets feedback and escalates parameters for next round.
Warm-starts from the previous round's best suffix.

When ``gan_multiturn`` is enabled, the loop maintains a conversation history
across rounds. Each round appends the attack prompt + model response as a
new turn, so subsequent rounds optimize against the accumulated context.
The attack suffix optimization stays single-turn (GCG/EGD optimize one
user message at a time), but the prompt is pre-encoded with the full
conversation history prepended as hard token IDs.
"""

from dataclasses import replace

from vauban._array import Array
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._defense_eval import (
    evaluate_against_defenses,
    evaluate_against_defenses_multiturn,
)
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._gcg import _gcg_attack
from vauban.softprompt._utils import _pre_encode_prompts_with_history
from vauban.types import (
    CausalLM,
    GanRoundResult,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
)


def _dispatch_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None,
) -> SoftPromptResult:
    """Dispatch to the appropriate attack mode (no defense eval)."""
    if config.mode == "continuous":
        return _continuous_attack(
            model, tokenizer, prompts, config, direction, ref_model,
        )
    if config.mode == "gcg":
        return _gcg_attack(
            model, tokenizer, prompts, config, direction, ref_model,
        )
    if config.mode == "egd":
        return _egd_attack(
            model, tokenizer, prompts, config, direction, ref_model,
        )
    msg = (
        f"Unknown soft prompt mode: {config.mode!r},"
        " must be 'continuous', 'gcg', or 'egd'"
    )
    raise ValueError(msg)


def _dispatch_attack_multiturn(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None,
    history: list[dict[str, str]],
) -> SoftPromptResult:
    """Dispatch attack with conversation history baked into prompt encoding.

    Pre-encodes prompts with ``history`` prepended as hard token IDs,
    then passes the pre-encoded IDs to the attack function via
    ``all_prompt_ids_override``. The attack still optimizes a single
    hard-token suffix, but the loss is computed against the full
    multi-turn context. Only GCG and EGD are supported — continuous
    mode produces soft embeddings that cannot be sent through an API.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Attack prompts for the current turn.
        config: Soft prompt configuration (mode must be 'gcg' or 'egd').
        direction: Refusal direction vector.
        ref_model: Optional reference model for KL collision loss.
        history: Previous conversation turns as role/content dicts.

    Returns:
        SoftPromptResult from the attack.
    """
    all_prompt_ids = _pre_encode_prompts_with_history(
        tokenizer, prompts, history, config.system_prompt,
    )

    if config.mode == "gcg":
        return _gcg_attack(
            model, tokenizer, prompts, config, direction, ref_model,
            all_prompt_ids_override=all_prompt_ids,
        )
    if config.mode == "egd":
        return _egd_attack(
            model, tokenizer, prompts, config, direction, ref_model,
            all_prompt_ids_override=all_prompt_ids,
        )
    msg = (
        f"Multi-turn attack requires mode 'gcg' or 'egd',"
        f" got {config.mode!r}"
    )
    raise ValueError(msg)


def gan_loop(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
) -> SoftPromptResult:
    """Run an iterative GAN-style attack-defense loop.

    Each round:
      1. Attack: run softprompt_attack with current config
      2. Defend: evaluate suffix against SIC and/or CAST
      3. Verdict: did the suffix bypass both defenses?
      4. Escalate: if defender won, increase attack parameters
         (more tokens, more steps, higher direction weight)
      5. Warm-start: pass previous suffix as init_tokens

    When ``config.gan_multiturn`` is True, the loop maintains a
    conversation history. Each round selects a prompt (rotating through
    the prompt list), and after attack + defense, appends the
    attack prompt + model response as a new turn. Subsequent rounds
    see the accumulated history via pre-encoded hard token IDs.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Attack prompts.
        config: Soft prompt configuration with gan_rounds > 0.
        direction: Refusal direction vector.
        ref_model: Optional reference model for KL collision loss.

    Returns:
        SoftPromptResult with gan_history populated.
    """
    # Multi-turn requires hard tokens — continuous mode can't be sent via API
    if config.gan_multiturn and config.mode == "continuous":
        msg = (
            "gan_multiturn requires mode 'gcg' or 'egd' (hard tokens);"
            " continuous mode produces soft embeddings that cannot be"
            " appended to conversation history or sent through an API"
        )
        raise ValueError(msg)

    rounds: list[GanRoundResult] = []
    current_config = config
    best_result: SoftPromptResult | None = None
    best_bypass_score = -1.0

    layer_index = (
        config.defense_eval_layer
        if config.defense_eval_layer is not None
        else 0
    )

    attack_result: SoftPromptResult | None = None
    history: list[dict[str, str]] = []

    for round_idx in range(config.gan_rounds):
        # Select prompt for this round (rotate through list)
        if config.gan_multiturn:
            prompt_idx = round_idx % len(prompts)
            round_prompts = [prompts[prompt_idx]]
        else:
            round_prompts = prompts

        # --- Attack phase ---
        if config.gan_multiturn and history:
            attack_result = _dispatch_attack_multiturn(
                model, tokenizer, round_prompts, current_config,
                direction, ref_model, history,
            )
        else:
            attack_result = _dispatch_attack(
                model, tokenizer, round_prompts, current_config,
                direction, ref_model,
            )

        # --- Defense phase ---
        defense_result = None
        attacker_won = False
        defense_mode = config.defense_eval or "both"

        if direction is not None:
            if config.gan_multiturn and history:
                defense_result = evaluate_against_defenses_multiturn(
                    model, tokenizer, round_prompts, current_config,
                    direction, layer_index, attack_result.token_text,
                    history,
                )
            else:
                defense_result = evaluate_against_defenses(
                    model, tokenizer, round_prompts, current_config,
                    direction, layer_index, attack_result.token_text,
                )

            # Determine if attacker won this round
            sic_survived = defense_result.sic_blocked == 0
            cast_survived = defense_result.cast_refusal_rate == 0.0

            if defense_mode == "sic":
                attacker_won = sic_survived
            elif defense_mode == "cast":
                attacker_won = cast_survived
            else:  # "both"
                attacker_won = sic_survived and cast_survived

        # --- Multi-turn history management ---
        if config.gan_multiturn:
            suffix_text = (
                " " + attack_result.token_text
                if attack_result.token_text
                else ""
            )
            history.append({
                "role": "user",
                "content": round_prompts[0] + suffix_text,
            })
            if attack_result.eval_responses:
                history.append({
                    "role": "assistant",
                    "content": attack_result.eval_responses[0],
                })
            # Trim history to max turns (each turn = user + assistant)
            max_messages = config.gan_multiturn_max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]

        # Track bypass score: higher = better for attacker
        bypass_score = (
            attack_result.success_rate
            + (defense_result.sic_bypass_rate if defense_result else 0.0)
        )

        # Config snapshot for this round
        snapshot: dict[str, object] = {
            "n_tokens": current_config.n_tokens,
            "n_steps": current_config.n_steps,
            "direction_weight": current_config.direction_weight,
            "round": round_idx,
        }

        round_result = GanRoundResult(
            round_index=round_idx,
            attack_result=replace(
                attack_result, defense_eval=defense_result,
            ),
            defense_result=defense_result,
            attacker_won=attacker_won,
            config_snapshot=snapshot,
        )
        rounds.append(round_result)

        # Track best result across all rounds
        if bypass_score > best_bypass_score:
            best_bypass_score = bypass_score
            best_result = replace(
                attack_result, defense_eval=defense_result,
            )

        # --- Early stop: attacker already bypassed defenses ---
        if attacker_won:
            break

        # --- Feedback: escalate attack if defender won ---
        if round_idx < config.gan_rounds - 1:
            current_config = _escalate_config(
                current_config, attack_result,
            )

    # Use best result, or last if none found
    if best_result is not None:
        final = best_result
    elif attack_result is not None:
        final = attack_result
    else:
        msg = "gan_loop called with gan_rounds=0"
        raise ValueError(msg)
    return replace(final, gan_history=rounds)


def _escalate_config(
    config: SoftPromptConfig,
    prev_result: SoftPromptResult,
) -> SoftPromptConfig:
    """Escalate attack config after a failed round.

    Increases n_steps, direction_weight, and n_tokens.
    Warm-starts from previous suffix tokens.

    Args:
        config: Current config to escalate.
        prev_result: Result from the previous attack round.

    Returns:
        New SoftPromptConfig with escalated parameters.
    """
    new_n_steps = int(config.n_steps * config.gan_step_multiplier)
    new_direction_weight = (
        config.direction_weight + config.gan_direction_escalation
    )
    new_n_tokens = config.n_tokens + config.gan_token_escalation

    # Warm-start from previous suffix
    init_tokens = prev_result.token_ids

    return replace(
        config,
        n_steps=new_n_steps,
        direction_weight=new_direction_weight,
        n_tokens=new_n_tokens,
        init_tokens=init_tokens,
    )
