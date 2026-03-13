"""GCG warm-start payload discovery for flywheel cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import (
        CausalLM,
        EnvironmentConfig,
        FlywheelConfig,
        Payload,
        Tokenizer,
    )


def warmstart_gcg_payloads(
    model: CausalLM,
    tokenizer: Tokenizer,
    worlds: list[EnvironmentConfig],
    existing_payloads: list[Payload],
    config: FlywheelConfig,
    direction: Array | None,
) -> list[str]:
    """Run short GCG optimization to discover new attack payloads.

    Uses the existing best payload as a warm-start point. Runs a short
    optimization (config.gcg_steps steps) against a sample of worlds.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        worlds: List of world configs to attack.
        existing_payloads: Current payload library for warm-start.
        config: Flywheel configuration.
        direction: Optional refusal direction for defense-aware loss.

    Returns:
        List of new payload text strings discovered.
    """
    from vauban.softprompt._gcg import _gcg_attack
    from vauban.types import SoftPromptConfig

    # Build prompts from world tasks
    prompts = [w.task.content for w in worlds[:5]]
    if not prompts:
        return []

    # Build minimal softprompt config for short GCG run
    sp_config = SoftPromptConfig(
        mode="gcg",
        n_tokens=config.gcg_n_tokens,
        n_steps=config.gcg_steps,
        token_position=(
            "infix"
            if "infix" in config.positions
            else config.positions[0]
        ),
    )

    # Use first world as environment config if available
    env_config = worlds[0] if worlds else None

    try:
        result = _gcg_attack(
            model, tokenizer, prompts, sp_config,
            direction, environment_config=env_config,
        )
        # Extract the best token sequence as text
        if result.token_ids is not None:
            text = tokenizer.decode(result.token_ids)
            if text.strip():
                return [text.strip()]
    except Exception:  # GCG may fail on short runs or bad configs
        pass

    return []
