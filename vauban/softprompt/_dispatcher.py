"""Entry point dispatcher for soft prompt attacks."""

import random

import mlx.core as mx

from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._gcg import _gcg_attack
from vauban.types import CausalLM, SoftPromptConfig, SoftPromptResult, Tokenizer


def softprompt_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None = None,
    ref_model: CausalLM | None = None,
) -> SoftPromptResult:
    """Run a soft prompt attack against a model.

    Optimizes a learnable prefix in embedding space that steers generation
    away from refusal. Supports continuous (gradient-based), GCG
    (discrete token search), and EGD (exponentiated gradient descent) modes.

    Args:
        model: The causal language model to attack.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration.
        direction: Optional refusal direction for direction-guided mode.
        ref_model: Optional reference model for KL collision loss.
    """
    if config.seed is not None:
        mx.random.seed(config.seed)
        random.seed(config.seed)

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
