"""Parse the [grpo] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import GRPOConfig


def _parse_grpo(raw: TomlDict) -> GRPOConfig | None:
    """Parse the optional [grpo] section into a GRPOConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("grpo")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[grpo] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[grpo]", sec)

    # -- n_steps --
    n_steps = reader.integer("n_steps", default=50)
    if n_steps < 1:
        msg = f"[grpo].n_steps must be >= 1, got {n_steps}"
        raise ValueError(msg)

    # -- group_size --
    group_size = reader.integer("group_size", default=4)
    if group_size < 2:
        msg = f"[grpo].group_size must be >= 2, got {group_size}"
        raise ValueError(msg)

    # -- learning_rate --
    learning_rate = reader.number("learning_rate", default=1e-4)
    if learning_rate <= 0:
        msg = f"[grpo].learning_rate must be > 0, got {learning_rate}"
        raise ValueError(msg)

    # -- max_gen_tokens --
    max_gen_tokens = reader.integer("max_gen_tokens", default=100)
    if max_gen_tokens < 1:
        msg = f"[grpo].max_gen_tokens must be >= 1, got {max_gen_tokens}"
        raise ValueError(msg)

    # -- kl_weight --
    kl_weight = reader.number("kl_weight", default=0.01)
    if kl_weight < 0:
        msg = f"[grpo].kl_weight must be >= 0, got {kl_weight}"
        raise ValueError(msg)

    # -- clip_range --
    clip_range = reader.number("clip_range", default=0.2)
    if clip_range <= 0 or clip_range >= 1.0:
        msg = f"[grpo].clip_range must be in (0, 1), got {clip_range}"
        raise ValueError(msg)

    # -- reward_mode --
    reward_mode = reader.literal(
        "reward_mode",
        ("keyword",),
        default="keyword",
    )

    # -- batch_size --
    batch_size = reader.integer("batch_size", default=4)
    if batch_size < 1:
        msg = f"[grpo].batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    # -- prompt_pool_size --
    prompt_pool_size = reader.optional_integer("prompt_pool_size")

    return GRPOConfig(
        n_steps=n_steps,
        group_size=group_size,
        learning_rate=learning_rate,
        max_gen_tokens=max_gen_tokens,
        kl_weight=kl_weight,
        clip_range=clip_range,
        reward_mode=reward_mode,
        batch_size=batch_size,
        prompt_pool_size=prompt_pool_size,
    )
