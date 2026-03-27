"""Parse the [circuit] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import CircuitConfig


def _parse_circuit(raw: TomlDict) -> CircuitConfig | None:
    """Parse the optional [circuit] section into a CircuitConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("circuit")
    if sec is None:
        return None
    reader = SectionReader("[circuit]", require_toml_table("[circuit]", sec))

    # -- clean_prompts (required) --
    clean_prompts = reader.string_list("clean_prompts")
    if not clean_prompts:
        msg = "[circuit].clean_prompts must not be empty"
        raise ValueError(msg)

    # -- corrupt_prompts (required) --
    corrupt_prompts = reader.string_list("corrupt_prompts")
    if not corrupt_prompts:
        msg = "[circuit].corrupt_prompts must not be empty"
        raise ValueError(msg)
    if len(clean_prompts) != len(corrupt_prompts):
        msg = (
            f"[circuit].clean_prompts and corrupt_prompts must have"
            f" the same length ({len(clean_prompts)} != {len(corrupt_prompts)})"
        )
        raise ValueError(msg)

    # -- metric (optional, default "kl") --
    metric = reader.literal("metric", ("kl", "logit_diff"), default="kl")

    # -- granularity (optional, default "layer") --
    granularity = reader.literal(
        "granularity", ("layer", "component"), default="layer",
    )

    # -- layers (optional) --
    layers = reader.optional_int_list("layers")

    # -- token_position (optional, default -1) --
    token_position = reader.integer("token_position", default=-1)

    # -- attribute_direction (optional, default false) --
    attribute_direction = reader.boolean("attribute_direction", default=False)

    # -- logit_diff_tokens (optional) --
    logit_diff_tokens = reader.optional_int_list("logit_diff_tokens")
    if logit_diff_tokens is not None and not logit_diff_tokens:
        msg = "[circuit].logit_diff_tokens must not be empty"
        raise ValueError(msg)

    return CircuitConfig(
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        metric=metric,
        granularity=granularity,
        layers=layers,
        token_position=token_position,
        attribute_direction=attribute_direction,
        logit_diff_tokens=logit_diff_tokens,
    )
