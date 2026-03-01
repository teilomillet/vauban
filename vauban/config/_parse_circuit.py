"""Parse the [circuit] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import CircuitConfig


def _parse_circuit(raw: TomlDict) -> CircuitConfig | None:
    """Parse the optional [circuit] section into a CircuitConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("circuit")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[circuit] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- clean_prompts (required) --
    clean_raw = sec.get("clean_prompts")  # type: ignore[arg-type]
    if clean_raw is None:
        msg = "[circuit].clean_prompts is required"
        raise ValueError(msg)
    if not isinstance(clean_raw, list):
        msg = (
            f"[circuit].clean_prompts must be a list of strings,"
            f" got {type(clean_raw).__name__}"
        )
        raise TypeError(msg)
    for i, item in enumerate(clean_raw):
        if not isinstance(item, str):
            msg = (
                f"[circuit].clean_prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
    clean_prompts: list[str] = list(clean_raw)
    if not clean_prompts:
        msg = "[circuit].clean_prompts must not be empty"
        raise ValueError(msg)

    # -- corrupt_prompts (required) --
    corrupt_raw = sec.get("corrupt_prompts")  # type: ignore[arg-type]
    if corrupt_raw is None:
        msg = "[circuit].corrupt_prompts is required"
        raise ValueError(msg)
    if not isinstance(corrupt_raw, list):
        msg = (
            f"[circuit].corrupt_prompts must be a list of strings,"
            f" got {type(corrupt_raw).__name__}"
        )
        raise TypeError(msg)
    for i, item in enumerate(corrupt_raw):
        if not isinstance(item, str):
            msg = (
                f"[circuit].corrupt_prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
    corrupt_prompts: list[str] = list(corrupt_raw)
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
    metric_raw = sec.get("metric", "kl")  # type: ignore[arg-type]
    if not isinstance(metric_raw, str):
        msg = (
            f"[circuit].metric must be a string,"
            f" got {type(metric_raw).__name__}"
        )
        raise TypeError(msg)
    if metric_raw not in ("kl", "logit_diff"):
        msg = (
            f"[circuit].metric must be 'kl' or 'logit_diff',"
            f" got {metric_raw!r}"
        )
        raise ValueError(msg)

    # -- granularity (optional, default "layer") --
    gran_raw = sec.get("granularity", "layer")  # type: ignore[arg-type]
    if not isinstance(gran_raw, str):
        msg = (
            f"[circuit].granularity must be a string,"
            f" got {type(gran_raw).__name__}"
        )
        raise TypeError(msg)
    if gran_raw not in ("layer", "component"):
        msg = (
            f"[circuit].granularity must be 'layer' or 'component',"
            f" got {gran_raw!r}"
        )
        raise ValueError(msg)

    # -- layers (optional) --
    layers_raw = sec.get("layers")  # type: ignore[arg-type]
    layers: list[int] | None = None
    if layers_raw is not None:
        if not isinstance(layers_raw, list):
            msg = (
                f"[circuit].layers must be a list of integers,"
                f" got {type(layers_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(layers_raw):
            if not isinstance(item, int):
                msg = (
                    f"[circuit].layers[{i}] must be an integer,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
        layers = list(layers_raw)

    # -- token_position (optional, default -1) --
    tok_pos_raw = sec.get("token_position", -1)  # type: ignore[arg-type]
    if not isinstance(tok_pos_raw, int):
        msg = (
            f"[circuit].token_position must be an integer,"
            f" got {type(tok_pos_raw).__name__}"
        )
        raise TypeError(msg)

    # -- attribute_direction (optional, default false) --
    attr_dir_raw = sec.get("attribute_direction", False)  # type: ignore[arg-type]
    if not isinstance(attr_dir_raw, bool):
        msg = (
            f"[circuit].attribute_direction must be a boolean,"
            f" got {type(attr_dir_raw).__name__}"
        )
        raise TypeError(msg)

    # -- logit_diff_tokens (optional) --
    ldt_raw = sec.get("logit_diff_tokens")  # type: ignore[arg-type]
    logit_diff_tokens: list[int] | None = None
    if ldt_raw is not None:
        if not isinstance(ldt_raw, list):
            msg = (
                f"[circuit].logit_diff_tokens must be a list of integers,"
                f" got {type(ldt_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(ldt_raw):
            if not isinstance(item, int):
                msg = (
                    f"[circuit].logit_diff_tokens[{i}] must be an integer,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
        logit_diff_tokens = list(ldt_raw)
        if not logit_diff_tokens:
            msg = "[circuit].logit_diff_tokens must not be empty"
            raise ValueError(msg)

    return CircuitConfig(
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        metric=metric_raw,
        granularity=gran_raw,
        layers=layers,
        token_position=tok_pos_raw,
        attribute_direction=attr_dir_raw,
        logit_diff_tokens=logit_diff_tokens,
    )
