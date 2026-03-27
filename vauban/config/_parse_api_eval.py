# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [api_eval] section of a TOML config."""

from typing import cast

from vauban.config._types import TomlDict
from vauban.types import ApiEvalConfig, ApiEvalEndpoint


def _parse_api_eval(raw: TomlDict) -> ApiEvalConfig | None:
    """Parse the optional [api_eval] section into an ApiEvalConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("api_eval")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[api_eval] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)
    sec = cast("dict[str, object]", sec)

    # -- endpoints (required) --
    endpoints_raw = sec.get("endpoints")
    if endpoints_raw is None:
        msg = "[api_eval] must have 'endpoints' key"
        raise ValueError(msg)
    if not isinstance(endpoints_raw, list):
        msg = (
            f"[api_eval].endpoints must be a list,"
            f" got {type(endpoints_raw).__name__}"
        )
        raise TypeError(msg)
    if len(endpoints_raw) == 0:
        msg = "[api_eval].endpoints must not be empty"
        raise ValueError(msg)

    endpoints: list[ApiEvalEndpoint] = []
    for i, ep_raw in enumerate(endpoints_raw):
        if not isinstance(ep_raw, dict):
            msg = (
                f"[[api_eval.endpoints]][{i}] must be a table,"
                f" got {type(ep_raw).__name__}"
            )
            raise TypeError(msg)
        ep = cast("dict[str, object]", ep_raw)
        endpoints.append(_parse_endpoint(ep, i))

    # -- max_tokens --
    max_tokens_raw = sec.get("max_tokens", 100)
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[api_eval].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if max_tokens_raw < 1:
        msg = f"[api_eval].max_tokens must be >= 1, got {max_tokens_raw}"
        raise ValueError(msg)

    # -- timeout --
    timeout_raw = sec.get("timeout", 30)
    if not isinstance(timeout_raw, int):
        msg = (
            f"[api_eval].timeout must be an integer,"
            f" got {type(timeout_raw).__name__}"
        )
        raise TypeError(msg)
    if timeout_raw < 1:
        msg = f"[api_eval].timeout must be >= 1, got {timeout_raw}"
        raise ValueError(msg)

    # -- system_prompt --
    system_prompt_raw = sec.get("system_prompt")
    system_prompt: str | None = None
    if system_prompt_raw is not None:
        if not isinstance(system_prompt_raw, str):
            msg = (
                f"[api_eval].system_prompt must be a string,"
                f" got {type(system_prompt_raw).__name__}"
            )
            raise TypeError(msg)
        system_prompt = system_prompt_raw

    # -- multiturn --
    multiturn_raw = sec.get("multiturn", False)
    if not isinstance(multiturn_raw, bool):
        msg = (
            f"[api_eval].multiturn must be a boolean,"
            f" got {type(multiturn_raw).__name__}"
        )
        raise TypeError(msg)

    # -- multiturn_max_turns --
    multiturn_max_turns_raw = sec.get("multiturn_max_turns", 3)
    if not isinstance(multiturn_max_turns_raw, int):
        msg = (
            f"[api_eval].multiturn_max_turns must be an integer,"
            f" got {type(multiturn_max_turns_raw).__name__}"
        )
        raise TypeError(msg)
    if multiturn_max_turns_raw < 1:
        msg = (
            f"[api_eval].multiturn_max_turns must be >= 1,"
            f" got {multiturn_max_turns_raw}"
        )
        raise ValueError(msg)

    # -- follow_up_prompts --
    follow_up_raw = sec.get("follow_up_prompts", [])
    if not isinstance(follow_up_raw, list):
        msg = (
            f"[api_eval].follow_up_prompts must be a list,"
            f" got {type(follow_up_raw).__name__}"
        )
        raise TypeError(msg)
    follow_up_prompts: list[str] = []
    for i, item in enumerate(follow_up_raw):
        if not isinstance(item, str):
            msg = (
                f"[api_eval].follow_up_prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
        follow_up_prompts.append(item)

    # -- token_text (standalone mode) --
    token_text: str | None = None
    token_text_raw = sec.get("token_text")
    if token_text_raw is not None:
        if not isinstance(token_text_raw, str):
            msg = (
                f"[api_eval].token_text must be a string,"
                f" got {type(token_text_raw).__name__}"
            )
            raise TypeError(msg)
        if not token_text_raw:
            msg = "[api_eval].token_text must be non-empty"
            raise ValueError(msg)
        token_text = token_text_raw

    # -- token_position --
    _valid_positions = {"prefix", "suffix", "infix"}
    token_position_raw = sec.get("token_position", "suffix")
    if not isinstance(token_position_raw, str):
        msg = (
            f"[api_eval].token_position must be a string,"
            f" got {type(token_position_raw).__name__}"
        )
        raise TypeError(msg)
    if token_position_raw not in _valid_positions:
        msg = (
            f"[api_eval].token_position must be one of"
            f" {sorted(_valid_positions)}, got {token_position_raw!r}"
        )
        raise ValueError(msg)

    # -- prompts --
    prompts_raw = sec.get("prompts", [])
    if not isinstance(prompts_raw, list):
        msg = (
            f"[api_eval].prompts must be a list,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    prompts: list[str] = []
    for i, item in enumerate(prompts_raw):
        if not isinstance(item, str):
            msg = (
                f"[api_eval].prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
        prompts.append(item)

    # Validate: standalone mode requires prompts
    if token_text is not None and not prompts:
        msg = (
            "[api_eval].prompts must be non-empty"
            " when token_text is set (standalone mode)"
        )
        raise ValueError(msg)

    # -- defense_proxy --
    _valid_proxy = {"sic", "cast", "both"}
    defense_proxy: str | None = None
    defense_proxy_raw = sec.get("defense_proxy")
    if defense_proxy_raw is not None:
        if not isinstance(defense_proxy_raw, str):
            msg = (
                f"[api_eval].defense_proxy must be a string,"
                f" got {type(defense_proxy_raw).__name__}"
            )
            raise TypeError(msg)
        if defense_proxy_raw not in _valid_proxy:
            msg = (
                f"[api_eval].defense_proxy must be one of"
                f" {sorted(_valid_proxy)}, got {defense_proxy_raw!r}"
            )
            raise ValueError(msg)
        defense_proxy = defense_proxy_raw

    # -- defense_proxy_sic_mode --
    _valid_sic_modes = {"direction", "generation", "svf"}
    dp_sic_mode_raw = sec.get("defense_proxy_sic_mode", "direction")
    if not isinstance(dp_sic_mode_raw, str):
        msg = (
            f"[api_eval].defense_proxy_sic_mode must be a string,"
            f" got {type(dp_sic_mode_raw).__name__}"
        )
        raise TypeError(msg)
    if dp_sic_mode_raw not in _valid_sic_modes:
        msg = (
            f"[api_eval].defense_proxy_sic_mode must be one of"
            f" {sorted(_valid_sic_modes)}, got {dp_sic_mode_raw!r}"
        )
        raise ValueError(msg)

    # -- defense_proxy_sic_threshold --
    dp_sic_threshold_raw = sec.get("defense_proxy_sic_threshold", 0.0)
    if not isinstance(dp_sic_threshold_raw, (int, float)):
        msg = (
            f"[api_eval].defense_proxy_sic_threshold must be a number,"
            f" got {type(dp_sic_threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- defense_proxy_sic_max_iterations --
    dp_sic_max_iter_raw = sec.get("defense_proxy_sic_max_iterations", 3)
    if not isinstance(dp_sic_max_iter_raw, int):
        msg = (
            f"[api_eval].defense_proxy_sic_max_iterations must be an integer,"
            f" got {type(dp_sic_max_iter_raw).__name__}"
        )
        raise TypeError(msg)
    if dp_sic_max_iter_raw < 1:
        msg = (
            f"[api_eval].defense_proxy_sic_max_iterations must be >= 1,"
            f" got {dp_sic_max_iter_raw}"
        )
        raise ValueError(msg)

    # -- defense_proxy_cast_mode --
    _valid_cast_modes = {"gate", "full"}
    dp_cast_mode_raw = sec.get("defense_proxy_cast_mode", "gate")
    if not isinstance(dp_cast_mode_raw, str):
        msg = (
            f"[api_eval].defense_proxy_cast_mode must be a string,"
            f" got {type(dp_cast_mode_raw).__name__}"
        )
        raise TypeError(msg)
    if dp_cast_mode_raw not in _valid_cast_modes:
        msg = (
            f"[api_eval].defense_proxy_cast_mode must be one of"
            f" {sorted(_valid_cast_modes)}, got {dp_cast_mode_raw!r}"
        )
        raise ValueError(msg)

    # -- defense_proxy_cast_threshold --
    dp_cast_threshold_raw = sec.get("defense_proxy_cast_threshold", 0.0)
    if not isinstance(dp_cast_threshold_raw, (int, float)):
        msg = (
            f"[api_eval].defense_proxy_cast_threshold must be a number,"
            f" got {type(dp_cast_threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- defense_proxy_cast_layers --
    dp_cast_layers: list[int] | None = None
    dp_cast_layers_raw = sec.get("defense_proxy_cast_layers")
    if dp_cast_layers_raw is not None:
        if not isinstance(dp_cast_layers_raw, list):
            msg = (
                f"[api_eval].defense_proxy_cast_layers must be a list,"
                f" got {type(dp_cast_layers_raw).__name__}"
            )
            raise TypeError(msg)
        dp_cast_layers = []
        for i, item in enumerate(dp_cast_layers_raw):
            if not isinstance(item, int):
                msg = (
                    f"[api_eval].defense_proxy_cast_layers[{i}]"
                    f" must be an integer, got {type(item).__name__}"
                )
                raise TypeError(msg)
            dp_cast_layers.append(item)

    # -- defense_proxy_cast_alpha --
    dp_cast_alpha_raw = sec.get("defense_proxy_cast_alpha", 1.0)
    if not isinstance(dp_cast_alpha_raw, (int, float)):
        msg = (
            f"[api_eval].defense_proxy_cast_alpha must be a number,"
            f" got {type(dp_cast_alpha_raw).__name__}"
        )
        raise TypeError(msg)

    # -- defense_proxy_cast_max_tokens --
    dp_cast_max_tokens_raw = sec.get("defense_proxy_cast_max_tokens", 100)
    if not isinstance(dp_cast_max_tokens_raw, int):
        msg = (
            f"[api_eval].defense_proxy_cast_max_tokens must be an integer,"
            f" got {type(dp_cast_max_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if dp_cast_max_tokens_raw < 1:
        msg = (
            f"[api_eval].defense_proxy_cast_max_tokens must be >= 1,"
            f" got {dp_cast_max_tokens_raw}"
        )
        raise ValueError(msg)

    return ApiEvalConfig(
        endpoints=endpoints,
        max_tokens=max_tokens_raw,
        timeout=timeout_raw,
        system_prompt=system_prompt,
        multiturn=multiturn_raw,
        multiturn_max_turns=multiturn_max_turns_raw,
        follow_up_prompts=follow_up_prompts,
        token_text=token_text,
        token_position=token_position_raw,
        prompts=prompts,
        defense_proxy=defense_proxy,
        defense_proxy_sic_mode=dp_sic_mode_raw,
        defense_proxy_sic_threshold=float(dp_sic_threshold_raw),
        defense_proxy_sic_max_iterations=dp_sic_max_iter_raw,
        defense_proxy_cast_mode=dp_cast_mode_raw,
        defense_proxy_cast_threshold=float(dp_cast_threshold_raw),
        defense_proxy_cast_layers=dp_cast_layers,
        defense_proxy_cast_alpha=float(dp_cast_alpha_raw),
        defense_proxy_cast_max_tokens=dp_cast_max_tokens_raw,
    )


def _parse_endpoint(ep: dict[str, object], index: int) -> ApiEvalEndpoint:
    """Parse a single [[api_eval.endpoints]] entry."""
    prefix = f"[[api_eval.endpoints]][{index}]"

    # -- name --
    name_raw = ep.get("name")
    if not isinstance(name_raw, str) or not name_raw:
        msg = f"{prefix}.name must be a non-empty string"
        raise ValueError(msg)

    # -- base_url --
    base_url_raw = ep.get("base_url")
    if not isinstance(base_url_raw, str) or not base_url_raw:
        msg = f"{prefix}.base_url must be a non-empty string"
        raise ValueError(msg)
    if not base_url_raw.startswith(("http://", "https://")):
        msg = (
            f"{prefix}.base_url must start with http:// or https://,"
            f" got {base_url_raw!r}"
        )
        raise ValueError(msg)

    # -- model --
    model_raw = ep.get("model")
    if not isinstance(model_raw, str) or not model_raw:
        msg = f"{prefix}.model must be a non-empty string"
        raise ValueError(msg)

    # -- api_key_env --
    api_key_env_raw = ep.get("api_key_env")
    if not isinstance(api_key_env_raw, str) or not api_key_env_raw:
        msg = f"{prefix}.api_key_env must be a non-empty string"
        raise ValueError(msg)

    # -- system_prompt (per-endpoint override) --
    system_prompt_raw = ep.get("system_prompt")
    system_prompt: str | None = None
    if system_prompt_raw is not None:
        if not isinstance(system_prompt_raw, str):
            msg = (
                f"{prefix}.system_prompt must be a string,"
                f" got {type(system_prompt_raw).__name__}"
            )
            raise TypeError(msg)
        system_prompt = system_prompt_raw

    # -- auth_header (custom auth header name) --
    auth_header_raw = ep.get("auth_header")
    auth_header: str | None = None
    if auth_header_raw is not None:
        if not isinstance(auth_header_raw, str) or not auth_header_raw:
            msg = f"{prefix}.auth_header must be a non-empty string"
            raise ValueError(msg)
        auth_header = auth_header_raw

    return ApiEvalEndpoint(
        name=name_raw,
        base_url=base_url_raw,
        model=model_raw,
        api_key_env=api_key_env_raw,
        system_prompt=system_prompt,
        auth_header=auth_header,
    )
