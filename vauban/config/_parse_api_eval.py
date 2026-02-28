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

    return ApiEvalConfig(
        endpoints=endpoints,
        max_tokens=max_tokens_raw,
        timeout=timeout_raw,
        system_prompt=system_prompt,
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

    return ApiEvalEndpoint(
        name=name_raw,
        base_url=base_url_raw,
        model=model_raw,
        api_key_env=api_key_env_raw,
        system_prompt=system_prompt,
    )
