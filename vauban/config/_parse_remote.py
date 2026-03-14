"""Parse the [remote] section of a TOML config."""

from typing import cast

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import RemoteConfig


def _parse_remote(raw: TomlDict) -> RemoteConfig | None:
    """Parse the optional [remote] section into a RemoteConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("remote")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[remote] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)
    sec = cast("TomlDict", sec)

    reader = SectionReader("[remote]", sec)

    backend = reader.literal("backend", ("jsinfer",))
    api_key_env = reader.string("api_key_env")

    models = reader.string_list("models")
    if not models:
        msg = "[remote].models must be non-empty"
        raise ValueError(msg)

    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[remote].prompts must be non-empty"
        raise ValueError(msg)

    activations = reader.boolean("activations", default=False)
    activation_layers = reader.int_list("activation_layers", default=[])
    activation_modules = reader.string_list(
        "activation_modules",
        default=["model.layers.{layer}.mlp.down_proj"],
    )
    max_tokens = reader.integer("max_tokens", default=512)
    timeout = reader.integer("timeout", default=600)

    return RemoteConfig(
        backend=backend,
        api_key_env=api_key_env,
        models=models,
        prompts=prompts,
        activations=activations,
        activation_layers=activation_layers,
        activation_modules=activation_modules,
        max_tokens=max_tokens,
        timeout=timeout,
    )
