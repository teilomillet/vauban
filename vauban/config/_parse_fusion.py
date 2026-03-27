"""Parse the [fusion] section of a TOML config."""

from pathlib import Path

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import FusionConfig


def _parse_fusion(base_dir: Path, raw: TomlDict) -> FusionConfig | None:
    """Parse the optional [fusion] section into a FusionConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("fusion")
    if sec is None:
        return None
    reader = SectionReader("[fusion]", require_toml_table("[fusion]", sec))

    # -- harmful_prompts (required) --
    harmful_prompts = reader.string_list("harmful_prompts")
    if not harmful_prompts:
        msg = "[fusion].harmful_prompts must be non-empty"
        raise ValueError(msg)

    # -- benign_prompts (required) --
    benign_prompts = reader.string_list("benign_prompts")
    if not benign_prompts:
        msg = "[fusion].benign_prompts must be non-empty"
        raise ValueError(msg)

    # -- layer --
    layer = reader.integer("layer", default=-1)

    # -- alpha --
    alpha = reader.number("alpha", default=0.5)
    if alpha < 0.0 or alpha > 1.0:
        msg = f"[fusion].alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)

    # -- n_tokens --
    n_tokens = reader.integer("n_tokens", default=128)
    if n_tokens < 1:
        msg = f"[fusion].n_tokens must be >= 1, got {n_tokens}"
        raise ValueError(msg)

    # -- temperature --
    temperature = reader.number("temperature", default=0.7)
    if temperature <= 0:
        msg = f"[fusion].temperature must be > 0, got {temperature}"
        raise ValueError(msg)

    return FusionConfig(
        harmful_prompts=harmful_prompts,
        benign_prompts=benign_prompts,
        layer=layer,
        alpha=alpha,
        n_tokens=n_tokens,
        temperature=temperature,
    )
