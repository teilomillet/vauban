"""Parse the [surface] section of a TOML config."""

from pathlib import Path

from vauban.config._types import TomlDict
from vauban.types import SurfaceConfig


def _parse_surface(base_dir: Path, raw: TomlDict) -> SurfaceConfig | None:
    """Parse the optional [surface] section into a SurfaceConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("surface")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[surface] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    prompts_raw = sec.get("prompts", "default")  # type: ignore[arg-type]
    if not isinstance(prompts_raw, str):
        msg = (
            f"[surface].prompts must be a string,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    prompts_path: Path | str
    if prompts_raw in ("default", "default_multilingual"):
        prompts_path = prompts_raw
    else:
        prompts_path = base_dir / prompts_raw

    generate_raw = sec.get("generate", True)  # type: ignore[arg-type]
    if not isinstance(generate_raw, bool):
        msg = (
            f"[surface].generate must be a boolean,"
            f" got {type(generate_raw).__name__}"
        )
        raise TypeError(msg)

    max_tokens_raw = sec.get("max_tokens", 20)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[surface].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    progress_raw = sec.get("progress", True)  # type: ignore[arg-type]
    if not isinstance(progress_raw, bool):
        msg = (
            f"[surface].progress must be a boolean,"
            f" got {type(progress_raw).__name__}"
        )
        raise TypeError(msg)

    return SurfaceConfig(
        prompts_path=prompts_path,
        generate=generate_raw,
        max_tokens=max_tokens_raw,
        progress=progress_raw,
    )
