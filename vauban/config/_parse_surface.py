"""Parse the [surface] section of a TOML config."""

from pathlib import Path
from typing import cast

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
    sec_dict = cast("dict[str, object]", sec)

    prompts_raw = sec_dict.get("prompts", "default")
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

    generate_raw = sec_dict.get("generate", True)
    if not isinstance(generate_raw, bool):
        msg = (
            f"[surface].generate must be a boolean,"
            f" got {type(generate_raw).__name__}"
        )
        raise TypeError(msg)

    max_tokens_raw = sec_dict.get("max_tokens", 20)
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[surface].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    progress_raw = sec_dict.get("progress", True)
    if not isinstance(progress_raw, bool):
        msg = (
            f"[surface].progress must be a boolean,"
            f" got {type(progress_raw).__name__}"
        )
        raise TypeError(msg)

    max_worst_after_raw = sec_dict.get("max_worst_cell_refusal_after")
    if (
        max_worst_after_raw is not None
        and (
            isinstance(max_worst_after_raw, bool)
            or not isinstance(max_worst_after_raw, int | float)
        )
    ):
        msg = (
            "[surface].max_worst_cell_refusal_after must be a number in [0, 1],"
            f" got {type(max_worst_after_raw).__name__}"
        )
        raise TypeError(msg)
    max_worst_after: float | None = None
    if max_worst_after_raw is not None:
        max_worst_after = float(max_worst_after_raw)
        if max_worst_after < 0.0 or max_worst_after > 1.0:
            msg = (
                "[surface].max_worst_cell_refusal_after must be in [0, 1],"
                f" got {max_worst_after}"
            )
            raise ValueError(msg)

    max_worst_delta_raw = sec_dict.get("max_worst_cell_refusal_delta")
    if (
        max_worst_delta_raw is not None
        and (
            isinstance(max_worst_delta_raw, bool)
            or not isinstance(max_worst_delta_raw, int | float)
        )
    ):
        msg = (
            "[surface].max_worst_cell_refusal_delta must be a number in [0, 1],"
            f" got {type(max_worst_delta_raw).__name__}"
        )
        raise TypeError(msg)
    max_worst_delta: float | None = None
    if max_worst_delta_raw is not None:
        max_worst_delta = float(max_worst_delta_raw)
        if max_worst_delta < 0.0 or max_worst_delta > 1.0:
            msg = (
                "[surface].max_worst_cell_refusal_delta must be in [0, 1],"
                f" got {max_worst_delta}"
            )
            raise ValueError(msg)

    min_coverage_raw = sec_dict.get("min_coverage_score")
    if (
        min_coverage_raw is not None
        and (
            isinstance(min_coverage_raw, bool)
            or not isinstance(min_coverage_raw, int | float)
        )
    ):
        msg = (
            "[surface].min_coverage_score must be a number in [0, 1],"
            f" got {type(min_coverage_raw).__name__}"
        )
        raise TypeError(msg)
    min_coverage: float | None = None
    if min_coverage_raw is not None:
        min_coverage = float(min_coverage_raw)
        if min_coverage < 0.0 or min_coverage > 1.0:
            msg = (
                "[surface].min_coverage_score must be in [0, 1],"
                f" got {min_coverage}"
            )
            raise ValueError(msg)

    return SurfaceConfig(
        prompts_path=prompts_path,
        generate=generate_raw,
        max_tokens=max_tokens_raw,
        progress=progress_raw,
        max_worst_cell_refusal_after=max_worst_after,
        max_worst_cell_refusal_delta=max_worst_delta,
        min_coverage_score=min_coverage,
    )
