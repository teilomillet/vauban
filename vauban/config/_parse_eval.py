"""Parse the [eval] section of a TOML config."""

from pathlib import Path

from vauban.config._types import TomlDict
from vauban.types import EvalConfig


def _parse_eval(base_dir: Path, raw: TomlDict) -> EvalConfig:
    """Parse the optional [eval] section into an EvalConfig.

    Returns a default EvalConfig if the section is absent.
    """
    sec = raw.get("eval")
    if sec is None:
        return EvalConfig()
    if not isinstance(sec, dict):
        msg = f"[eval] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # prompts (optional string path)
    prompts_path: Path | None = None
    prompts_raw = sec.get("prompts")  # type: ignore[arg-type]
    if prompts_raw is not None:
        if not isinstance(prompts_raw, str):
            msg = (
                f"[eval].prompts must be a string,"
                f" got {type(prompts_raw).__name__}"
            )
            raise TypeError(msg)
        prompts_path = base_dir / prompts_raw

    # max_tokens (int >= 1, default 100)
    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[eval].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if max_tokens_raw < 1:
        msg = f"[eval].max_tokens must be >= 1, got {max_tokens_raw}"
        raise ValueError(msg)

    # num_prompts (int >= 1, default 20)
    num_prompts_raw = sec.get("num_prompts", 20)  # type: ignore[arg-type]
    if not isinstance(num_prompts_raw, int):
        msg = (
            f"[eval].num_prompts must be an integer,"
            f" got {type(num_prompts_raw).__name__}"
        )
        raise TypeError(msg)
    if num_prompts_raw < 1:
        msg = f"[eval].num_prompts must be >= 1, got {num_prompts_raw}"
        raise ValueError(msg)

    # refusal_phrases (optional string path)
    refusal_phrases_path: Path | None = None
    refusal_raw = sec.get("refusal_phrases")  # type: ignore[arg-type]
    if refusal_raw is not None:
        if not isinstance(refusal_raw, str):
            msg = (
                f"[eval].refusal_phrases must be a string,"
                f" got {type(refusal_raw).__name__}"
            )
            raise TypeError(msg)
        refusal_phrases_path = base_dir / refusal_raw

    return EvalConfig(
        prompts_path=prompts_path,
        max_tokens=max_tokens_raw,
        num_prompts=num_prompts_raw,
        refusal_phrases_path=refusal_phrases_path,
    )
