"""Parse the [eval] section of a TOML config."""

from pathlib import Path
from typing import cast

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import EvalConfig, ResponseScoreWeights


def _parse_scoring_weights(sec: dict[str, object]) -> ResponseScoreWeights | None:
    """Parse the optional [eval.scoring] sub-table."""
    scoring_raw = sec.get("scoring")
    if scoring_raw is None:
        return None
    if not isinstance(scoring_raw, dict):
        msg = f"[eval.scoring] must be a table, got {type(scoring_raw).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[eval.scoring]", cast("TomlDict", scoring_raw))
    return ResponseScoreWeights(
        length=reader.number("length", default=0.15),
        structure=reader.number("structure", default=0.15),
        anti_refusal=reader.number("anti_refusal", default=0.30),
        directness=reader.number("directness", default=0.20),
        relevance=reader.number("relevance", default=0.20),
    )


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

    reader = SectionReader("[eval]", sec)

    # prompts (optional string path)
    prompts_str = reader.optional_string("prompts")
    prompts_path = base_dir / prompts_str if prompts_str is not None else None

    # max_tokens (int >= 1, default 100)
    max_tokens = reader.integer("max_tokens", default=100)
    if max_tokens < 1:
        msg = f"[eval].max_tokens must be >= 1, got {max_tokens}"
        raise ValueError(msg)

    # num_prompts (int >= 1, default 20)
    num_prompts = reader.integer("num_prompts", default=20)
    if num_prompts < 1:
        msg = f"[eval].num_prompts must be >= 1, got {num_prompts}"
        raise ValueError(msg)

    # refusal_phrases (optional string path)
    refusal_str = reader.optional_string("refusal_phrases")
    refusal_phrases_path = base_dir / refusal_str if refusal_str is not None else None

    # refusal_mode (string, default "phrases")
    refusal_mode = reader.literal(
        "refusal_mode", ("phrases", "judge"), default="phrases",
    )

    # scoring weights (optional sub-table)
    scoring_weights = _parse_scoring_weights(cast("dict[str, object]", sec))

    return EvalConfig(
        prompts_path=prompts_path,
        max_tokens=max_tokens,
        num_prompts=num_prompts,
        refusal_phrases_path=refusal_phrases_path,
        refusal_mode=refusal_mode,
        scoring_weights=scoring_weights,
    )
