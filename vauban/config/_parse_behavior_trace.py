# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [behavior_trace] section of a TOML config."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import BehaviorTraceConfig, BehaviorTracePromptConfig

if TYPE_CHECKING:
    from vauban.config._types import TomlDict

_EXPECTED_BEHAVIOR_CHOICES: tuple[str, ...] = (
    "refuse",
    "comply",
    "express_uncertainty",
    "ask_clarifying_question",
    "defer",
    "unknown",
)
_REDACTION_CHOICES: tuple[str, ...] = ("safe", "redacted", "omitted")


def _parse_behavior_trace(
    base_dir: Path,
    raw: TomlDict,
) -> BehaviorTraceConfig | None:
    """Parse the optional [behavior_trace] section."""
    sec = raw.get("behavior_trace")
    if sec is None:
        return None
    reader = SectionReader(
        "[behavior_trace]",
        require_toml_table("[behavior_trace]", sec),
    )

    suite_path = _optional_path(base_dir, reader.optional_string("suite"))
    suite_config = (
        _parse_suite_file(suite_path)
        if suite_path is not None
        else _empty_suite_config()
    )
    inline_prompts = _parse_prompt_configs(
        reader.data.get("prompts"),
        section="[behavior_trace].prompts",
    )
    prompts = [*suite_config.prompts, *inline_prompts]
    if not prompts:
        msg = (
            "[behavior_trace] requires prompts either via suite = \"...\""
            " or [[behavior_trace.prompts]]"
        )
        raise ValueError(msg)
    _reject_duplicate_prompt_ids(prompts)

    max_tokens = reader.integer("max_tokens", default=80)
    if max_tokens < 1:
        msg = "[behavior_trace].max_tokens must be >= 1"
        raise ValueError(msg)

    refusal_phrases = reader.string_list(
        "refusal_phrases",
        default=list(DEFAULT_REFUSAL_PHRASES),
    )
    if not refusal_phrases:
        msg = "[behavior_trace].refusal_phrases must be non-empty"
        raise ValueError(msg)

    return BehaviorTraceConfig(
        model_label=reader.string(
            "model_label",
            default=suite_config.model_label,
        ),
        suite=suite_path,
        suite_name=reader.string("suite_name", default=suite_config.name),
        suite_description=reader.string(
            "suite_description",
            default=suite_config.description,
        ),
        suite_version=(
            reader.optional_string("suite_version") or suite_config.version
        ),
        suite_source=(
            reader.optional_string("suite_source") or suite_config.source
        ),
        safety_policy=reader.string(
            "safety_policy",
            default=suite_config.safety_policy,
        ),
        prompts=prompts,
        max_tokens=max_tokens,
        refusal_phrases=refusal_phrases,
        record_outputs=reader.boolean("record_outputs", default=False),
        output_trace=_optional_path(
            base_dir,
            reader.optional_string("output_trace"),
        ),
        trace_filename=reader.string(
            "trace_filename",
            default="behavior_trace.jsonl",
        ),
        json_filename=reader.string(
            "json_filename",
            default="behavior_trace_report.json",
        ),
    )


@dataclass(frozen=True, slots=True)
class _SuiteConfig:
    """Parsed behavior suite metadata and prompts."""

    name: str
    description: str
    model_label: str
    safety_policy: str
    prompts: list[BehaviorTracePromptConfig]
    version: str | None = None
    source: str | None = None


def _empty_suite_config() -> _SuiteConfig:
    """Return default suite metadata when no external suite is used."""
    return _SuiteConfig(
        name="behavior-change-suite",
        description="Behavior trace collection suite.",
        model_label="model",
        safety_policy="safe_or_redacted_prompts",
        prompts=[],
    )


def _parse_suite_file(path: Path) -> _SuiteConfig:
    """Parse an external [behavior_suite] TOML file."""
    if not path.exists():
        msg = f"[behavior_trace].suite does not exist: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as handle:
        raw: TomlDict = tomllib.load(handle)
    sec = raw.get("behavior_suite")
    reader = SectionReader(
        "[behavior_suite]",
        require_toml_table("[behavior_suite]", sec),
    )
    source = reader.optional_string("source") or str(path)
    return _SuiteConfig(
        name=reader.string("name", default="behavior-change-suite"),
        description=reader.string(
            "description",
            default="Behavior trace collection suite.",
        ),
        model_label=reader.string("model_label", default="model"),
        version=reader.optional_string("version"),
        source=source,
        safety_policy=reader.string(
            "safety_policy",
            default="safe_or_redacted_prompts",
        ),
        prompts=_parse_prompt_configs(
            reader.data.get("prompts"),
            section="[behavior_suite].prompts",
        ),
    )


def _parse_prompt_configs(
    raw: object,
    *,
    section: str,
) -> list[BehaviorTracePromptConfig]:
    """Parse prompt entries from an array of strings or tables."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = f"{section} must be an array"
        raise TypeError(msg)

    prompts: list[BehaviorTracePromptConfig] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append(
                BehaviorTracePromptConfig(
                    prompt_id=f"prompt-{index + 1:03d}",
                    text=item,
                ),
            )
            continue
        table = require_toml_table(f"{section}[{index}]", item)
        reader = SectionReader(f"{section}[{index}]", table)
        prompt_id = reader.optional_string("id")
        if prompt_id is None:
            prompt_id = reader.optional_string("prompt_id")
        if prompt_id is None:
            msg = f"{section}[{index}].id is required"
            raise ValueError(msg)
        text = reader.optional_string("text")
        if text is None:
            text = reader.optional_string("prompt")
        if text is None:
            msg = f"{section}[{index}].text is required"
            raise ValueError(msg)
        prompts.append(
            BehaviorTracePromptConfig(
                prompt_id=prompt_id,
                text=text,
                category=reader.string("category", default="default"),
                expected_behavior=reader.literal(
                    "expected_behavior",
                    _EXPECTED_BEHAVIOR_CHOICES,
                    default="unknown",
                ),
                redaction=reader.literal(
                    "redaction",
                    _REDACTION_CHOICES,
                    default="safe",
                ),
                tags=reader.string_list("tags", default=[]),
            ),
        )
    return prompts


def _optional_path(base_dir: Path, raw: str | None) -> Path | None:
    """Resolve an optional path string relative to the config directory."""
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _reject_duplicate_prompt_ids(
    prompts: list[BehaviorTracePromptConfig],
) -> None:
    """Reject duplicate prompt IDs before trace collection."""
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.prompt_id in seen:
            msg = f"[behavior_trace] duplicate prompt id: {prompt.prompt_id!r}"
            raise ValueError(msg)
        seen.add(prompt.prompt_id)
