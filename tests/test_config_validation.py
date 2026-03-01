"""Tests for config validation rules."""

import json
from pathlib import Path

import pytest

from vauban.config._validation import (
    ValidationContext,
    _add_warning,
    _load_refusal_phrases,
    _rule_depth_extract_direction,
    _rule_early_mode_conflicts,
    _rule_eval_without_prompts,
    _rule_output_dir,
    _rule_skipped_sections,
    _validate_prompt_jsonl_file,
)
from vauban.types import (
    CastConfig,
    CutConfig,
    DepthConfig,
    DetectConfig,
    EvalConfig,
    PipelineConfig,
    ProbeConfig,
    SurfaceConfig,
)


def _make_config(**overrides: object) -> PipelineConfig:
    """Build a PipelineConfig with sensible defaults."""
    defaults: dict[str, object] = {
        "model_path": "test-model",
        "harmful_path": Path("/fake/harmful.jsonl"),
        "harmless_path": Path("/fake/harmless.jsonl"),
        "cut": CutConfig(),
        "eval": EvalConfig(),
        "output_dir": Path("/tmp/output"),
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)  # type: ignore[arg-type]


def _make_context(
    config: PipelineConfig,
    raw: dict[str, object] | None = None,
) -> ValidationContext:
    return ValidationContext(
        config_path=Path("/fake/config.toml"),
        raw=raw or {},
        config=config,
    )


# ── _add_warning ─────────────────────────────────────────────────────


class TestAddWarning:
    def test_basic_warning(self) -> None:
        warnings: list[str] = []
        _add_warning(warnings, "HIGH", "something went wrong")
        assert len(warnings) == 1
        assert "[HIGH]" in warnings[0]
        assert "something went wrong" in warnings[0]

    def test_warning_with_fix(self) -> None:
        warnings: list[str] = []
        _add_warning(warnings, "LOW", "issue", fix="do this")
        assert "fix: do this" in warnings[0]

    def test_multiple_warnings(self) -> None:
        warnings: list[str] = []
        _add_warning(warnings, "HIGH", "a")
        _add_warning(warnings, "LOW", "b")
        assert len(warnings) == 2


# ── _validate_prompt_jsonl_file ──────────────────────────────────────


class TestValidatePromptJsonlFile:
    def test_missing_file(self, tmp_path: Path) -> None:
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            tmp_path / "missing.jsonl", "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result is None
        assert any("[HIGH]" in w for w in warnings)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result == 0
        assert any("empty" in w for w in warnings)

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.jsonl"
        lines = [json.dumps({"prompt": f"prompt {i}"}) for i in range(20)]
        f.write_text("\n".join(lines))
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result == 20
        assert len(warnings) == 0

    def test_small_file_warns(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.jsonl"
        f.write_text(json.dumps({"prompt": "one"}) + "\n")
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result == 1
        assert any("[LOW]" in w for w in warnings)

    def test_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n")
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result is None
        assert any("[HIGH]" in w for w in warnings)

    def test_non_object_line(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text('"just a string"\n')
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result is None
        assert any("JSON object" in w for w in warnings)

    def test_missing_prompt_key(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text(json.dumps({"text": "no prompt key"}) + "\n")
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result is None
        assert any("prompt" in w for w in warnings)

    def test_empty_prompt_value(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text(json.dumps({"prompt": ""}) + "\n")
        warnings: list[str] = []
        result = _validate_prompt_jsonl_file(
            f, "[data].harmful", warnings,
            min_recommended=16, missing_fix="fix it",
        )
        assert result is None


# ── _load_refusal_phrases ────────────────────────────────────────────


class TestLoadRefusalPhrases:
    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "phrases.txt"
        f.write_text("I cannot\nI will not\n# comment\n\n")
        phrases = _load_refusal_phrases(f)
        assert phrases == ["I cannot", "I will not"]

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("# only comments\n\n")
        with pytest.raises(ValueError, match="empty"):
            _load_refusal_phrases(f)


# ── _rule_output_dir ─────────────────────────────────────────────────


class TestRuleOutputDir:
    def test_file_not_dir_warns(self, tmp_path: Path) -> None:
        f = tmp_path / "output_file"
        f.write_text("I am a file")
        config = _make_config(output_dir=f)
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_output_dir(ctx, warnings)
        assert any("file, not a directory" in w for w in warnings)

    def test_dir_no_warning(self, tmp_path: Path) -> None:
        config = _make_config(output_dir=tmp_path)
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_output_dir(ctx, warnings)
        assert len(warnings) == 0

    def test_nonexistent_no_warning(self) -> None:
        config = _make_config(output_dir=Path("/nonexistent/dir"))
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_output_dir(ctx, warnings)
        assert len(warnings) == 0


# ── _rule_early_mode_conflicts ───────────────────────────────────────


class TestRuleEarlyModeConflicts:
    def test_no_early_modes_no_warning(self) -> None:
        config = _make_config()
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_early_mode_conflicts(ctx, warnings)
        assert len(warnings) == 0

    def test_single_early_mode_no_warning(self) -> None:
        config = _make_config(depth=DepthConfig(prompts=["test"]))
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_early_mode_conflicts(ctx, warnings)
        assert len(warnings) == 0

    def test_multiple_early_modes_warns(self) -> None:
        config = _make_config(
            depth=DepthConfig(prompts=["test"]),
            probe=ProbeConfig(prompts=["test"]),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_early_mode_conflicts(ctx, warnings)
        assert len(warnings) == 1
        assert "Multiple early-return modes" in warnings[0]


# ── _rule_depth_extract_direction ────────────────────────────────────


class TestRuleDepthExtractDirection:
    def test_no_depth_no_warning(self) -> None:
        config = _make_config()
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_depth_extract_direction(ctx, warnings)
        assert len(warnings) == 0

    def test_extract_false_no_warning(self) -> None:
        config = _make_config(
            depth=DepthConfig(prompts=["a", "b"], extract_direction=False),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_depth_extract_direction(ctx, warnings)
        assert len(warnings) == 0

    def test_too_few_prompts_warns(self) -> None:
        config = _make_config(
            depth=DepthConfig(prompts=["only_one"], extract_direction=True),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_depth_extract_direction(ctx, warnings)
        assert len(warnings) == 1
        assert "need >= 2" in warnings[0]

    def test_enough_prompts_no_warning(self) -> None:
        config = _make_config(
            depth=DepthConfig(prompts=["a", "b"], extract_direction=True),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_depth_extract_direction(ctx, warnings)
        assert len(warnings) == 0


# ── _rule_eval_without_prompts ───────────────────────────────────────


class TestRuleEvalWithoutPrompts:
    def test_eval_section_without_prompts_warns(self) -> None:
        config = _make_config(eval=EvalConfig())
        raw: dict[str, object] = {"eval": {"max_tokens": 100}}
        ctx = _make_context(config, raw)
        warnings: list[str] = []
        _rule_eval_without_prompts(ctx, warnings)
        assert len(warnings) == 1
        assert "eval" in warnings[0].lower()

    def test_eval_with_prompts_no_warning(self) -> None:
        config = _make_config(
            eval=EvalConfig(prompts_path=Path("eval.jsonl")),
        )
        raw: dict[str, object] = {"eval": {"prompts": "eval.jsonl"}}
        ctx = _make_context(config, raw)
        warnings: list[str] = []
        _rule_eval_without_prompts(ctx, warnings)
        assert len(warnings) == 0


# ── _rule_skipped_sections ───────────────────────────────────────────


class TestRuleSkippedSections:
    def test_no_early_mode_no_warning(self) -> None:
        config = _make_config(surface=SurfaceConfig(prompts_path="default"))
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_skipped_sections(ctx, warnings)
        assert len(warnings) == 0

    def test_depth_with_surface_warns(self) -> None:
        config = _make_config(
            depth=DepthConfig(prompts=["test"]),
            surface=SurfaceConfig(prompts_path="default"),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_skipped_sections(ctx, warnings)
        assert len(warnings) == 1
        assert "[surface]" in warnings[0]

    def test_depth_with_detect_warns(self) -> None:
        config = _make_config(
            depth=DepthConfig(prompts=["test"]),
            detect=DetectConfig(),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_skipped_sections(ctx, warnings)
        assert len(warnings) == 1
        assert "[detect]" in warnings[0]

    def test_cast_with_eval_warns(self) -> None:
        config = _make_config(
            cast=CastConfig(prompts=["test"], threshold=0.1),
            eval=EvalConfig(prompts_path=Path("eval.jsonl")),
        )
        ctx = _make_context(config)
        warnings: list[str] = []
        _rule_skipped_sections(ctx, warnings)
        assert len(warnings) == 1
        assert "[eval]" in warnings[0]
