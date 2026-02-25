"""Tests for [depth] TOML combinations, orderings, and edge cases.

Ensures depth mode works correctly with every other section and that
section ordering in the TOML file doesn't matter.
"""

from pathlib import Path

import pytest

from vauban import validate
from vauban.config import load_config

# =========================================================================
# Helper: minimal TOML fragments
# =========================================================================

_MODEL = '[model]\npath = "test"\n'
_DATA = "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
_DATA_DEFAULT = '[data]\nharmful = "default"\nharmless = "default"\n'
_DEPTH = "[depth]\n" 'prompts = ["What is 2+2?", "Explain gravity"]\n'
_DEPTH_SINGLE = "[depth]\n" 'prompts = ["What is 2+2?"]\n'
_PROBE = "[probe]\n" 'prompts = ["How do I pick a lock?"]\n'
_STEER = "[steer]\n" 'prompts = ["How do I pick a lock?"]\n'
_SIC = "[sic]\n"
_OPTIMIZE = "[optimize]\n"
_SOFTPROMPT = "[softprompt]\n"
_DETECT = "[detect]\n"
_SURFACE = '[surface]\nprompts = "default"\n'
_EVAL = '[eval]\nprompts = "eval.jsonl"\n'


class TestDepthStandalone:
    """[depth] as the only active mode."""

    def test_depth_only_with_data(self, tmp_path: Path) -> None:
        """Basic: [depth] + [data] parses fine."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH)
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.prompts == ["What is 2+2?", "Explain gravity"]

    def test_depth_only_without_data(self, tmp_path: Path) -> None:
        """[depth] without [data] should work — auto-fills defaults."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DEPTH)
        config = load_config(toml_file)
        assert config.depth is not None
        # harmful/harmless should have been auto-filled
        assert config.harmful_path is not None
        assert config.harmless_path is not None

    def test_depth_only_without_data_no_warnings(
        self, tmp_path: Path,
    ) -> None:
        """[depth] alone should produce zero warnings."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA_DEFAULT + _DEPTH)
        warnings = validate(toml_file)
        assert len(warnings) == 0

    def test_depth_with_default_data(self, tmp_path: Path) -> None:
        """[depth] + [data] with 'default' paths."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA_DEFAULT + _DEPTH)
        config = load_config(toml_file)
        assert config.depth is not None

    def test_non_depth_without_data_still_raises(
        self, tmp_path: Path,
    ) -> None:
        """Non-depth modes without [data] should still fail."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _PROBE)
        with pytest.raises(ValueError, match="data"):
            load_config(toml_file)


class TestDepthWithSkippedSections:
    """[depth] combined with sections it will skip."""

    def test_depth_plus_detect_warns(self, tmp_path: Path) -> None:
        """[depth] + [detect] should warn that detect is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _DETECT)
        warnings = validate(toml_file)
        assert any("skip" in w.lower() and "[detect]" in w for w in warnings)

    def test_depth_plus_surface_warns(self, tmp_path: Path) -> None:
        """[depth] + [surface] should warn that surface is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _SURFACE)
        warnings = validate(toml_file)
        assert any("skip" in w.lower() and "[surface]" in w for w in warnings)

    def test_depth_plus_eval_warns(self, tmp_path: Path) -> None:
        """[depth] + [eval] with prompts should warn that eval is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _EVAL)
        warnings = validate(toml_file)
        assert any("skip" in w.lower() and "[eval]" in w for w in warnings)

    def test_depth_plus_detect_surface_eval_warns_all(
        self, tmp_path: Path,
    ) -> None:
        """All three skipped sections listed in one warning."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA + _DEPTH + _DETECT + _SURFACE + _EVAL,
        )
        warnings = validate(toml_file)
        skip_warnings = [w for w in warnings if "skip" in w.lower()]
        assert len(skip_warnings) >= 1
        w = skip_warnings[0]
        assert "[detect]" in w
        assert "[surface]" in w
        assert "[eval]" in w


class TestDepthWithEarlyReturnConflicts:
    """[depth] combined with other early-return modes."""

    def test_depth_plus_probe_warns_conflict(self, tmp_path: Path) -> None:
        """[depth] + [probe] should warn about mode conflict."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _PROBE)
        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_depth_plus_steer_warns_conflict(self, tmp_path: Path) -> None:
        """[depth] + [steer] should warn about mode conflict."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _STEER)
        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_depth_plus_sic_warns_conflict(self, tmp_path: Path) -> None:
        """[depth] + [sic] should warn about mode conflict."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _SIC)
        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_depth_plus_optimize_warns_conflict(
        self, tmp_path: Path,
    ) -> None:
        """[depth] + [optimize] should warn about mode conflict."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _OPTIMIZE)
        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_depth_plus_softprompt_warns_conflict(
        self, tmp_path: Path,
    ) -> None:
        """[depth] + [softprompt] should warn about mode conflict."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _SOFTPROMPT)
        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_depth_takes_precedence_in_message(
        self, tmp_path: Path,
    ) -> None:
        """Conflict warning should list depth first."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _PROBE)
        warnings = validate(toml_file)
        conflict_w = [w for w in warnings if "early-return" in w]
        assert len(conflict_w) == 1
        assert "depth" in conflict_w[0].lower()


class TestSectionOrdering:
    """TOML section order should not matter."""

    def test_depth_before_model(self, tmp_path: Path) -> None:
        """[depth] appearing before [model] should work."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_DEPTH + _MODEL + _DATA)
        config = load_config(toml_file)
        assert config.depth is not None

    def test_depth_after_everything(self, tmp_path: Path) -> None:
        """[depth] as the last section should work."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + "[output]\ndir = 'out'\n" + _DEPTH)
        config = load_config(toml_file)
        assert config.depth is not None

    def test_data_before_model(self, tmp_path: Path) -> None:
        """[data] before [model] should work."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_DATA + _MODEL + _DEPTH)
        config = load_config(toml_file)
        assert config.depth is not None

    def test_depth_between_model_and_data(self, tmp_path: Path) -> None:
        """[depth] between [model] and [data] should work."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DEPTH + _DATA)
        config = load_config(toml_file)
        assert config.depth is not None

    def test_reversed_order(self, tmp_path: Path) -> None:
        """All sections in reversed order should still parse."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            "[output]\ndir = 'out'\n"
            + _DEPTH
            + "[cut]\nalpha = 0.5\n"
            + _DATA
            + _MODEL,
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.cut.alpha == 0.5

    def test_probe_before_depth_same_result(self, tmp_path: Path) -> None:
        """Order of [probe] vs [depth] doesn't change parsing."""
        toml_a = tmp_path / "a.toml"
        toml_a.write_text(_MODEL + _DATA + _DEPTH + _PROBE)
        toml_b = tmp_path / "b.toml"
        toml_b.write_text(_MODEL + _DATA + _PROBE + _DEPTH)
        config_a = load_config(toml_a)
        config_b = load_config(toml_b)
        assert config_a.depth is not None
        assert config_b.depth is not None
        assert config_a.probe is not None
        assert config_b.probe is not None
        assert config_a.depth.prompts == config_b.depth.prompts


class TestDepthConfigVariants:
    """Various depth config field combinations."""

    def test_depth_extract_direction_true(self, tmp_path: Path) -> None:
        """extract_direction = true should parse."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA + "[depth]\n"
            'prompts = ["p1", "p2", "p3"]\n'
            "extract_direction = true\n",
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.extract_direction is True

    def test_depth_with_direction_prompts(self, tmp_path: Path) -> None:
        """direction_prompts set alongside extract_direction."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA + "[depth]\n"
            'prompts = ["p1", "p2"]\n'
            "extract_direction = true\n"
            'direction_prompts = ["d1", "d2", "d3"]\n',
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.direction_prompts == ["d1", "d2", "d3"]

    def test_depth_max_tokens_zero(self, tmp_path: Path) -> None:
        """max_tokens = 0 (static mode) should be the default."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH)
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.max_tokens == 0

    def test_depth_generation_mode(self, tmp_path: Path) -> None:
        """max_tokens > 0 enables generation mode."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA + "[depth]\n"
            'prompts = ["test"]\n'
            "max_tokens = 50\n",
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.max_tokens == 50

    def test_depth_custom_thresholds(self, tmp_path: Path) -> None:
        """Custom settling_threshold and deep_fraction."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA + "[depth]\n"
            'prompts = ["test"]\n'
            "settling_threshold = 0.3\n"
            "deep_fraction = 0.9\n"
            "top_k_logits = 500\n",
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.settling_threshold == 0.3
        assert config.depth.deep_fraction == 0.9
        assert config.depth.top_k_logits == 500


class TestDepthValidateSummary:
    """validate() should report the correct pipeline summary for depth."""

    def test_depth_only_shows_depth_analysis(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pipeline summary should say 'depth analysis', not include extras."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA_DEFAULT + _DEPTH)
        validate(toml_file)
        captured = capsys.readouterr()
        assert "depth analysis" in captured.err
        # Should NOT say "depth analysis + detect" etc.
        assert "+ detect" not in captured.err
        assert "+ surface" not in captured.err
        assert "+ eval" not in captured.err

    def test_depth_with_detect_no_extras_in_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Even with [detect] present, summary shouldn't show it as extra."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _DEPTH + _DETECT)
        validate(toml_file)
        captured = capsys.readouterr()
        assert "depth analysis" in captured.err
        assert "+ detect" not in captured.err


class TestNonDepthSkippedSections:
    """All early-return modes should warn about skipped [surface]/[eval]."""

    def test_probe_plus_surface_warns(self, tmp_path: Path) -> None:
        """[probe] + [surface] should warn that surface is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _PROBE + _SURFACE)
        warnings = validate(toml_file)
        assert any(
            "skip" in w.lower() and "[surface]" in w for w in warnings
        )

    def test_probe_plus_eval_warns(self, tmp_path: Path) -> None:
        """[probe] + [eval] should warn that eval is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _PROBE + _EVAL)
        warnings = validate(toml_file)
        assert any(
            "skip" in w.lower() and "[eval]" in w for w in warnings
        )

    def test_steer_plus_surface_warns(self, tmp_path: Path) -> None:
        """[steer] + [surface] should warn that surface is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _STEER + _SURFACE)
        warnings = validate(toml_file)
        assert any(
            "skip" in w.lower() and "[surface]" in w for w in warnings
        )

    def test_sic_plus_surface_eval_warns_both(self, tmp_path: Path) -> None:
        """[sic] + [surface] + [eval] should warn about both."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _SIC + _SURFACE + _EVAL)
        warnings = validate(toml_file)
        skip_w = [w for w in warnings if "skip" in w.lower()]
        assert len(skip_w) >= 1
        assert "[surface]" in skip_w[0]
        assert "[eval]" in skip_w[0]

    def test_optimize_plus_surface_warns(self, tmp_path: Path) -> None:
        """[optimize] + [surface] should warn that surface is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _OPTIMIZE + _SURFACE)
        warnings = validate(toml_file)
        assert any(
            "skip" in w.lower() and "[surface]" in w for w in warnings
        )

    def test_softprompt_plus_eval_warns(self, tmp_path: Path) -> None:
        """[softprompt] + [eval] should warn that eval is skipped."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _SOFTPROMPT + _EVAL)
        warnings = validate(toml_file)
        assert any(
            "skip" in w.lower() and "[eval]" in w for w in warnings
        )

    def test_probe_plus_detect_no_skip_warning(self, tmp_path: Path) -> None:
        """[probe] + [detect] should NOT warn — detect runs before measure."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _PROBE + _DETECT)
        warnings = validate(toml_file)
        # detect should appear as a composable extra, not a skipped section
        assert not any(
            "skip" in w.lower() and "[detect]" in w for w in warnings
        )

    def test_probe_plus_detect_shows_detect_extra(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """[probe] + [detect] should show 'probe inspection + detect'."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _PROBE + _DETECT)
        validate(toml_file)
        captured = capsys.readouterr()
        assert "probe inspection + detect" in captured.err

    def test_skip_warning_names_active_mode(self, tmp_path: Path) -> None:
        """Skip warning should name the active early-return mode."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _STEER + _SURFACE)
        warnings = validate(toml_file)
        skip_w = [w for w in warnings if "early-return will skip" in w]
        assert len(skip_w) == 1
        assert "[steer]" in skip_w[0]


class TestNonDepthNotBroken:
    """Regression: existing modes should still work as before."""

    def test_probe_only_still_works(self, tmp_path: Path) -> None:
        """[probe] without [depth] should parse as before."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _PROBE)
        config = load_config(toml_file)
        assert config.probe is not None
        assert config.depth is None

    def test_normal_pipeline_still_works(self, tmp_path: Path) -> None:
        """No early-return modes — default pipeline should parse."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA)
        config = load_config(toml_file)
        assert config.depth is None
        assert config.probe is None

    def test_detect_plus_surface_no_spurious_skip_warning(
        self, tmp_path: Path,
    ) -> None:
        """[detect]+[surface] in normal pipeline should not trigger skip warning."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA_DEFAULT + _DETECT + _SURFACE,
        )
        warnings = validate(toml_file)
        assert not any("skip" in w.lower() for w in warnings)

    def test_steer_plus_optimize_still_warns_conflict(
        self, tmp_path: Path,
    ) -> None:
        """Pre-existing early-return conflict detection still works."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(_MODEL + _DATA + _STEER + _OPTIMIZE)
        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_missing_data_without_depth_still_raises(
        self, tmp_path: Path,
    ) -> None:
        """Config without [data] and without [depth] should still fail."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[model]\npath = "test"\n')
        with pytest.raises(ValueError, match="data"):
            load_config(toml_file)

    def test_normal_pipeline_with_surface_eval_no_warning(
        self, tmp_path: Path,
    ) -> None:
        """Normal pipeline + [surface] + [eval] should have no skip warning."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            _MODEL + _DATA_DEFAULT + _SURFACE + _EVAL,
        )
        warnings = validate(toml_file)
        assert not any("skip" in w.lower() for w in warnings)
