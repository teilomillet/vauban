"""Tests for pipeline mode registry predicates and helpers."""

from pathlib import Path

from vauban.config._mode_registry import (
    EARLY_MODE_SPECS,
    EARLY_RETURN_PRECEDENCE,
    active_early_mode_for_phase,
    active_early_modes,
)
from vauban.types import (
    CastConfig,
    ComposeOptimizeConfig,
    CutConfig,
    DefenseStackConfig,
    DepthConfig,
    EvalConfig,
    OptimizeConfig,
    PipelineConfig,
    ProbeConfig,
    SICConfig,
    SoftPromptConfig,
    SteerConfig,
    SVFConfig,
)


def _minimal_config(**kwargs: object) -> PipelineConfig:
    """Build a PipelineConfig with only the specified overrides."""
    defaults: dict[str, object] = {
        "model_path": "test-model",
        "harmful_path": "harmful.jsonl",
        "harmless_path": "harmless.jsonl",
        "cut": CutConfig(),
        "measure": None,
        "eval": EvalConfig(),
    }
    defaults.update(kwargs)
    return PipelineConfig(**defaults)  # type: ignore[arg-type]


class TestEarlyModePredicates:
    """Test each _has_* predicate individually."""

    def test_has_depth(self) -> None:
        config = _minimal_config(depth=DepthConfig(prompts=["test"]))
        assert active_early_modes(config) == ["[depth]"]

    def test_has_probe(self) -> None:
        config = _minimal_config(probe=ProbeConfig(prompts=["test"]))
        assert "[probe]" in active_early_modes(config)

    def test_has_steer(self) -> None:
        config = _minimal_config(steer=SteerConfig(prompts=["test"], alpha=1.0))
        assert "[steer]" in active_early_modes(config)

    def test_has_cast(self) -> None:
        config = _minimal_config(cast=CastConfig(prompts=["test"], threshold=0.1))
        assert "[cast]" in active_early_modes(config)

    def test_has_sic(self) -> None:
        config = _minimal_config(sic=SICConfig())
        assert "[sic]" in active_early_modes(config)

    def test_has_optimize(self) -> None:
        config = _minimal_config(optimize=OptimizeConfig())
        assert "[optimize]" in active_early_modes(config)

    def test_has_compose_optimize(self) -> None:
        config = _minimal_config(
            compose_optimize=ComposeOptimizeConfig(bank_path="bank.npz"),
        )
        assert "[compose_optimize]" in active_early_modes(config)

    def test_has_softprompt(self) -> None:
        config = _minimal_config(softprompt=SoftPromptConfig(n_tokens=8))
        assert "[softprompt]" in active_early_modes(config)

    def test_has_svf(self) -> None:
        config = _minimal_config(
            svf=SVFConfig(
                prompts_target=Path("t.txt"),
                prompts_opposite=Path("o.txt"),
            ),
        )
        assert "[svf]" in active_early_modes(config)

    def test_has_defend(self) -> None:
        config = _minimal_config(defend=DefenseStackConfig())
        assert "[defend]" in active_early_modes(config)


class TestActiveEarlyModes:
    """Test active_early_modes with various configurations."""

    def test_no_early_modes(self) -> None:
        config = _minimal_config()
        assert active_early_modes(config) == []

    def test_multiple_modes_returns_all_in_precedence_order(self) -> None:
        config = _minimal_config(
            depth=DepthConfig(prompts=["test"]),
            probe=ProbeConfig(prompts=["test"]),
            cast=CastConfig(prompts=["test"], threshold=0.1),
        )
        modes = active_early_modes(config)
        assert modes == ["[depth]", "[probe]", "[cast]"]

    def test_precedence_order_depth_before_probe(self) -> None:
        config = _minimal_config(
            depth=DepthConfig(prompts=["test"]),
            probe=ProbeConfig(prompts=["test"]),
        )
        modes = active_early_modes(config)
        depth_idx = modes.index("[depth]")
        probe_idx = modes.index("[probe]")
        assert depth_idx < probe_idx


class TestActiveEarlyModeForPhase:
    """Test active_early_mode_for_phase routing."""

    def test_before_prompts_returns_depth(self) -> None:
        config = _minimal_config(depth=DepthConfig(prompts=["test"]))
        spec = active_early_mode_for_phase(config, "before_prompts")
        assert spec is not None
        assert spec.mode == "depth"

    def test_before_prompts_returns_svf(self) -> None:
        config = _minimal_config(
            svf=SVFConfig(
                prompts_target=Path("t.txt"),
                prompts_opposite=Path("o.txt"),
            ),
        )
        spec = active_early_mode_for_phase(config, "before_prompts")
        assert spec is not None
        assert spec.mode == "svf"

    def test_after_measure_returns_probe(self) -> None:
        config = _minimal_config(probe=ProbeConfig(prompts=["test"]))
        spec = active_early_mode_for_phase(config, "after_measure")
        assert spec is not None
        assert spec.mode == "probe"

    def test_no_match_returns_none(self) -> None:
        config = _minimal_config()
        assert active_early_mode_for_phase(config, "before_prompts") is None
        assert active_early_mode_for_phase(config, "after_measure") is None

    def test_phase_precedence_depth_over_svf(self) -> None:
        config = _minimal_config(
            depth=DepthConfig(prompts=["test"]),
            svf=SVFConfig(
                prompts_target=Path("t.txt"),
                prompts_opposite=Path("o.txt"),
            ),
        )
        spec = active_early_mode_for_phase(config, "before_prompts")
        assert spec is not None
        assert spec.mode == "depth"

    def test_phase_precedence_probe_over_steer(self) -> None:
        config = _minimal_config(
            probe=ProbeConfig(prompts=["test"]),
            steer=SteerConfig(prompts=["test"], alpha=1.0),
        )
        spec = active_early_mode_for_phase(config, "after_measure")
        assert spec is not None
        assert spec.mode == "probe"


class TestEarlyModeSpecs:
    """Test structural properties of the registry."""

    def test_all_specs_have_unique_sections(self) -> None:
        sections = [s.section for s in EARLY_MODE_SPECS]
        assert len(sections) == len(set(sections))

    def test_all_specs_have_unique_modes(self) -> None:
        modes = [s.mode for s in EARLY_MODE_SPECS]
        assert len(modes) == len(set(modes))

    def test_precedence_tuple_matches_specs(self) -> None:
        expected = tuple(s.section for s in EARLY_MODE_SPECS)
        assert expected == EARLY_RETURN_PRECEDENCE

    def test_requires_direction_flags(self) -> None:
        spec_map = {s.section: s for s in EARLY_MODE_SPECS}
        assert spec_map["[depth]"].requires_direction is False
        assert spec_map["[svf]"].requires_direction is False
        assert spec_map["[features]"].requires_direction is False
        assert spec_map["[probe]"].requires_direction is True
        assert spec_map["[steer]"].requires_direction is True
        assert spec_map["[cast]"].requires_direction is True
        assert spec_map["[sic]"].requires_direction is False
        assert spec_map["[optimize]"].requires_direction is True
        assert spec_map["[compose_optimize]"].requires_direction is False
        assert spec_map["[softprompt]"].requires_direction is False
        assert spec_map["[defend]"].requires_direction is False
        assert spec_map["[circuit]"].requires_direction is False

    def test_phase_assignments(self) -> None:
        spec_map = {s.section: s for s in EARLY_MODE_SPECS}
        assert spec_map["[depth]"].phase == "before_prompts"
        assert spec_map["[svf]"].phase == "before_prompts"
        assert spec_map["[features]"].phase == "before_prompts"
        assert spec_map["[probe]"].phase == "after_measure"
        assert spec_map["[steer]"].phase == "after_measure"
        assert spec_map["[cast]"].phase == "after_measure"
        assert spec_map["[sic]"].phase == "after_measure"
        assert spec_map["[optimize]"].phase == "after_measure"
        assert spec_map["[compose_optimize]"].phase == "after_measure"
        assert spec_map["[softprompt]"].phase == "after_measure"
        assert spec_map["[defend]"].phase == "after_measure"
        assert spec_map["[circuit]"].phase == "after_measure"
