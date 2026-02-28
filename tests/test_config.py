"""Tests for vauban.config: TOML parsing, validation, path resolution."""

from pathlib import Path

import pytest

from vauban.config import load_config
from vauban.types import DatasetRef


class TestLoadConfig:
    def test_minimal_config(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.model_path == "mlx-community/tiny-llama"
        assert config.harmful_path == fixtures_dir / "harmful.jsonl"
        assert config.harmless_path == fixtures_dir / "harmless.jsonl"
        assert config.eval.prompts_path == fixtures_dir / "eval.jsonl"
        assert config.output_dir == fixtures_dir / "output"

    def test_cut_defaults(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.cut.alpha == 1.0
        assert config.cut.layers is None  # "auto" -> None
        assert config.cut.norm_preserve is False

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("[data]\nharmful = 'x'\nharmless = 'y'\n")
        with pytest.raises(ValueError, match="model"):
            load_config(toml_file)

    def test_missing_data_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[model]\npath = "test"\n')
        with pytest.raises(ValueError, match="data"):
            load_config(toml_file)

    def test_layers_list(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cut]\nlayers = [14, 15]\nalpha = 0.8\n"
        )
        config = load_config(toml_file)
        assert config.cut.layers == [14, 15]
        assert config.cut.alpha == 0.8

    def test_path_resolution_relative(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "sub" / "config.toml"
        toml_file.parent.mkdir()
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = '../harmful.jsonl'\nharmless = '../harmless.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.harmful_path == tmp_path / "sub" / ".." / "harmful.jsonl"

    def test_default_data_paths(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
        )
        config = load_config(toml_file)
        from vauban.measure import default_prompt_paths

        expected_harmful, expected_harmless = default_prompt_paths()
        assert config.harmful_path == expected_harmful
        assert config.harmless_path == expected_harmless

    def test_measure_defaults(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.measure.mode == "direction"
        assert config.measure.top_k == 5

    def test_measure_subspace_mode(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmode = "subspace"\ntop_k = 10\n'
        )
        config = load_config(toml_file)
        assert config.measure.mode == "subspace"
        assert config.measure.top_k == 10

    def test_measure_only_flag(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmode = "diff"\ndiff_model = "base"\nmeasure_only = true\n'
        )
        config = load_config(toml_file)
        assert config.measure.measure_only is True

    def test_measure_invalid_mode_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmode = "invalid"\n'
        )
        with pytest.raises(ValueError, match="subspace"):
            load_config(toml_file)

    def test_hf_short_form(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "hf:mlabonne/harmful_behaviors"\n'
            'harmless = "default"\n'
        )
        config = load_config(toml_file)
        assert isinstance(config.harmful_path, DatasetRef)
        assert config.harmful_path.repo_id == "mlabonne/harmful_behaviors"
        assert config.harmful_path.split == "train"
        assert config.harmful_path.column == "prompt"
        assert config.harmful_path.limit is None

    def test_hf_table_form(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmless = "default"\n'
            "[data.harmful]\n"
            'hf = "JailbreakBench/JBB-Behaviors"\n'
            'split = "harmful"\n'
            'column = "Goal"\n'
            "limit = 200\n"
        )
        config = load_config(toml_file)
        assert isinstance(config.harmful_path, DatasetRef)
        assert config.harmful_path.repo_id == "JailbreakBench/JBB-Behaviors"
        assert config.harmful_path.split == "harmful"
        assert config.harmful_path.column == "Goal"
        assert config.harmful_path.limit == 200

    def test_hf_mixed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\n"
            'harmful = "hf:mlabonne/harmful_behaviors"\n'
            'harmless = "local.jsonl"\n'
        )
        config = load_config(toml_file)
        assert isinstance(config.harmful_path, DatasetRef)
        assert config.harmful_path.repo_id == "mlabonne/harmful_behaviors"
        assert config.harmless_path == tmp_path / "local.jsonl"

    def test_layer_strategy_defaults(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.cut.layer_strategy == "all"
        assert config.cut.layer_top_k == 10

    def test_layer_strategy_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\nlayer_strategy = "top_k"\nlayer_top_k = 5\n'
        )
        config = load_config(toml_file)
        assert config.cut.layer_strategy == "top_k"
        assert config.cut.layer_top_k == 5

    def test_layer_strategy_invalid_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\nlayer_strategy = "bogus"\n'
        )
        with pytest.raises(ValueError, match="layer_strategy"):
            load_config(toml_file)

    def test_layer_weights_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cut]\nlayer_weights = [0.5, 1.0, 1.5]\n"
        )
        config = load_config(toml_file)
        assert config.cut.layer_weights == [0.5, 1.0, 1.5]

    def test_layer_weights_default_none(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.cut.layer_weights is None

    def test_sparsity_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cut]\nsparsity = 0.5\n"
        )
        config = load_config(toml_file)
        assert config.cut.sparsity == 0.5

    def test_sparsity_default_zero(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.cut.sparsity == 0.0

    def test_sparsity_out_of_range_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cut]\nsparsity = 1.5\n"
        )
        with pytest.raises(ValueError, match="sparsity"):
            load_config(toml_file)

    def test_clip_quantile_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[measure]\nclip_quantile = 0.01\n"
        )
        config = load_config(toml_file)
        assert config.measure.clip_quantile == 0.01

    def test_clip_quantile_default_zero(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.measure.clip_quantile == 0.0

    def test_clip_quantile_out_of_range_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[measure]\nclip_quantile = 0.6\n"
        )
        with pytest.raises(ValueError, match="clip_quantile"):
            load_config(toml_file)

    def test_surface_default_prompts(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[surface]\nprompts = "default"\n'
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.prompts_path == "default"
        assert config.surface.generate is True
        assert config.surface.max_tokens == 20

    def test_surface_custom_path(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[surface]\nprompts = "my_prompts.jsonl"\ngenerate = false\n'
            "max_tokens = 50\n"
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.prompts_path == tmp_path / "my_prompts.jsonl"
        assert config.surface.generate is False
        assert config.surface.max_tokens == 50

    def test_surface_gate_thresholds_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[surface]\n"
            "max_worst_cell_refusal_after = 0.25\n"
            "max_worst_cell_refusal_delta = 0.10\n"
            "min_coverage_score = 0.60\n"
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.max_worst_cell_refusal_after == 0.25
        assert config.surface.max_worst_cell_refusal_delta == 0.10
        assert config.surface.min_coverage_score == 0.60

    def test_surface_gate_threshold_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[surface]\nmax_worst_cell_refusal_after = "high"\n'
        )
        with pytest.raises(TypeError, match="max_worst_cell_refusal_after"):
            load_config(toml_file)

    def test_surface_gate_threshold_range_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[surface]\nmin_coverage_score = 1.2\n"
        )
        with pytest.raises(ValueError, match="min_coverage_score"):
            load_config(toml_file)

    def test_surface_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.surface is None

    def test_surface_prompts_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[surface]\nprompts = 42\n"
        )
        with pytest.raises(TypeError, match="prompts"):
            load_config(toml_file)

    def test_surface_implicit_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[surface]\n"
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.prompts_path == "default"
        assert config.surface.generate is True
        assert config.surface.max_tokens == 20

    def test_measure_dbdi_mode(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmode = "dbdi"\n'
        )
        config = load_config(toml_file)
        assert config.measure.mode == "dbdi"

    def test_dbdi_target_defaults_to_red(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.cut.dbdi_target == "red"

    def test_dbdi_target_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\ndbdi_target = "both"\n'
        )
        config = load_config(toml_file)
        assert config.cut.dbdi_target == "both"

    def test_dbdi_target_invalid_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\ndbdi_target = "invalid"\n'
        )
        with pytest.raises(ValueError, match="dbdi_target"):
            load_config(toml_file)

    def test_false_refusal_ortho_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            'borderline = "b.jsonl"\n'
            "[cut]\nfalse_refusal_ortho = true\n"
        )
        config = load_config(toml_file)
        assert config.cut.false_refusal_ortho is True
        assert config.borderline_path == tmp_path / "b.jsonl"

    def test_false_refusal_ortho_default_false(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.cut.false_refusal_ortho is False

    def test_false_refusal_ortho_missing_borderline_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cut]\nfalse_refusal_ortho = true\n"
        )
        with pytest.raises(ValueError, match="borderline"):
            load_config(toml_file)

    def test_borderline_path_default_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.borderline_path is None

    def test_detect_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[detect]\n"
        )
        config = load_config(toml_file)
        assert config.detect is not None
        assert config.detect.mode == "full"
        assert config.detect.top_k == 5
        assert config.detect.alpha == 1.0
        assert config.detect.max_tokens == 100
        assert config.detect.clip_quantile == 0.0

    def test_detect_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[detect]\nmode = "probe"\ntop_k = 3\nalpha = 0.5\n'
            "max_tokens = 50\n"
        )
        config = load_config(toml_file)
        assert config.detect is not None
        assert config.detect.mode == "probe"
        assert config.detect.top_k == 3
        assert config.detect.alpha == 0.5
        assert config.detect.max_tokens == 50

    def test_detect_invalid_mode_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[detect]\nmode = "invalid"\n'
        )
        with pytest.raises(ValueError, match="mode"):
            load_config(toml_file)

    def test_detect_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.detect is None

    def test_layer_type_filter_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.cut.layer_type_filter is None

    def test_layer_type_filter_global(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\nlayer_type_filter = "global"\n'
        )
        config = load_config(toml_file)
        assert config.cut.layer_type_filter == "global"

    def test_layer_type_filter_sliding(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\nlayer_type_filter = "sliding"\n'
        )
        config = load_config(toml_file)
        assert config.cut.layer_type_filter == "sliding"

    def test_layer_type_filter_invalid_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cut]\nlayer_type_filter = "invalid"\n'
        )
        with pytest.raises(ValueError, match="layer_type_filter"):
            load_config(toml_file)

    def test_optimize_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[optimize]\n"
        )
        config = load_config(toml_file)
        assert config.optimize is not None
        assert config.optimize.n_trials == 50
        assert config.optimize.alpha_min == 0.1
        assert config.optimize.alpha_max == 5.0
        assert config.optimize.sparsity_min == 0.0
        assert config.optimize.sparsity_max == 0.9
        assert config.optimize.search_norm_preserve is True
        assert config.optimize.search_strategies == [
            "all", "above_median", "top_k",
        ]
        assert config.optimize.layer_top_k_min == 3
        assert config.optimize.layer_top_k_max is None
        assert config.optimize.max_tokens == 100
        assert config.optimize.seed is None
        assert config.optimize.timeout is None

    def test_optimize_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[optimize]\n"
            "n_trials = 100\n"
            "alpha_min = 0.5\n"
            "alpha_max = 3.0\n"
            "sparsity_min = 0.1\n"
            "sparsity_max = 0.8\n"
            "search_norm_preserve = false\n"
            'search_strategies = ["all", "top_k"]\n'
            "layer_top_k_min = 5\n"
            "layer_top_k_max = 20\n"
            "max_tokens = 50\n"
            "seed = 42\n"
            "timeout = 3600\n"
        )
        config = load_config(toml_file)
        assert config.optimize is not None
        assert config.optimize.n_trials == 100
        assert config.optimize.alpha_min == 0.5
        assert config.optimize.alpha_max == 3.0
        assert config.optimize.sparsity_min == 0.1
        assert config.optimize.sparsity_max == 0.8
        assert config.optimize.search_norm_preserve is False
        assert config.optimize.search_strategies == ["all", "top_k"]
        assert config.optimize.layer_top_k_min == 5
        assert config.optimize.layer_top_k_max == 20
        assert config.optimize.max_tokens == 50
        assert config.optimize.seed == 42
        assert config.optimize.timeout == 3600.0

    def test_optimize_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.optimize is None

    def test_optimize_invalid_n_trials_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[optimize]\nn_trials = 0\n"
        )
        with pytest.raises(ValueError, match="n_trials"):
            load_config(toml_file)

    def test_optimize_alpha_range_validation(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[optimize]\nalpha_min = 5.0\nalpha_max = 1.0\n"
        )
        with pytest.raises(ValueError, match="alpha_min"):
            load_config(toml_file)

    def test_softprompt_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.mode == "continuous"
        assert config.softprompt.n_tokens == 16
        assert config.softprompt.n_steps == 200
        assert config.softprompt.learning_rate == 0.01
        assert config.softprompt.init_scale == 0.1
        assert config.softprompt.batch_size == 64
        assert config.softprompt.top_k == 256
        assert config.softprompt.direction_weight == 0.0
        assert config.softprompt.target_prefixes == ["Sure", "Here"]
        assert config.softprompt.max_gen_tokens == 100
        assert config.softprompt.seed is None

    def test_softprompt_gcg_mode(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
            'mode = "gcg"\n'
            "n_tokens = 32\n"
            "n_steps = 100\n"
            "batch_size = 128\n"
            "top_k = 512\n"
            "direction_weight = 0.5\n"
            'target_prefixes = ["OK", "Yes"]\n'
            "seed = 42\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.mode == "gcg"
        assert config.softprompt.n_tokens == 32
        assert config.softprompt.n_steps == 100
        assert config.softprompt.batch_size == 128
        assert config.softprompt.top_k == 512
        assert config.softprompt.direction_weight == 0.5
        assert config.softprompt.target_prefixes == ["OK", "Yes"]
        assert config.softprompt.seed == 42

    def test_softprompt_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is None

    def test_softprompt_invalid_mode_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nmode = "invalid"\n'
        )
        with pytest.raises(ValueError, match="mode"):
            load_config(toml_file)

    def test_softprompt_invalid_n_tokens_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nn_tokens = 0\n"
        )
        with pytest.raises(ValueError, match="n_tokens"):
            load_config(toml_file)

    # -- embed_reg_weight --

    def test_softprompt_embed_reg_weight_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.embed_reg_weight == 0.0

    def test_softprompt_embed_reg_weight_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nembed_reg_weight = 0.5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.embed_reg_weight == 0.5

    def test_softprompt_embed_reg_weight_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nembed_reg_weight = -0.1\n"
        )
        with pytest.raises(ValueError, match="embed_reg_weight"):
            load_config(toml_file)

    # -- patience --

    def test_softprompt_patience_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.patience == 0

    def test_softprompt_patience_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\npatience = 10\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.patience == 10

    def test_softprompt_patience_negative_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\npatience = -1\n"
        )
        with pytest.raises(ValueError, match="patience"):
            load_config(toml_file)

    # -- lr_schedule --

    def test_softprompt_lr_schedule_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.lr_schedule == "constant"

    def test_softprompt_lr_schedule_cosine(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nlr_schedule = "cosine"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.lr_schedule == "cosine"

    def test_softprompt_lr_schedule_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nlr_schedule = "linear"\n'
        )
        with pytest.raises(ValueError, match="lr_schedule"):
            load_config(toml_file)

    # -- n_restarts --

    def test_softprompt_n_restarts_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.n_restarts == 1

    def test_softprompt_n_restarts_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nn_restarts = 5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.n_restarts == 5

    def test_softprompt_n_restarts_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nn_restarts = 0\n"
        )
        with pytest.raises(ValueError, match="n_restarts"):
            load_config(toml_file)

    # -- prompt_strategy --

    def test_softprompt_prompt_strategy_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.prompt_strategy == "all"

    def test_softprompt_prompt_strategy_cycle(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nprompt_strategy = "cycle"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.prompt_strategy == "cycle"

    def test_softprompt_prompt_strategy_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nprompt_strategy = "random"\n'
        )
        with pytest.raises(ValueError, match="prompt_strategy"):
            load_config(toml_file)

    # -- direction_mode --

    def test_softprompt_direction_mode_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.direction_mode == "last"

    def test_softprompt_direction_mode_raid(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ndirection_mode = "raid"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.direction_mode == "raid"

    def test_softprompt_direction_mode_all_positions(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ndirection_mode = "all_positions"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.direction_mode == "all_positions"

    def test_softprompt_direction_mode_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ndirection_mode = "bogus"\n'
        )
        with pytest.raises(ValueError, match="direction_mode"):
            load_config(toml_file)

    # -- direction_layers --

    def test_softprompt_direction_layers_default_none(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.direction_layers is None

    def test_softprompt_direction_layers_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ndirection_layers = [0, 1, 5]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.direction_layers == [0, 1, 5]

    # -- loss_mode --

    def test_softprompt_loss_mode_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.loss_mode == "targeted"

    def test_softprompt_loss_mode_untargeted(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nloss_mode = "untargeted"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.loss_mode == "untargeted"

    def test_softprompt_loss_mode_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nloss_mode = "bogus"\n'
        )
        with pytest.raises(ValueError, match="loss_mode"):
            load_config(toml_file)

    # -- egd_temperature --

    def test_softprompt_egd_temperature_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.egd_temperature == 1.0

    def test_softprompt_egd_temperature_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\negd_temperature = 0.5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.egd_temperature == 0.5

    def test_softprompt_egd_temperature_zero_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\negd_temperature = 0\n"
        )
        with pytest.raises(ValueError, match="egd_temperature"):
            load_config(toml_file)

    def test_softprompt_egd_temperature_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\negd_temperature = -1.0\n"
        )
        with pytest.raises(ValueError, match="egd_temperature"):
            load_config(toml_file)

    # -- mode = "egd" --

    def test_softprompt_egd_mode(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nmode = "egd"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.mode == "egd"

    # -- token_constraint --

    def test_softprompt_token_constraint_default_none(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.token_constraint is None

    def test_softprompt_token_constraint_ascii(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ntoken_constraint = "ascii"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.token_constraint == "ascii"

    def test_softprompt_token_constraint_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ntoken_constraint = "bogus"\n'
        )
        with pytest.raises(ValueError, match="token_constraint"):
            load_config(toml_file)

    # -- eos_loss_mode --

    def test_softprompt_eos_loss_mode_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.eos_loss_mode == "none"

    def test_softprompt_eos_loss_mode_force(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\neos_loss_mode = "force"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.eos_loss_mode == "force"

    def test_softprompt_eos_loss_mode_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\neos_loss_mode = "invalid"\n'
        )
        with pytest.raises(ValueError, match="eos_loss_mode"):
            load_config(toml_file)

    # -- eos_loss_weight --

    def test_softprompt_eos_loss_weight_default(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.eos_loss_weight == 0.0

    def test_softprompt_eos_loss_weight_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\neos_loss_weight = -0.1\n"
        )
        with pytest.raises(ValueError, match="eos_loss_weight"):
            load_config(toml_file)

    # -- kl_ref_weight --

    def test_softprompt_kl_ref_weight_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.kl_ref_weight == 0.0

    def test_softprompt_kl_ref_weight_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nkl_ref_weight = -0.1\n"
        )
        with pytest.raises(ValueError, match="kl_ref_weight"):
            load_config(toml_file)

    # -- ref_model --

    def test_softprompt_ref_model_default_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.ref_model is None

    def test_softprompt_ref_model_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nref_model = "mlx-community/tiny-llama"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.ref_model == "mlx-community/tiny-llama"

    def test_softprompt_ref_model_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nref_model = 42\n"
        )
        with pytest.raises(TypeError, match="ref_model"):
            load_config(toml_file)

    def test_softprompt_kl_ref_weight_requires_ref_model(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nkl_ref_weight = 0.5\n"
        )
        with pytest.raises(ValueError, match="ref_model"):
            load_config(toml_file)

    def test_softprompt_kl_ref_weight_with_ref_model_ok(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
            "kl_ref_weight = 0.5\n"
            'ref_model = "mlx-community/tiny-llama"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.kl_ref_weight == 0.5
        assert config.softprompt.ref_model == "mlx-community/tiny-llama"

    # =========================================================================
    # SIC config tests
    # =========================================================================

    def test_sic_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\n"
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.mode == "direction"
        assert config.sic.threshold == 0.0
        assert config.sic.max_iterations == 3
        assert config.sic.max_tokens == 100
        assert config.sic.target_layer is None
        assert config.sic.max_sanitize_tokens == 200
        assert config.sic.block_on_failure is True

    def test_sic_full_parse(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\n"
            'mode = "generation"\n'
            "threshold = -0.5\n"
            "max_iterations = 5\n"
            "max_tokens = 50\n"
            "target_layer = 14\n"
            'sanitize_system_prompt = "Custom prompt"\n'
            "max_sanitize_tokens = 100\n"
            "block_on_failure = false\n"
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.mode == "generation"
        assert config.sic.threshold == -0.5
        assert config.sic.max_iterations == 5
        assert config.sic.max_tokens == 50
        assert config.sic.target_layer == 14
        assert config.sic.sanitize_system_prompt == "Custom prompt"
        assert config.sic.max_sanitize_tokens == 100
        assert config.sic.block_on_failure is False

    def test_sic_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.sic is None

    def test_sic_invalid_mode_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[sic]\nmode = "invalid"\n'
        )
        with pytest.raises(ValueError, match="mode"):
            load_config(toml_file)

    def test_sic_invalid_max_iterations_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\nmax_iterations = 0\n"
        )
        with pytest.raises(ValueError, match="max_iterations"):
            load_config(toml_file)

    def test_sic_invalid_max_tokens_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\nmax_tokens = 0\n"
        )
        with pytest.raises(ValueError, match="max_tokens"):
            load_config(toml_file)

    def test_sic_invalid_max_sanitize_tokens_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\nmax_sanitize_tokens = 0\n"
        )
        with pytest.raises(ValueError, match="max_sanitize_tokens"):
            load_config(toml_file)

    def test_sic_mode_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\nmode = 42\n"
        )
        with pytest.raises(TypeError, match="mode"):
            load_config(toml_file)

    def test_sic_custom_system_prompt(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[sic]\nsanitize_system_prompt = "Remove bad stuff"\n'
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.sanitize_system_prompt == "Remove bad stuff"

    def test_sic_target_layer(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\ntarget_layer = 7\n"
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.target_layer == 7

    def test_sic_calibrate_default_false(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\n"
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.calibrate is False

    def test_sic_calibrate_parsed_true(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\ncalibrate = true\n"
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.calibrate is True

    def test_sic_calibrate_prompts_default_harmless(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\n"
        )
        config = load_config(toml_file)
        assert config.sic is not None
        assert config.sic.calibrate_prompts == "harmless"

    def test_sic_calibrate_prompts_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[sic]\ncalibrate_prompts = "invalid"\n'
        )
        with pytest.raises(ValueError, match="calibrate_prompts"):
            load_config(toml_file)

    def test_sic_calibrate_prompts_type_error(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[sic]\ncalibrate_prompts = 42\n"
        )
        with pytest.raises(TypeError, match="calibrate_prompts"):
            load_config(toml_file)

    # =========================================================================
    # Worst-K config tests
    # =========================================================================

    def test_softprompt_worst_k_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.worst_k == 5

    def test_softprompt_worst_k_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nworst_k = 10\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.worst_k == 10

    def test_softprompt_worst_k_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\nworst_k = 0\n"
        )
        with pytest.raises(ValueError, match="worst_k"):
            load_config(toml_file)

    def test_softprompt_prompt_strategy_worst_k(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\nprompt_strategy = "worst_k"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.prompt_strategy == "worst_k"

    # =========================================================================
    # Gradient accumulation config tests
    # =========================================================================

    def test_softprompt_grad_accum_steps_default(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.grad_accum_steps == 1

    def test_softprompt_grad_accum_steps_parsed(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngrad_accum_steps = 4\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.grad_accum_steps == 4

    def test_softprompt_grad_accum_steps_zero_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngrad_accum_steps = 0\n"
        )
        with pytest.raises(ValueError, match="grad_accum_steps"):
            load_config(toml_file)

    # =========================================================================
    # Transfer models config tests
    # =========================================================================

    def test_softprompt_transfer_models_default(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.transfer_models == []

    def test_softprompt_transfer_models_parsed(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ntransfer_models = ["model-a", "model-b"]\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.transfer_models == ["model-a", "model-b"]

    def test_softprompt_transfer_models_type_error(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ntransfer_models = "not-a-list"\n'
        )
        with pytest.raises(TypeError, match="transfer_models"):
            load_config(toml_file)

    # =========================================================================
    # Depth config tests
    # =========================================================================

    def test_depth_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["What is 2+2?"]\n'
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.prompts == ["What is 2+2?"]
        assert config.depth.settling_threshold == 0.5
        assert config.depth.deep_fraction == 0.85
        assert config.depth.top_k_logits == 1000
        assert config.depth.max_tokens == 0
        assert config.depth.extract_direction is False
        assert config.depth.direction_prompts is None

    def test_depth_full_parse(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["p1", "p2"]\n'
            "settling_threshold = 0.3\n"
            "deep_fraction = 0.9\n"
            "top_k_logits = 500\n"
            "max_tokens = 10\n"
            "extract_direction = true\n"
            'direction_prompts = ["d1", "d2"]\n'
            "clip_quantile = 0.05\n"
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.prompts == ["p1", "p2"]
        assert config.depth.settling_threshold == 0.3
        assert config.depth.deep_fraction == 0.9
        assert config.depth.top_k_logits == 500
        assert config.depth.max_tokens == 10
        assert config.depth.extract_direction is True
        assert config.depth.direction_prompts == ["d1", "d2"]
        assert config.depth.clip_quantile == 0.05

    def test_depth_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.depth is None

    def test_depth_missing_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_depth_empty_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\nprompts = []\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_depth_settling_threshold_out_of_range_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "settling_threshold = 1.5\n"
        )
        with pytest.raises(ValueError, match="settling_threshold"):
            load_config(toml_file)

    def test_depth_settling_threshold_zero_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "settling_threshold = 0.0\n"
        )
        with pytest.raises(ValueError, match="settling_threshold"):
            load_config(toml_file)

    def test_depth_top_k_logits_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "top_k_logits = 0\n"
        )
        with pytest.raises(ValueError, match="top_k_logits"):
            load_config(toml_file)

    def test_depth_max_tokens_negative_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "max_tokens = -1\n"
        )
        with pytest.raises(ValueError, match="max_tokens"):
            load_config(toml_file)

    def test_depth_clip_quantile_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.clip_quantile == 0.0

    def test_depth_clip_quantile_out_of_range_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "clip_quantile = 0.5\n"
        )
        with pytest.raises(ValueError, match="clip_quantile"):
            load_config(toml_file)

    def test_depth_clip_quantile_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "clip_quantile = -0.1\n"
        )
        with pytest.raises(ValueError, match="clip_quantile"):
            load_config(toml_file)

    def test_depth_extract_direction_too_few_prompts_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["only_one"]\n'
            "extract_direction = true\n"
        )
        with pytest.raises(ValueError, match=r"extract_direction.*>= 2"):
            load_config(toml_file)

    def test_depth_extract_direction_too_few_direction_prompts_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["a", "b", "c"]\n'
            "extract_direction = true\n"
            'direction_prompts = ["only_one"]\n'
        )
        with pytest.raises(ValueError, match=r"extract_direction.*>= 2"):
            load_config(toml_file)

    def test_depth_extract_direction_enough_prompts_ok(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["a", "b"]\n'
            "extract_direction = true\n"
        )
        config = load_config(toml_file)
        assert config.depth is not None
        assert config.depth.extract_direction is True

    def test_depth_mode_conflict(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[depth]\n"
            'prompts = ["test"]\n'
            "[probe]\n"
            'prompts = ["test"]\n'
        )
        from vauban import validate

        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    # =========================================================================
    # Eval config tests
    # =========================================================================

    def test_eval_max_tokens_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.eval.max_tokens == 100

    def test_eval_max_tokens_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[eval]\nmax_tokens = 200\n"
        )
        config = load_config(toml_file)
        assert config.eval.max_tokens == 200

    def test_eval_max_tokens_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[eval]\nmax_tokens = 0\n"
        )
        with pytest.raises(ValueError, match="max_tokens"):
            load_config(toml_file)

    def test_eval_num_prompts_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.eval.num_prompts == 20

    def test_eval_num_prompts_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[eval]\nnum_prompts = 50\n"
        )
        config = load_config(toml_file)
        assert config.eval.num_prompts == 50

    def test_eval_num_prompts_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[eval]\nnum_prompts = 0\n"
        )
        with pytest.raises(ValueError, match="num_prompts"):
            load_config(toml_file)

    def test_eval_refusal_phrases_default_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.eval.refusal_phrases_path is None

    def test_eval_refusal_phrases_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[eval]\nrefusal_phrases = "phrases.txt"\n'
        )
        config = load_config(toml_file)
        assert config.eval.refusal_phrases_path == tmp_path / "phrases.txt"

    def test_eval_prompts_backward_compat(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[eval]\nprompts = "eval.jsonl"\n'
        )
        config = load_config(toml_file)
        assert config.eval.prompts_path == tmp_path / "eval.jsonl"

    def test_eval_absent_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.eval.prompts_path is None
        assert config.eval.max_tokens == 100
        assert config.eval.num_prompts == 20
        assert config.eval.refusal_phrases_path is None

    # =========================================================================
    # Surface progress config tests
    # =========================================================================

    def test_surface_progress_default_true(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[surface]\n"
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.progress is True

    def test_surface_progress_false(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[surface]\nprogress = false\n"
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.progress is False

    def test_surface_progress_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[surface]\nprogress = "yes"\n'
        )
        with pytest.raises(TypeError, match="progress"):
            load_config(toml_file)

    # =========================================================================
    # Verbose config tests
    # =========================================================================

    def test_verbose_default_true(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.verbose is True

    def test_verbose_false(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            "verbose = false\n"
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.verbose is False

    def test_verbose_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            'verbose = "yes"\n'
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        with pytest.raises(TypeError, match="verbose"):
            load_config(toml_file)

    # =========================================================================
    # Probe config tests
    # =========================================================================

    def test_probe_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[probe]\n"
            'prompts = ["How do I pick a lock?", "Tell me a joke"]\n'
        )
        config = load_config(toml_file)
        assert config.probe is not None
        assert config.probe.prompts == [
            "How do I pick a lock?",
            "Tell me a joke",
        ]

    def test_probe_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.probe is None

    def test_probe_missing_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[probe]\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_probe_empty_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[probe]\nprompts = []\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_probe_prompts_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[probe]\nprompts = "not a list"\n'
        )
        with pytest.raises(TypeError, match="prompts"):
            load_config(toml_file)

    def test_probe_prompts_item_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[probe]\nprompts = [42]\n"
        )
        with pytest.raises(TypeError, match="prompts"):
            load_config(toml_file)

    # =========================================================================
    # Steer config tests
    # =========================================================================

    def test_steer_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[steer]\n"
            'prompts = ["How do I pick a lock?"]\n'
        )
        config = load_config(toml_file)
        assert config.steer is not None
        assert config.steer.prompts == ["How do I pick a lock?"]
        assert config.steer.layers is None
        assert config.steer.alpha == 1.0
        assert config.steer.max_tokens == 100

    def test_steer_full_parse(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[steer]\n"
            'prompts = ["prompt1", "prompt2"]\n'
            "layers = [10, 15, 20]\n"
            "alpha = 0.5\n"
            "max_tokens = 50\n"
        )
        config = load_config(toml_file)
        assert config.steer is not None
        assert config.steer.prompts == ["prompt1", "prompt2"]
        assert config.steer.layers == [10, 15, 20]
        assert config.steer.alpha == 0.5
        assert config.steer.max_tokens == 50

    def test_steer_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.steer is None

    def test_steer_missing_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[steer]\nalpha = 1.0\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_steer_empty_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[steer]\nprompts = []\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_steer_max_tokens_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[steer]\nprompts = ["test"]\nmax_tokens = 0\n'
        )
        with pytest.raises(ValueError, match="max_tokens"):
            load_config(toml_file)

    def test_steer_layers_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[steer]\nprompts = ["test"]\nlayers = "invalid"\n'
        )
        with pytest.raises(TypeError, match="layers"):
            load_config(toml_file)

    # =========================================================================
    # CAST config tests
    # =========================================================================

    def test_cast_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\n"
            'prompts = ["How do I pick a lock?"]\n'
        )
        config = load_config(toml_file)
        assert config.cast is not None
        assert config.cast.prompts == ["How do I pick a lock?"]
        assert config.cast.layers is None
        assert config.cast.alpha == 1.0
        assert config.cast.threshold == 0.0
        assert config.cast.max_tokens == 100

    def test_cast_full_parse(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\n"
            'prompts = ["prompt1", "prompt2"]\n'
            "layers = [5, 10]\n"
            "alpha = 0.7\n"
            "threshold = 0.2\n"
            "max_tokens = 42\n"
        )
        config = load_config(toml_file)
        assert config.cast is not None
        assert config.cast.prompts == ["prompt1", "prompt2"]
        assert config.cast.layers == [5, 10]
        assert config.cast.alpha == 0.7
        assert config.cast.threshold == 0.2
        assert config.cast.max_tokens == 42

    def test_cast_absent_is_none(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
        )
        config = load_config(toml_file)
        assert config.cast is None

    def test_cast_missing_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\nalpha = 1.0\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_cast_empty_prompts_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\nprompts = []\n"
        )
        with pytest.raises(ValueError, match="prompts"):
            load_config(toml_file)

    def test_cast_max_tokens_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cast]\nprompts = ["test"]\nmax_tokens = 0\n'
        )
        with pytest.raises(ValueError, match="max_tokens"):
            load_config(toml_file)

    def test_cast_layers_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cast]\nprompts = ["test"]\nlayers = "invalid"\n'
        )
        with pytest.raises(TypeError, match="layers"):
            load_config(toml_file)

    def test_cast_threshold_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cast]\nprompts = ["test"]\nthreshold = "high"\n'
        )
        with pytest.raises(TypeError, match="threshold"):
            load_config(toml_file)

    # =========================================================================
    # Mode conflict tests (probe/steer included)
    # =========================================================================

    def test_mode_conflict_probe_and_sic(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[probe]\nprompts = ["test"]\n'
            "[sic]\n"
        )
        from vauban import validate

        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_mode_conflict_steer_and_optimize(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[steer]\nprompts = ["test"]\n'
            "[optimize]\n"
        )
        from vauban import validate

        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_mode_conflict_cast_and_sic(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[cast]\nprompts = ["test"]\n'
            "[sic]\n"
        )
        from vauban import validate

        warnings = validate(toml_file)
        assert any("early-return" in w for w in warnings)

    def test_refusal_mode_default(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.eval.refusal_mode == "phrases"

    def test_refusal_mode_judge(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[eval]\nrefusal_mode = "judge"\n'
        )
        config = load_config(toml_file)
        assert config.eval.refusal_mode == "judge"

    def test_refusal_mode_invalid_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[eval]\nrefusal_mode = "invalid"\n'
        )
        with pytest.raises(ValueError, match="refusal_mode"):
            load_config(toml_file)

    def test_transfer_models_default(self, fixtures_dir: Path) -> None:
        config = load_config(fixtures_dir / "config.toml")
        assert config.measure.transfer_models == []

    def test_transfer_models_list(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\ntransfer_models = ["model-a", "model-b"]\n'
        )
        config = load_config(toml_file)
        assert config.measure.transfer_models == ["model-a", "model-b"]

    def test_surface_default_multilingual(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[surface]\nprompts = "default_multilingual"\n'
        )
        config = load_config(toml_file)
        assert config.surface is not None
        assert config.surface.prompts_path == "default_multilingual"

    # -- measure diff mode --

    def test_measure_diff_mode(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmode = "diff"\ndiff_model = "base-model"\n'
        )
        config = load_config(toml_file)
        assert config.measure.mode == "diff"
        assert config.measure.diff_model == "base-model"

    def test_measure_only_must_be_boolean(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmeasure_only = "yes"\n'
        )
        with pytest.raises(TypeError, match="measure_only"):
            load_config(toml_file)

    def test_measure_diff_without_diff_model_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[measure]\nmode = "diff"\n'
        )
        with pytest.raises(ValueError, match="diff_model"):
            load_config(toml_file)

    # -- cast condition_direction --

    def test_cast_condition_direction_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\n"
            'prompts = ["test prompt"]\n'
            'condition_direction = "hdd.npy"\n'
        )
        config = load_config(toml_file)
        assert config.cast is not None
        assert config.cast.condition_direction_path == "hdd.npy"

    # -- cast alpha_tiers --

    def test_cast_alpha_tiers_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\n"
            'prompts = ["test prompt"]\n'
            "[[cast.alpha_tiers]]\n"
            "threshold = 0.0\n"
            "alpha = 0.5\n"
            "[[cast.alpha_tiers]]\n"
            "threshold = 1.0\n"
            "alpha = 2.5\n"
        )
        config = load_config(toml_file)
        assert config.cast is not None
        assert config.cast.alpha_tiers is not None
        assert len(config.cast.alpha_tiers) == 2
        assert config.cast.alpha_tiers[0].threshold == 0.0
        assert config.cast.alpha_tiers[0].alpha == 0.5
        assert config.cast.alpha_tiers[1].threshold == 1.0
        assert config.cast.alpha_tiers[1].alpha == 2.5

    def test_cast_alpha_tiers_unsorted_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[cast]\n"
            'prompts = ["test prompt"]\n'
            "[[cast.alpha_tiers]]\n"
            "threshold = 1.0\n"
            "alpha = 2.5\n"
            "[[cast.alpha_tiers]]\n"
            "threshold = 0.0\n"
            "alpha = 0.5\n"
        )
        with pytest.raises(ValueError, match="sorted"):
            load_config(toml_file)


class TestGanConfigParsing:
    """Tests for GAN loop TOML config parsing."""

    def test_gan_rounds_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_rounds == 0

    def test_gan_rounds_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_rounds = 5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_rounds == 5

    def test_gan_step_multiplier_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_step_multiplier = 2.0\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_step_multiplier == 2.0

    def test_gan_direction_escalation_parsed(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_direction_escalation = 0.5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_direction_escalation == 0.5

    def test_gan_token_escalation_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_token_escalation = 8\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_token_escalation == 8

    def test_gan_rounds_negative_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_rounds = -1\n"
        )
        with pytest.raises(ValueError, match="gan_rounds"):
            load_config(toml_file)

    def test_gan_step_multiplier_zero_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_step_multiplier = 0\n"
        )
        with pytest.raises(ValueError, match="gan_step_multiplier"):
            load_config(toml_file)

    def test_defense_eval_sic_mode_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ndefense_eval_sic_mode = "generation"\n'
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.defense_eval_sic_mode == "generation"

    def test_defense_eval_sic_mode_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ndefense_eval_sic_mode = "bogus"\n'
        )
        with pytest.raises(ValueError, match="defense_eval_sic_mode"):
            load_config(toml_file)

    def test_defense_eval_sic_max_iterations_parsed(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ndefense_eval_sic_max_iterations = 5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.defense_eval_sic_max_iterations == 5

    def test_defense_eval_cast_layers_parsed(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ndefense_eval_cast_layers = [5, 10, 15]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.defense_eval_cast_layers == [5, 10, 15]

    def test_init_tokens_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ninit_tokens = [100, 200, 300]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.init_tokens == [100, 200, 300]

    def test_gan_multiturn_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_multiturn is False
        assert config.softprompt.gan_multiturn_max_turns == 10

    def test_gan_multiturn_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_multiturn = true\n"
            "gan_multiturn_max_turns = 5\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_multiturn is True
        assert config.softprompt.gan_multiturn_max_turns == 5

    def test_gan_multiturn_max_turns_zero_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_multiturn_max_turns = 0\n"
        )
        with pytest.raises(ValueError, match="gan_multiturn_max_turns"):
            load_config(toml_file)

    def test_gan_multiturn_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ngan_multiturn = "yes"\n'
        )
        with pytest.raises(TypeError, match="gan_multiturn"):
            load_config(toml_file)

    # -- gan_defense_escalation --

    def test_gan_defense_escalation_default(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_defense_escalation is False
        assert config.softprompt.gan_defense_alpha_multiplier == 1.5
        assert config.softprompt.gan_defense_threshold_escalation == 0.5
        assert config.softprompt.gan_defense_sic_iteration_escalation == 1

    def test_gan_defense_escalation_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\n"
            "gan_defense_escalation = true\n"
            "gan_defense_alpha_multiplier = 2.0\n"
            "gan_defense_threshold_escalation = 0.3\n"
            "gan_defense_sic_iteration_escalation = 2\n"
        )
        config = load_config(toml_file)
        assert config.softprompt is not None
        assert config.softprompt.gan_defense_escalation is True
        assert config.softprompt.gan_defense_alpha_multiplier == 2.0
        assert config.softprompt.gan_defense_threshold_escalation == 0.3
        assert config.softprompt.gan_defense_sic_iteration_escalation == 2

    def test_gan_defense_escalation_type_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ngan_defense_escalation = "yes"\n'
        )
        with pytest.raises(TypeError, match="gan_defense_escalation"):
            load_config(toml_file)

    def test_gan_defense_alpha_multiplier_type_error(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ngan_defense_alpha_multiplier = "big"\n'
        )
        with pytest.raises(TypeError, match="gan_defense_alpha_multiplier"):
            load_config(toml_file)

    def test_gan_defense_alpha_multiplier_zero_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_defense_alpha_multiplier = 0\n"
        )
        with pytest.raises(ValueError, match="gan_defense_alpha_multiplier"):
            load_config(toml_file)

    def test_gan_defense_threshold_escalation_type_error(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ngan_defense_threshold_escalation = "low"\n'
        )
        with pytest.raises(
            TypeError, match="gan_defense_threshold_escalation",
        ):
            load_config(toml_file)

    def test_gan_defense_threshold_escalation_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_defense_threshold_escalation = -1\n"
        )
        with pytest.raises(
            ValueError, match="gan_defense_threshold_escalation",
        ):
            load_config(toml_file)

    def test_gan_defense_sic_iteration_escalation_type_error(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            '[softprompt]\ngan_defense_sic_iteration_escalation = 1.5\n'
        )
        with pytest.raises(
            TypeError, match="gan_defense_sic_iteration_escalation",
        ):
            load_config(toml_file)

    def test_gan_defense_sic_iteration_escalation_negative_raises(
        self, tmp_path: Path,
    ) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            "[data]\nharmful = 'h.jsonl'\nharmless = 'hl.jsonl'\n"
            "[softprompt]\ngan_defense_sic_iteration_escalation = -1\n"
        )
        with pytest.raises(
            ValueError, match="gan_defense_sic_iteration_escalation",
        ):
            load_config(toml_file)
