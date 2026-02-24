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
        assert config.eval_prompts_path == fixtures_dir / "eval.jsonl"
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
            '[softprompt]\ntoken_constraint = "emoji"\n'
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
