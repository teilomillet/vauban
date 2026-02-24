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
