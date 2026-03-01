"""Tests for vauban.measure: subspace extraction via SVD."""

from pathlib import Path

import mlx.core as mx

from tests.conftest import NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.measure import measure_subspace, measure_subspace_bank


class TestMeasureSubspace:
    def test_returns_valid_subspace_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        assert result.basis.shape[0] == 2
        assert result.d_model > 0
        assert 0 <= result.layer_index < NUM_LAYERS
        assert len(result.singular_values) == 2
        assert len(result.explained_variance) == 2
        assert len(result.per_layer_bases) == NUM_LAYERS

    def test_basis_is_orthonormal(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        gram = result.basis @ result.basis.T
        mx.eval(gram)
        identity = mx.eye(2)
        diff = float(mx.linalg.norm(gram - identity).item())
        assert diff < 1e-3

    def test_singular_values_decreasing(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=3,
        )

        for i in range(len(result.singular_values) - 1):
            assert result.singular_values[i] >= result.singular_values[i + 1] - 1e-6

    def test_best_direction_compat(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        direction = result.best_direction()
        assert direction.direction.shape == (result.d_model,)
        assert direction.layer_index == result.layer_index

    def test_top_k_capped_by_prompts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """If top_k > num_prompts, actual k should be capped."""
        harmful = ["bad one", "bad two"]
        harmless = ["good one", "good two"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=10,
        )

        # Should get at most min(num_prompts, d_model) directions
        assert result.basis.shape[0] <= 2

    def test_explained_variance_sums_leq_one(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        total = sum(result.explained_variance)
        assert total <= 1.0 + 1e-6


class TestMeasureSubspaceBank:
    def test_returns_named_results(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        entries = [
            ("safety", ["bad one", "bad two"], ["good one", "good two"]),
            ("format", ["format bad", "format bad2"], ["format ok", "format ok2"]),
        ]
        results = measure_subspace_bank(
            mock_model, mock_tokenizer, entries, top_k=2,
        )
        assert set(results.keys()) == {"safety", "format"}
        for _name, result in results.items():
            assert result.basis.shape[0] == 2
            assert result.d_model > 0
            assert len(result.per_layer_bases) == NUM_LAYERS

    def test_single_entry_matches_direct_call(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        direct = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )
        bank = measure_subspace_bank(
            mock_model, mock_tokenizer,
            [("test", harmful, harmless)], top_k=2,
        )
        assert "test" in bank
        assert bank["test"].layer_index == direct.layer_index
        assert bank["test"].d_model == direct.d_model


class TestBankConfigParse:
    def test_parse_bank_entries(self, tmp_path: Path) -> None:
        target = tmp_path / "safety_harmful.jsonl"
        opposite = tmp_path / "safety_harmless.jsonl"
        target.write_text('{"prompt": "bad"}\n')
        opposite.write_text('{"prompt": "good"}\n')

        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            "[measure]\nmode = \"subspace\"\n"
            "bank = [\n"
            '  { name = "safety", '
            f'harmful = "{target.name}", '
            f'harmless = "{opposite.name}" }},\n'
            '  { name = "format", '
            'harmful = "default", harmless = "default" },\n'
            "]\n"
        )
        from vauban.config import load_config

        config = load_config(toml_file)
        assert config.measure.mode == "subspace"
        assert len(config.measure.bank) == 2
        assert config.measure.bank[0].name == "safety"
        assert config.measure.bank[0].harmful == target.name
        assert config.measure.bank[1].name == "format"
        assert config.measure.bank[1].harmful == "default"

    def test_parse_bank_absent(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            "[measure]\nmode = \"subspace\"\n"
        )
        from vauban.config import load_config

        config = load_config(toml_file)
        assert config.measure.bank == []
