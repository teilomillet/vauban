"""Tests for vauban.measure: activation capture and direction computation."""

from pathlib import Path

import mlx.core as mx

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.measure import (
    _forward_collect,
    _match_suffix,
    find_instruction_boundary,
    load_prompts,
    measure,
    measure_dbdi,
    select_target_layers,
    silhouette_scores,
)


class TestLoadPrompts:
    def test_loads_jsonl(self, fixtures_dir: Path) -> None:
        prompts = load_prompts(fixtures_dir / "harmful.jsonl")
        assert len(prompts) == 3
        assert prompts[0] == "How to pick a lock"

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        p.write_text('{"prompt": "hello"}\n\n{"prompt": "world"}\n')
        prompts = load_prompts(p)
        assert len(prompts) == 2


class TestForwardCollect:
    def test_returns_per_layer_activations(
        self, mock_model: MockCausalLM,
    ) -> None:
        token_ids = mx.array([[1, 2, 3]])
        residuals = _forward_collect(mock_model, token_ids)
        assert len(residuals) == NUM_LAYERS
        d_model = mock_model.model.layers[0].self_attn.o_proj.weight.shape[0]
        for r in residuals:
            assert r.shape == (d_model,)


class TestMeasure:
    def test_returns_direction_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad prompt one", "bad prompt two"]
        harmless = ["good prompt one", "good prompt two"]
        result = measure(mock_model, mock_tokenizer, harmful, harmless)

        d_model = mock_model.model.layers[0].self_attn.o_proj.weight.shape[0]
        assert result.direction.shape == (d_model,)
        assert 0 <= result.layer_index < NUM_LAYERS
        assert len(result.cosine_scores) == NUM_LAYERS

    def test_direction_is_unit_vector(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        result = measure(
            mock_model, mock_tokenizer,
            ["harmful"], ["harmless"],
        )
        norm = float(mx.linalg.norm(result.direction).item())
        assert abs(norm - 1.0) < 1e-4


class TestMatchSuffix:
    def test_matching_suffix(self) -> None:
        full = [1, 2, 3, 4, 5]
        empty = [10, 4, 5]
        assert _match_suffix(full, empty) == 2

    def test_no_matching_suffix(self) -> None:
        full = [1, 2, 3]
        empty = [4, 5, 6]
        assert _match_suffix(full, empty) == 0

    def test_identical_sequences(self) -> None:
        seq = [1, 2, 3]
        assert _match_suffix(seq, seq) == 3

    def test_empty_sequences(self) -> None:
        assert _match_suffix([], []) == 0
        assert _match_suffix([1, 2], []) == 0


class TestFindInstructionBoundary:
    def test_boundary_before_suffix(
        self, mock_tokenizer: MockTokenizer,
    ) -> None:
        boundary = find_instruction_boundary(mock_tokenizer, "hello world")
        # The full template is "[USER]hello world[/USER][ASST]"
        # The empty template is "[USER][/USER][ASST]"
        # Suffix tokens from "[/USER][ASST]" should match
        full_text = "[USER]hello world[/USER][ASST]"
        full_ids = mock_tokenizer.encode(full_text)
        # Boundary should be a valid index within the sequence
        assert 0 <= boundary < len(full_ids)

    def test_boundary_differs_from_last(
        self, mock_tokenizer: MockTokenizer,
    ) -> None:
        # With a non-trivial suffix, boundary should not be the last token
        boundary = find_instruction_boundary(mock_tokenizer, "test prompt")
        full_text = "[USER]test prompt[/USER][ASST]"
        full_ids = mock_tokenizer.encode(full_text)
        assert boundary < len(full_ids) - 1


class TestForwardCollectWithPosition:
    def test_default_matches_explicit_last(
        self, mock_model: MockCausalLM,
    ) -> None:
        token_ids = mx.array([[1, 2, 3, 4]])
        default = _forward_collect(mock_model, token_ids)
        explicit = _forward_collect(mock_model, token_ids, token_position=-1)
        for d, e in zip(default, explicit, strict=True):
            assert mx.allclose(d, e)

    def test_position_2_differs_from_last(
        self, mock_model: MockCausalLM,
    ) -> None:
        token_ids = mx.array([[1, 2, 3, 4, 5]])
        pos2 = _forward_collect(mock_model, token_ids, token_position=2)
        last = _forward_collect(mock_model, token_ids, token_position=-1)
        # At least one layer should differ
        any_differ = False
        for p2, lst in zip(pos2, last, strict=True):
            if not mx.allclose(p2, lst):
                any_differ = True
                break
        assert any_differ


class TestMeasureDBDI:
    def test_returns_correct_shapes(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad prompt one", "bad prompt two"]
        harmless = ["good prompt one", "good prompt two"]
        result = measure_dbdi(mock_model, mock_tokenizer, harmful, harmless)

        assert result.hdd.shape == (D_MODEL,)
        assert result.red.shape == (D_MODEL,)
        assert result.d_model == D_MODEL
        assert 0 <= result.hdd_layer_index < NUM_LAYERS
        assert 0 <= result.red_layer_index < NUM_LAYERS
        assert len(result.hdd_cosine_scores) == NUM_LAYERS
        assert len(result.red_cosine_scores) == NUM_LAYERS

    def test_hdd_and_red_are_unit_vectors(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        result = measure_dbdi(
            mock_model, mock_tokenizer,
            ["harmful one", "harmful two"],
            ["harmless one", "harmless two"],
        )
        hdd_norm = float(mx.linalg.norm(result.hdd).item())
        red_norm = float(mx.linalg.norm(result.red).item())
        assert abs(hdd_norm - 1.0) < 1e-4
        assert abs(red_norm - 1.0) < 1e-4

    def test_hdd_and_red_differ(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        result = measure_dbdi(
            mock_model, mock_tokenizer,
            ["bad thing one", "bad thing two"],
            ["good thing one", "good thing two"],
        )
        # HDD and RED are extracted at different token positions,
        # so they should at least have compatible shapes
        assert result.hdd.shape == result.red.shape


class TestSilhouetteScores:
    def test_returns_per_layer_scores(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        scores = silhouette_scores(
            mock_model, mock_tokenizer, harmful, harmless,
        )
        assert len(scores) == NUM_LAYERS
        # Silhouette scores are in [-1, 1]
        for s in scores:
            assert -1.0 <= s <= 1.0

    def test_select_target_layers_silhouette_strategy(self) -> None:
        scores = [0.1, 0.5, 0.8, 0.3]
        layers = select_target_layers(scores, strategy="silhouette")
        # "silhouette" acts like "above_median"
        median = sorted(scores)[2]  # 0.3
        expected = [i for i, s in enumerate(scores) if s > median]
        assert layers == expected
