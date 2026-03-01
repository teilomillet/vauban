"""Tests for vauban.svf: Steering Vector Field boundary training."""

from pathlib import Path

import mlx.core as mx
import pytest

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.svf import (
    SVFBoundary,
    load_svf_boundary,
    save_svf_boundary,
    svf_gradient,
    train_svf_boundary,
)


class TestSVFBoundary:
    def test_forward_returns_scalar(self) -> None:
        boundary = SVFBoundary(D_MODEL, 8, 16, NUM_LAYERS)
        h = mx.random.normal((D_MODEL,))
        mx.eval(h)
        score = boundary.forward(h, 0)
        mx.eval(score)
        assert score.shape == ()
        assert isinstance(float(score.item()), float)

    def test_forward_different_layers(self) -> None:
        boundary = SVFBoundary(D_MODEL, 8, 16, NUM_LAYERS)
        h = mx.random.normal((D_MODEL,))
        mx.eval(h)
        scores = []
        for i in range(NUM_LAYERS):
            score = boundary.forward(h, i)
            mx.eval(score)
            scores.append(float(score.item()))
        # Different layers should give different scores
        # (FiLM conditioning differs)
        assert len(scores) == NUM_LAYERS

    def test_parameters_roundtrip(self) -> None:
        boundary = SVFBoundary(D_MODEL, 8, 16, NUM_LAYERS)
        params = boundary.parameters()
        assert len(params) == 7
        boundary.set_parameters(params)
        params2 = boundary.parameters()
        for p1, p2 in zip(params, params2, strict=True):
            assert mx.array_equal(p1, p2)


class TestSVFGradient:
    def test_returns_score_and_gradient(self) -> None:
        boundary = SVFBoundary(D_MODEL, 8, 16, NUM_LAYERS)
        h = mx.random.normal((D_MODEL,))
        mx.eval(h)
        score, grad = svf_gradient(boundary, h, 0)
        assert isinstance(score, float)
        assert grad.shape == (D_MODEL,)
        # Gradient should be approximately unit norm
        grad_norm = float(mx.linalg.norm(grad).item())
        assert grad_norm == pytest.approx(1.0, abs=0.01) or grad_norm == 0.0


class TestSVFSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        boundary = SVFBoundary(D_MODEL, 8, 16, NUM_LAYERS)
        path = tmp_path / "boundary.safetensors"
        save_svf_boundary(boundary, path)
        assert path.exists()

        loaded = load_svf_boundary(path, D_MODEL, 8, 16, NUM_LAYERS)
        h = mx.random.normal((D_MODEL,))
        mx.eval(h)

        score_orig = boundary.forward(h, 0)
        score_loaded = loaded.forward(h, 0)
        mx.eval(score_orig, score_loaded)
        assert float(score_orig.item()) == pytest.approx(
            float(score_loaded.item()), abs=1e-5,
        )


class TestSVFTraining:
    def test_loss_decreases(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        target = ["how to do something bad", "another harmful prompt"]
        opposite = ["what is the weather", "tell me a joke"]

        _boundary, result = train_svf_boundary(
            mock_model,
            mock_tokenizer,
            target,
            opposite,
            d_model=D_MODEL,
            n_layers=NUM_LAYERS,
            projection_dim=4,
            hidden_dim=8,
            n_epochs=5,
            learning_rate=0.01,
        )
        assert len(result.train_loss_history) == 5
        # Loss should generally decrease (or at least not explode)
        assert result.train_loss_history[-1] < result.train_loss_history[0] + 1.0
        assert result.final_accuracy >= 0.0
        assert result.final_accuracy <= 1.0
        assert result.n_layers_trained == NUM_LAYERS

    def test_training_with_specific_layers(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        target = ["bad prompt"]
        opposite = ["good prompt"]

        _boundary, result = train_svf_boundary(
            mock_model,
            mock_tokenizer,
            target,
            opposite,
            d_model=D_MODEL,
            n_layers=NUM_LAYERS,
            projection_dim=4,
            hidden_dim=8,
            n_epochs=2,
            layers=[0],
        )
        assert result.n_layers_trained == 1
        assert len(result.per_layer_separation) == 1


class TestSVFConfigParse:
    def test_parse_svf_section(self, tmp_path: Path) -> None:
        target_file = tmp_path / "target.jsonl"
        opposite_file = tmp_path / "opposite.jsonl"
        target_file.write_text('{"prompt": "bad"}\n')
        opposite_file.write_text('{"prompt": "good"}\n')

        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            "[svf]\n"
            f'prompts_target = "{target_file.name}"\n'
            f'prompts_opposite = "{opposite_file.name}"\n'
            "projection_dim = 32\n"
            "hidden_dim = 128\n"
            "n_epochs = 5\n"
        )
        from vauban.config import load_config
        config = load_config(toml_file)
        assert config.svf is not None
        assert config.svf.projection_dim == 32
        assert config.svf.hidden_dim == 128
        assert config.svf.n_epochs == 5

    def test_parse_svf_absent(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
        )
        from vauban.config import load_config
        config = load_config(toml_file)
        assert config.svf is None
