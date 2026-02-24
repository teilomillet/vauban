"""Tests for vauban.dequantize: quantization detection and dequantization."""

import mlx.core as mx
import mlx.nn as nn

from tests.conftest import MockCausalLM
from vauban.dequantize import dequantize_model, is_quantized

# QuantizedLinear needs input_dims divisible by group_size (min 32).
# Create a slightly larger model for quantization tests.
_Q_D_MODEL = 64
_Q_NUM_LAYERS = 2
_Q_VOCAB_SIZE = 32
_Q_NUM_HEADS = 2


def _make_quantized_model() -> MockCausalLM:
    """Create a model with dimensions large enough for quantization."""
    model = MockCausalLM(_Q_D_MODEL, _Q_NUM_LAYERS, _Q_VOCAB_SIZE, _Q_NUM_HEADS)
    mx.eval(model.parameters())
    return model


def _make_quantized_linear(in_dim: int, out_dim: int) -> nn.QuantizedLinear:
    """Create a QuantizedLinear with default group_size=32."""
    return nn.QuantizedLinear(
        in_dim, out_dim, bias=False, group_size=32, bits=4,
    )


class TestIsQuantized:
    def test_normal_model_not_quantized(
        self, mock_model: MockCausalLM,
    ) -> None:
        assert is_quantized(mock_model) is False

    def test_model_with_quantized_linear(self) -> None:
        """A model containing nn.QuantizedLinear is detected as quantized."""
        model = _make_quantized_model()

        quantized = _make_quantized_linear(_Q_D_MODEL, _Q_D_MODEL)
        mx.eval(quantized.parameters())
        model.model.layers[0].self_attn.o_proj = quantized

        assert is_quantized(model) is True


class TestDequantizeModel:
    def test_returns_false_when_nothing_quantized(
        self, mock_model: MockCausalLM,
    ) -> None:
        assert dequantize_model(mock_model) is False

    def test_replaces_quantized_linear(self) -> None:
        """QuantizedLinear layers are replaced with Linear, weights become float."""
        model = _make_quantized_model()

        quantized = _make_quantized_linear(_Q_D_MODEL, _Q_D_MODEL)
        mx.eval(quantized.parameters())
        model.model.layers[0].self_attn.o_proj = quantized

        assert is_quantized(model) is True

        changed = dequantize_model(model)
        assert changed is True

        # The replaced layer should be nn.Linear now
        new_layer = model.model.layers[0].self_attn.o_proj
        assert isinstance(new_layer, nn.Linear)
        assert new_layer.weight.dtype != mx.uint32

        # Model should no longer be quantized
        assert is_quantized(model) is False
