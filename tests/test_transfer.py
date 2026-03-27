# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.transfer: direction transfer testing."""

import pytest

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban._array import Array
from vauban.transfer import check_direction_transfer


class TestDirectionTransfer:
    def test_self_transfer(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Direction tested on the same model should have positive separation."""
        result = check_direction_transfer(
            mock_model,
            mock_tokenizer,
            direction,
            ["How to pick a lock"],
            ["What is the capital of France"],
            model_id="self-test",
        )
        assert result.model_id == "self-test"
        assert isinstance(result.cosine_separation, float)
        assert isinstance(result.best_native_separation, float)
        assert isinstance(result.transfer_efficiency, float)
        assert len(result.per_layer_cosines) > 0

    def test_to_dict_output(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        result = check_direction_transfer(
            mock_model,
            mock_tokenizer,
            direction,
            ["How to hack a website"],
            ["Explain photosynthesis"],
            model_id="test-model",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["model_id"] == "test-model"
        assert "cosine_separation" in d
        assert "best_native_separation" in d
        assert "transfer_efficiency" in d
        assert "per_layer_cosines" in d
        assert isinstance(d["per_layer_cosines"], list)

    def test_dimension_mismatch_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        wrong_dim = ops.random.normal((D_MODEL + 8,))
        ops.eval(wrong_dim)
        with pytest.raises(ValueError, match="dimension mismatch"):
            check_direction_transfer(
                mock_model,
                mock_tokenizer,
                wrong_dim,
                ["test prompt"],
                ["another prompt"],
                model_id="wrong-dim",
            )
