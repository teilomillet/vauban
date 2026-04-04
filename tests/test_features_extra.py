# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for `vauban.features` branch coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.features import (
    SparseAutoencoder,
    load_sae,
    train_sae,
    train_sae_multi_layer,
)
from vauban.types import FeaturesResult, SAELayerResult

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _FakeWeight:
    shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _FakeEmbedTokens:
    weight: _FakeWeight


@dataclass(frozen=True, slots=True)
class _FakeTransformer:
    embed_tokens: _FakeEmbedTokens
    layers: list[object]


class TestTrainSae:
    def test_empty_activations_count_all_features_dead(self) -> None:
        sae = SparseAutoencoder(D_MODEL, 4)
        activations = ops.zeros((0, D_MODEL))
        ops.eval(activations)

        result = train_sae(
            sae,
            activations,
            n_epochs=0,
            batch_size=1,
        )

        assert result.n_dead_features == 4
        assert result.n_active_features == 0
        assert result.loss_history == []
        assert result.final_loss == 0.0


class TestTrainSaeMultiLayer:
    def test_without_direction_skips_alignment(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_acts = [ops.ones((2, 4)), ops.zeros((2, 4))]
        ops.eval(*fake_acts)

        monkeypatch.setattr(
            "vauban._forward.get_transformer",
            lambda *_args, **_kwargs: _FakeTransformer(
                embed_tokens=_FakeEmbedTokens(weight=_FakeWeight(shape=(16, 4))),
                layers=[object(), object()],
            ),
        )
        monkeypatch.setattr(
            "vauban.features._collect_per_prompt_activations",
            lambda *args, **kwargs: fake_acts,
        )
        monkeypatch.setattr(
            "vauban.features.train_sae",
            lambda *args, **kwargs: SAELayerResult(
                layer=-1,
                final_loss=0.5,
                loss_history=[1.0, 0.5],
                n_dead_features=1,
                n_active_features=3,
            ),
        )

        saes, result = train_sae_multi_layer(
            mock_model,
            mock_tokenizer,
            ["prompt-a", "prompt-b"],
            layers=[0, 1],
            d_sae=4,
            n_epochs=1,
            batch_size=1,
            model_path="test-model",
        )

        assert list(saes) == [0, 1]
        assert isinstance(result, FeaturesResult)
        assert result.direction_alignment is None
        assert [layer.layer for layer in result.layers] == [0, 1]

    def test_with_direction_collects_alignment(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_acts = [ops.ones((2, 4)), ops.zeros((2, 4))]
        ops.eval(*fake_acts)
        direction = ops.ones((4,))
        ops.eval(direction)

        monkeypatch.setattr(
            "vauban._forward.get_transformer",
            lambda *_args, **_kwargs: _FakeTransformer(
                embed_tokens=_FakeEmbedTokens(weight=_FakeWeight(shape=(16, 4))),
                layers=[object(), object()],
            ),
        )
        monkeypatch.setattr(
            "vauban.features._collect_per_prompt_activations",
            lambda *args, **kwargs: fake_acts,
        )
        monkeypatch.setattr(
            "vauban.features.train_sae",
            lambda *args, **kwargs: SAELayerResult(
                layer=-1,
                final_loss=0.5,
                loss_history=[1.0, 0.5],
                n_dead_features=1,
                n_active_features=3,
            ),
        )

        saes, result = train_sae_multi_layer(
            mock_model,
            mock_tokenizer,
            ["prompt-a", "prompt-b"],
            layers=[0, 1],
            d_sae=4,
            n_epochs=1,
            batch_size=1,
            direction=direction,
            model_path="test-model",
        )

        assert list(saes) == [0, 1]
        assert isinstance(result, FeaturesResult)
        assert result.direction_alignment is not None
        assert len(result.direction_alignment) == 2
        assert len(result.direction_alignment[0]) == 4


class TestLoadSae:
    def test_rejects_non_dict_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            "vauban.features.ops.load",
            lambda *_args, **_kwargs: ["not", "a", "dict"],
        )

        with pytest.raises(ValueError, match="Expected dict"):
            load_sae(tmp_path / "broken.safetensors", d_model=4, d_sae=4)
