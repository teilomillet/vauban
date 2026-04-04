# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Branch coverage tests for ``vauban._forward``."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import vauban._backend as backend_module
import vauban._forward as forward_module
from vauban import _ops as ops

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, LayerCache, Tokenizer, TransformerModel


class BadTokenizer:
    """Tokenizer stub that violates the chat-template contract."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
    ) -> list[int]:
        """Return a non-string object."""
        del messages, tokenize
        return [1, 2, 3]

    def encode(self, text: str) -> list[int]:
        """Unused fallback for interface completeness."""
        del text
        return [1]


class FakeLayer:
    """Minimal layer stub for transformer-path tests."""

    def __init__(self, delta: Array, *, is_linear: bool = False) -> None:
        self.delta = delta
        self.is_linear = is_linear
        self.calls: list[tuple[Array, Array | None, object | None]] = []

    def __call__(
        self,
        h: Array,
        mask: Array | None,
        *,
        cache: object | None = None,
    ) -> Array:
        """Record the call and add the configured delta."""
        self.calls.append((h, mask, cache))
        return h + self.delta


class FakeTransformer:
    """Transformer stub exposing layers."""

    def __init__(self, layers: list[FakeLayer]) -> None:
        self.layers = layers


class FakeEmbedTokens:
    """Embedding stub with a tied ``as_linear`` projection."""

    def __init__(self, offset: Array) -> None:
        self.offset = offset

    def as_linear(self, h: Array) -> Array:
        """Return a deterministic projection."""
        return h + self.offset


class FakePromptCacheModule(ModuleType):
    """Typed fake ``mlx_lm.models.cache`` module."""

    def __init__(self) -> None:
        super().__init__("mlx_lm.models.cache")
        self.calls: list[object] = []

    def make_prompt_cache(self, model: object) -> list[str]:
        """Return a deterministic cache payload."""
        self.calls.append(model)
        return ["prompt-cache"]


class TorchForwardTensor:
    """Minimal tensor type for the reloaded torch backend."""

    def __init__(self, data: object, device: str = "cpu") -> None:
        self.data = np.array(data, dtype=float)
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        """Expose the tensor shape."""
        return tuple(int(dim) for dim in self.data.shape)

    def to(self, device: str) -> TorchForwardTensor:
        """Move the tensor to a new device."""
        return TorchForwardTensor(self.data.copy(), device=device)

    def float(self) -> TorchForwardTensor:
        """Cast to float32."""
        return TorchForwardTensor(
            self.data.astype(np.float32),
            device=self.device,
        )

    def __getitem__(
        self,
        item: int | slice | tuple[int | slice, ...],
    ) -> TorchForwardTensor:
        """Support slicing."""
        return TorchForwardTensor(self.data[item], device=self.device)

    def __add__(self, other: TorchForwardTensor) -> TorchForwardTensor:
        """Elementwise addition."""
        return TorchForwardTensor(self.data + other.data, device=self.device)


class TorchForwardLinalgModule(ModuleType):
    """Typed fake ``torch.linalg`` module for reloaded forward tests."""

    def __init__(self) -> None:
        super().__init__("torch.linalg")

    def svd(
        self,
        matrix: TorchForwardTensor,
    ) -> tuple[TorchForwardTensor, TorchForwardTensor, TorchForwardTensor]:
        """Compute an SVD."""
        u, s, vt = np.linalg.svd(matrix.data, full_matrices=True)
        return (
            TorchForwardTensor(u, device=matrix.device),
            TorchForwardTensor(s, device=matrix.device),
            TorchForwardTensor(vt, device=matrix.device),
        )

    def qr(
        self,
        matrix: TorchForwardTensor,
    ) -> tuple[TorchForwardTensor, TorchForwardTensor]:
        """Compute a QR decomposition."""
        q, r = np.linalg.qr(matrix.data)
        return (
            TorchForwardTensor(q, device=matrix.device),
            TorchForwardTensor(r, device=matrix.device),
        )


class TorchForwardModule(ModuleType):
    """Typed fake ``torch`` module for reloaded forward tests."""

    Tensor = TorchForwardTensor

    def __init__(self) -> None:
        super().__init__("torch")
        self.linalg = TorchForwardLinalgModule()

    def cat(
        self,
        tensors: list[TorchForwardTensor],
        *,
        dim: int = 0,
    ) -> TorchForwardTensor:
        """Concatenate tensors."""
        return TorchForwardTensor(
            np.concatenate([tensor.data for tensor in tensors], axis=dim),
            device=tensors[0].device,
        )


class TorchForwardFunctionalModule(ModuleType):
    """Typed fake ``torch.nn.functional`` module for reloaded forward tests."""

    def __init__(self) -> None:
        super().__init__("torch.nn.functional")
        self.linear_calls: list[tuple[TorchForwardTensor, TorchForwardTensor]] = []

    def linear(
        self,
        hidden: TorchForwardTensor,
        weight: TorchForwardTensor,
    ) -> TorchForwardTensor:
        """Project hidden states through a weight matrix."""
        self.linear_calls.append((hidden, weight))
        return TorchForwardTensor(
            np.matmul(hidden.data, weight.data.T),
            device=hidden.device,
        )


class TorchForwardNNModule(ModuleType):
    """Typed fake ``torch.nn`` package."""

    functional: TorchForwardFunctionalModule


class TorchForwardEmbedTokens:
    """Embedding stub for the torch backend reload path."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.weight = TorchForwardTensor(np.ones((3, 2)), device=device)

    def __call__(self, token_ids: TorchForwardTensor) -> TorchForwardTensor:
        """Expand token IDs into a trivial embedding tensor."""
        expanded = np.repeat(token_ids.data[..., None], 2, axis=2)
        return TorchForwardTensor(expanded, device=self.weight.device)


class TorchForwardTransformer:
    """Transformer stub for the torch backend reload path."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.embed_tokens = TorchForwardEmbedTokens(device=device)


class TorchForwardModel:
    """Model stub for the torch backend reload path."""

    def __init__(self) -> None:
        self.lm_head = lambda h: TorchForwardTensor(h.data + 5.0, device=h.device)
        self.cache_calls = 0

    def make_cache(self) -> list[str]:
        """Return a deterministic cache payload."""
        self.cache_calls += 1
        return ["torch-cache"]


class TestForwardBranches:
    """Additional branch coverage for ``vauban._forward``."""

    def test_encode_chat_prompt_requires_string_template(self) -> None:
        with pytest.raises(
            TypeError,
            match="apply_chat_template must return str",
        ):
            forward_module.encode_chat_prompt(
                cast("Tokenizer", BadTokenizer()),
                [{"role": "user", "content": "hi"}],
            )

    def test_make_ssm_mask_and_select_mask_cover_linear_case(self) -> None:
        transformer = FakeTransformer(
            [
                FakeLayer(ops.zeros((1, 3, 2))),
                FakeLayer(ops.zeros((1, 3, 2)), is_linear=True),
            ],
        )
        hidden = ops.zeros((1, 3, 2))
        attn_mask = ops.array([[1.0, 2.0, 3.0]])

        ssm_mask = forward_module.make_ssm_mask(
            cast("TransformerModel", transformer),
            hidden,
        )

        assert ssm_mask is not None
        assert ssm_mask.shape == (1, 3)
        assert (
            forward_module.select_mask(
                transformer.layers[1],
                attn_mask,
                ssm_mask,
            )
            is ssm_mask
        )
        assert (
            forward_module.select_mask(
                transformer.layers[0],
                attn_mask,
                ssm_mask,
            )
            is attn_mask
        )

    def test_run_transformer_layers_covers_guard_differentiable_and_cache_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        layer = FakeLayer(ops.ones((1, 1, 1)))
        transformer = FakeTransformer([layer, layer])
        hidden = ops.zeros((1, 1, 1))
        mask = ops.zeros((1, 1, 1))

        with pytest.raises(
            ValueError,
            match="Cannot use both differentiable=True and cache",
        ):
            forward_module.run_transformer_layers(
                cast("TransformerModel", transformer),
                hidden,
                mask,
                cache=cast("list[LayerCache]", [object()]),
                differentiable=True,
            )

        ste_calls: list[tuple[Array, Array, Array | None]] = []

        def _fake_ste(
            typed_layer: object,
            current_h: Array,
            current_mask: Array,
            ssm_mask: Array | None,
        ) -> Array:
            del typed_layer
            ste_calls.append((current_h, current_mask, ssm_mask))
            return current_h + ops.ones(current_h.shape)

        monkeypatch.setattr(forward_module, "ste_layer_forward", _fake_ste)
        differentiable_out = forward_module.run_transformer_layers(
            cast("TransformerModel", transformer),
            hidden,
            mask,
            differentiable=True,
        )
        force_eval = forward_module.force_eval
        force_eval(differentiable_out)

        assert len(ste_calls) == 2
        assert np.array_equal(np.array(differentiable_out), np.array([[[2.0]]]))

        cache_layers = [
            FakeLayer(ops.ones((1, 1, 1))),
            FakeLayer(ops.ones((1, 1, 1))),
        ]
        cache_transformer = FakeTransformer(cache_layers)
        cache = [object(), object()]
        cached_out = forward_module.run_transformer_layers(
            cast("TransformerModel", cache_transformer),
            hidden,
            mask,
            cache=cast("list[LayerCache]", cache),
        )
        force_eval(cached_out)

        assert cache_layers[0].calls[0][2] is cache[0]
        assert cache_layers[1].calls[0][2] is cache[1]

    def test_ste_layer_forward_covers_linear_and_attention_paths(self) -> None:
        linear_layer = FakeLayer(ops.full((1, 1, 2), 3.0), is_linear=True)
        normal_layer = FakeLayer(ops.full((1, 1, 2), 2.0))
        hidden = ops.array([[[1.0, 2.0]]])
        mask = ops.zeros((1, 1, 1))
        ssm_mask = ops.ones((1, 1), dtype=ops.bool_)

        linear_out = forward_module.ste_layer_forward(
            linear_layer,
            hidden,
            mask,
            ssm_mask,
        )
        normal_out = forward_module.ste_layer_forward(
            normal_layer,
            hidden,
            mask,
            None,
        )

        forward_module.force_eval(linear_out, normal_out)
        assert np.array_equal(np.array(linear_out), np.array([[[4.0, 5.0]]]))
        assert np.array_equal(np.array(normal_out), np.array([[[3.0, 4.0]]]))

    def test_make_cache_fallback_and_tied_embedding_projection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_module = FakePromptCacheModule()
        monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
        monkeypatch.setitem(sys.modules, "mlx_lm.models", ModuleType("mlx_lm.models"))
        monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)

        model_without_cache = object()
        cache = forward_module.make_cache(
            cast("CausalLM", model_without_cache),
        )
        assert cache == ["prompt-cache"]
        assert cache_module.calls == [model_without_cache]

        tied_transformer = type(
            "TiedTransformer",
            (),
            {"embed_tokens": FakeEmbedTokens(ops.ones((1, 1, 2)))},
        )()
        monkeypatch.setattr(
            forward_module,
            "get_transformer",
            lambda model: tied_transformer,
        )

        projected = forward_module.lm_head_forward(
            cast("CausalLM", object()),
            ops.zeros((1, 1, 2)),
        )
        forward_module.force_eval(projected)

        assert np.array_equal(np.array(projected), np.array([[[1.0, 1.0]]]))

    def test_reload_forward_rejects_unknown_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_array = sys.modules.get("vauban._array")
        original_forward = sys.modules.get("vauban._forward")
        monkeypatch.setattr(backend_module, "get_backend", lambda: "unknown")
        sys.modules.pop("vauban._forward", None)

        with pytest.raises(ValueError, match="Unknown backend"):
            importlib.import_module("vauban._forward")

        if original_array is not None:
            sys.modules["vauban._array"] = original_array
        if original_forward is not None:
            sys.modules["vauban._forward"] = original_forward

    def test_reload_forward_torch_branch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del monkeypatch
        original_array = sys.modules.get("vauban._array")
        original_forward = sys.modules.get("vauban._forward")
        with pytest.MonkeyPatch.context() as patch:
            functional_module = TorchForwardFunctionalModule()
            nn_module = TorchForwardNNModule("torch.nn")
            nn_module.functional = functional_module
            patch.setattr(backend_module, "get_backend", lambda: "torch")
            patch.setitem(sys.modules, "torch", TorchForwardModule())
            patch.setitem(sys.modules, "torch.nn", nn_module)
            patch.setitem(sys.modules, "torch.nn.functional", functional_module)
            sys.modules.pop("vauban._array", None)
            sys.modules.pop("vauban._forward", None)
            importlib.import_module("vauban._array")
            torch_forward = importlib.import_module("vauban._forward")

            model = TorchForwardModel()
            transformer = TorchForwardTransformer(device="cuda:7")
            token_ids = TorchForwardTensor([[1.0, 2.0]], device="cpu")
            prefix = TorchForwardTensor([[[9.0, 9.0]]], device="cpu")

            torch_forward.force_eval(cast("Array", token_ids))
            cache = torch_forward.make_cache(cast("CausalLM", model))
            with_head = cast(
                "TorchForwardTensor",
                torch_forward.lm_head_forward(
                    cast("CausalLM", model),
                    cast(
                        "Array",
                        TorchForwardTensor([[[1.0, 2.0]]], device="cuda:7"),
                    ),
                ),
            )

            tied_model = object()
            patch.setattr(
                torch_forward,
                "get_transformer",
                lambda model_obj: transformer,
            )
            tied_logits = cast(
                "TorchForwardTensor",
                torch_forward.lm_head_forward(
                    cast("CausalLM", tied_model),
                    cast(
                        "Array",
                        TorchForwardTensor([[[1.0, 2.0]]], device="cuda:7"),
                    ),
                ),
            )
            tuple_logits = cast(
                "TorchForwardTensor",
                torch_forward.extract_logits(
                    cast(
                        "tuple[Array, ...]",
                        (
                            TorchForwardTensor([[1.0]]),
                            TorchForwardTensor([[0.0]]),
                        ),
                    ),
                ),
            )
            object_logits = cast(
                "TorchForwardTensor",
                torch_forward.extract_logits(
                    cast(
                        "Array",
                        type(
                            "LogitWrapper",
                            (),
                            {"logits": TorchForwardTensor([[2.0]])},
                        )(),
                    ),
                ),
            )
            bare_logits = cast(
                "TorchForwardTensor",
                torch_forward.extract_logits(
                    cast("Array", TorchForwardTensor([[3.0]])),
                ),
            )
            embedded, embedded_mask = torch_forward.embed_and_mask(
                cast("TransformerModel", transformer),
                cast("Array", token_ids),
            )
            prefixed, prefixed_mask = torch_forward.embed_and_mask_with_prefix(
                cast("TransformerModel", transformer),
                cast("Array", prefix),
                cast(
                    "Array",
                    TorchForwardTensor([[1.0, 2.0]], device="cpu"),
                ),
            )
            suffixed, _ = torch_forward.embed_and_mask_with_prefix(
                cast("TransformerModel", transformer),
                cast("Array", prefix),
                cast(
                    "Array",
                    TorchForwardTensor([[1.0, 2.0]], device="cpu"),
                ),
                token_position="suffix",
            )
            infixed, _ = torch_forward.embed_and_mask_with_prefix(
                cast("TransformerModel", transformer),
                cast("Array", prefix),
                cast(
                    "Array",
                    TorchForwardTensor([[1.0, 2.0, 3.0]], device="cpu"),
                ),
                token_position="infix",
                infix_split=1,
            )
            matrix = cast(
                "Array",
                TorchForwardTensor([[3.0, 0.0], [0.0, 4.0]], device="cuda:3"),
            )
            u, s, vt = torch_forward.svd_stable(matrix)
            q, r = torch_forward.qr_stable(matrix)

            embedded_tensor = cast("TorchForwardTensor", embedded)
            prefixed_tensor = cast("TorchForwardTensor", prefixed)
            suffixed_tensor = cast("TorchForwardTensor", suffixed)
            infixed_tensor = cast("TorchForwardTensor", infixed)
            u_tensor = cast("TorchForwardTensor", u)
            s_tensor = cast("TorchForwardTensor", s)
            vt_tensor = cast("TorchForwardTensor", vt)
            q_tensor = cast("TorchForwardTensor", q)
            r_tensor = cast("TorchForwardTensor", r)

            assert cache == ["torch-cache"]
            assert np.array_equal(with_head.data, np.array([[[6.0, 7.0]]]))
            assert np.array_equal(tied_logits.data, np.array([[[3.0, 3.0, 3.0]]]))
            assert np.array_equal(tuple_logits.data, np.array([[1.0]]))
            assert np.array_equal(object_logits.data, np.array([[2.0]]))
            assert np.array_equal(bare_logits.data, np.array([[3.0]]))
            assert embedded_mask is None
            assert embedded_tensor.device == "cuda:7"
            assert prefixed_mask is None
            assert prefixed_tensor.shape == (1, 3, 2)
            assert suffixed_tensor.shape == (1, 3, 2)
            assert infixed_tensor.shape == (1, 4, 2)
            assert functional_module.linear_calls
            assert u_tensor.device == "cuda:3"
            assert s_tensor.device == "cuda:3"
            assert vt_tensor.device == "cuda:3"
            assert q_tensor.device == "cuda:3"
            assert r_tensor.device == "cuda:3"

        if original_array is not None:
            sys.modules["vauban._array"] = original_array
        if original_forward is not None:
            sys.modules["vauban._forward"] = original_forward
