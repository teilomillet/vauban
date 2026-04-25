# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Protocol contracts implemented by Vauban runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from vauban.runtime._types import (
        BackendCapabilities,
        ForwardRequest,
        ForwardTrace,
        LoadedModel,
        ModelRef,
        TokenizedPrompt,
        TokenizeRequest,
    )

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class RuntimeStage(Protocol[InputT, OutputT]):
    """Composable runtime stage with a narrow input and output."""

    name: str

    def run(self, request: InputT) -> OutputT:
        """Run the stage."""
        ...


class ModelRuntime(Protocol):
    """Backend implementation of Vauban runtime primitives."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Declared runtime support."""
        ...

    def load_model(self, ref: ModelRef) -> LoadedModel:
        """Load a model reference into a backend-specific handle."""
        ...

    def tokenize(
        self,
        loaded: LoadedModel,
        request: TokenizeRequest,
    ) -> TokenizedPrompt:
        """Tokenize text with the loaded model tokenizer."""
        ...

    def forward(
        self,
        loaded: LoadedModel,
        request: ForwardRequest,
    ) -> ForwardTrace:
        """Run a forward pass and return observed evidence."""
        ...
