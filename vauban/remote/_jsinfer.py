# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""jsinfer backend -- Jane Street batch inference API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.types import RemoteBackend

from vauban.types import RemoteActivationResult, RemoteChatResult


class JsinferBackend:
    """RemoteBackend implementation using the jsinfer library.

    The ``jsinfer`` library is imported lazily inside methods so that
    vauban can be installed without it.

    Args:
        api_key: API key for the jsinfer service.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def chat(
        self,
        model_id: str,
        prompts: list[str],
        max_tokens: int,
    ) -> list[RemoteChatResult]:
        """Send chat completions via jsinfer batch API."""
        from jsinfer import BatchInferenceClient, ChatCompletionRequest, Message

        client = BatchInferenceClient(api_key=self._api_key)
        requests = [
            ChatCompletionRequest(
                custom_id=f"p{i:03d}",
                messages=[Message(role="user", content=prompt)],
            )
            for i, prompt in enumerate(prompts)
        ]

        results = await client.chat_completions(requests, model=model_id)

        out: list[RemoteChatResult] = []
        for i, prompt in enumerate(prompts):
            result = results.get(f"p{i:03d}")
            if result is not None:
                content = result.messages[-1].content
                out.append(RemoteChatResult(prompt=prompt, response=content))
            else:
                out.append(RemoteChatResult(prompt=prompt, error="no response"))
        return out

    async def activations(
        self,
        model_id: str,
        prompts: list[str],
        modules: list[str],
    ) -> list[RemoteActivationResult]:
        """Fetch activation tensors via jsinfer batch API."""
        from jsinfer import ActivationsRequest, BatchInferenceClient, Message

        client = BatchInferenceClient(api_key=self._api_key)
        requests = [
            ActivationsRequest(
                custom_id=f"act_{i:03d}",
                messages=[Message(role="user", content=prompt)],
                module_names=modules,
            )
            for i, prompt in enumerate(prompts)
        ]

        results = await client.activations(requests, model=model_id)

        out: list[RemoteActivationResult] = []
        for i, _prompt in enumerate(prompts):
            result = results.get(f"act_{i:03d}")
            if result is None:
                continue
            for module_name, arr in result.activations.items():
                # Convert to nested list for serialization
                data = arr.tolist() if hasattr(arr, "tolist") else list(arr)
                out.append(RemoteActivationResult(
                    prompt_index=i,
                    module_name=module_name,
                    data=data,
                ))
        return out


def create_jsinfer_backend(api_key: str) -> RemoteBackend:
    """Factory function for the jsinfer backend.

    Args:
        api_key: API key for the jsinfer service.

    Returns:
        A ``JsinferBackend`` instance.
    """
    return JsinferBackend(api_key)
