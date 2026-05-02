# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI-compatible behavior trace API calls."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from vauban.api_trace import call_api_behavior_trace
from vauban.types import BehaviorTraceApiConfig

if TYPE_CHECKING:
    from types import TracebackType

    import pytest

    from vauban.behavior import JsonValue


@dataclass(slots=True)
class _CapturedRequest:
    """Request fields observed by a fake API endpoint."""

    url: str = ""
    auth: str | None = None
    body: dict[str, JsonValue] = field(default_factory=dict)


class _FakeResponse:
    """Small context-manager response for urllib request tests."""

    def __init__(self, payload: dict[str, JsonValue]) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def read(self) -> bytes:
        """Return encoded response bytes."""
        return self._payload


def test_call_api_behavior_trace_builds_openai_compatible_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API traces should request chat completions and preserve logprobs."""
    endpoint = BehaviorTraceApiConfig(
        name="unit-api",
        base_url="https://example.test/v1",
        model="provider/model-candidate",
        api_key_env="VAUBAN_TEST_API_KEY",
        system_prompt="Answer concisely.",
        timeout=9,
    )
    captured = _CapturedRequest()

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: int,
    ) -> _FakeResponse:
        captured.url = request.full_url
        captured.auth = request.get_header("Authorization")
        assert timeout == 9
        data = request.data
        assert data is not None
        body = json.loads(data.decode())
        assert isinstance(body, dict)
        captured.body = cast("dict[str, JsonValue]", body)
        return _FakeResponse({
            "choices": [
                {
                    "message": {
                        "content": "Rainbows form when sunlight refracts."
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token": "Rainbows", "logprob": -0.25}
                        ],
                    },
                }
            ],
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    response = call_api_behavior_trace(
        endpoint=endpoint,
        api_key="secret",
        prompt="Explain rainbows.",
        max_tokens=16,
        refusal_phrases=["I cannot"],
        return_logprobs=True,
    )

    assert captured.url == "https://example.test/v1/chat/completions"
    assert captured.auth == "Bearer secret"
    assert captured.body["model"] == "provider/model-candidate"
    assert captured.body["max_tokens"] == 16
    assert captured.body["logprobs"] is True
    messages = cast("list[dict[str, str]]", captured.body["messages"])
    assert messages == [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Explain rainbows."},
    ]
    assert response.text == "Rainbows form when sunlight refracts."
    assert response.refused is False
    assert response.finish_reason == "stop"
    assert response.logprobs == [{"token": "Rainbows", "logprob": -0.25}]
