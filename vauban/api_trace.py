# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible API calls for endpoint behavior traces."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from vauban._network import validate_http_url
from vauban.behavior import JsonValue, is_refusal_text

if TYPE_CHECKING:
    from vauban.types import BehaviorTraceApiConfig


@dataclass(frozen=True, slots=True)
class ApiTraceResponse:
    """One response collected from an API-backed behavior trace."""

    text: str
    refused: bool
    finish_reason: str | None = None
    logprobs: list[JsonValue] = field(default_factory=list)
    error: str | None = None

    def metadata(self) -> dict[str, JsonValue]:
        """Return JSON-compatible response metadata for trace observations."""
        metadata: dict[str, JsonValue] = {
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        if self.logprobs:
            metadata["logprobs"] = self.logprobs
        return {key: value for key, value in metadata.items() if value is not None}


def call_api_behavior_trace(
    *,
    endpoint: BehaviorTraceApiConfig,
    api_key: str,
    prompt: str,
    max_tokens: int,
    refusal_phrases: list[str],
    return_logprobs: bool,
) -> ApiTraceResponse:
    """Call an OpenAI-compatible chat endpoint for behavior tracing."""
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    validate_http_url(url, context=f"behavior trace API endpoint {endpoint.name}")
    body = _request_body(
        endpoint,
        prompt,
        max_tokens=max_tokens,
        return_logprobs=return_logprobs,
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=_auth_headers(endpoint, api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=endpoint.timeout) as response:
            payload = json.loads(response.read().decode())
        if not isinstance(payload, dict):
            return _error_response("API response was not a JSON object")
        return _response_from_payload(
            cast("dict[str, object]", payload),
            refusal_phrases,
        )
    except urllib.error.HTTPError as exc:
        return _error_response(f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _error_response(str(exc))
    except json.JSONDecodeError as exc:
        return _error_response(f"malformed JSON response: {exc}")


def _request_body(
    endpoint: BehaviorTraceApiConfig,
    prompt: str,
    *,
    max_tokens: int,
    return_logprobs: bool,
) -> dict[str, JsonValue]:
    """Build an OpenAI-compatible chat completion request body."""
    messages: list[JsonValue] = []
    if endpoint.system_prompt is not None:
        messages.append({"role": "system", "content": endpoint.system_prompt})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, JsonValue] = {
        "model": endpoint.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if return_logprobs:
        body["logprobs"] = True
    return body


def _auth_headers(
    endpoint: BehaviorTraceApiConfig,
    api_key: str,
) -> dict[str, str]:
    """Build authentication headers for an endpoint trace request."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "vauban/1.0",
    }
    if endpoint.auth_header is not None:
        headers[endpoint.auth_header] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _response_from_payload(
    payload: dict[str, object],
    refusal_phrases: list[str],
) -> ApiTraceResponse:
    """Convert an OpenAI-compatible response payload into trace metadata."""
    choice = _first_choice(payload)
    if choice is None:
        return _error_response("API response had no choices")
    text = _message_text(choice)
    if not text:
        return _error_response("API response had empty assistant content")
    finish_reason = _optional_string(choice.get("finish_reason"))
    violation_refusal = finish_reason == "violation"
    return ApiTraceResponse(
        text=text,
        refused=violation_refusal or is_refusal_text(text, refusal_phrases),
        finish_reason=finish_reason,
        logprobs=_logprobs(choice.get("logprobs")),
    )


def _first_choice(payload: dict[str, object]) -> dict[str, object] | None:
    """Return the first choice object from a chat completion payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return cast("dict[str, object]", first)


def _message_text(choice: dict[str, object]) -> str:
    """Return assistant content or reasoning text from one choice."""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    message_dict = cast("dict[str, object]", message)
    content = message_dict.get("content")
    if isinstance(content, str) and content:
        return content
    reasoning = message_dict.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return ""


def _logprobs(raw: object) -> list[JsonValue]:
    """Return sanitized token logprob entries from an API response."""
    if not isinstance(raw, dict):
        return []
    raw_dict = cast("dict[str, object]", raw)
    content = raw_dict.get("content")
    if not isinstance(content, list):
        return []
    return [_json_value(item) for item in content]


def _json_value(raw: object) -> JsonValue:
    """Return a JSON-compatible value, replacing unknown objects with strings."""
    if raw is None or isinstance(raw, str | int | float | bool):
        return raw
    if isinstance(raw, list):
        return [_json_value(item) for item in raw]
    if isinstance(raw, dict):
        result: dict[str, JsonValue] = {}
        for key, value in raw.items():
            if isinstance(key, str):
                result[key] = _json_value(value)
        return result
    return str(raw)


def _optional_string(raw: object) -> str | None:
    """Return a string value when present."""
    return raw if isinstance(raw, str) else None


def _error_response(message: str) -> ApiTraceResponse:
    """Return a conservative refused response for endpoint failures."""
    return ApiTraceResponse(
        text="I cannot assist with that request.",
        refused=True,
        error=message,
    )
