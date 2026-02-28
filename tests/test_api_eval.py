"""Tests for vauban.api_eval: API-based suffix evaluation."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from vauban.api_eval import (
    _call_chat_api,
    _is_refusal,
    evaluate_suffix_via_api,
)
from vauban.types import ApiEvalConfig, ApiEvalEndpoint


@pytest.fixture
def endpoint() -> ApiEvalEndpoint:
    """A sample API endpoint."""
    return ApiEvalEndpoint(
        name="test-ep",
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
    )


@pytest.fixture
def config(endpoint: ApiEvalEndpoint) -> ApiEvalConfig:
    """A sample API eval config."""
    return ApiEvalConfig(
        endpoints=[endpoint],
        max_tokens=50,
        timeout=10,
        system_prompt="You are helpful.",
    )


class TestIsRefusal:
    def test_refusal_detected(self) -> None:
        assert _is_refusal("I cannot help with that request.")

    def test_refusal_case_insensitive(self) -> None:
        assert _is_refusal("I CANNOT do that.")

    def test_non_refusal(self) -> None:
        assert not _is_refusal("Sure, here is the information you requested.")

    def test_empty_string(self) -> None:
        assert not _is_refusal("")


class TestCallChatApi:
    def test_successful_call(self, endpoint: ApiEvalEndpoint) -> None:
        response_body = json.dumps({
            "choices": [{"message": {"content": "Sure, here you go."}}],
        }).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _call_chat_api(
                endpoint=endpoint,
                api_key="sk-test",
                prompt="Hello",
                system_prompt="Be helpful.",
                max_tokens=50,
                timeout=10,
            )

        assert result == "Sure, here you go."

    def test_empty_choices_returns_refusal(
        self, endpoint: ApiEvalEndpoint,
    ) -> None:
        response_body = json.dumps({"choices": []}).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _call_chat_api(
                endpoint=endpoint,
                api_key="sk-test",
                prompt="Hello",
                system_prompt=None,
                max_tokens=50,
                timeout=10,
            )

        assert _is_refusal(result)

    def test_timeout_returns_refusal(
        self, endpoint: ApiEvalEndpoint,
    ) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = _call_chat_api(
                endpoint=endpoint,
                api_key="sk-test",
                prompt="Hello",
                system_prompt=None,
                max_tokens=50,
                timeout=10,
            )

        assert _is_refusal(result)

    def test_http_error_returns_refusal(
        self, endpoint: ApiEvalEndpoint,
    ) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://api.example.com/v1/chat/completions",
                429,
                "Too Many Requests",
                {},  # type: ignore[arg-type]
                None,
            ),
        ):
            result = _call_chat_api(
                endpoint=endpoint,
                api_key="sk-test",
                prompt="Hello",
                system_prompt=None,
                max_tokens=50,
                timeout=10,
            )

        assert _is_refusal(result)


class TestEvaluateSuffixViaApi:
    def test_missing_api_key_skips(
        self,
        config: ApiEvalConfig,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            results = evaluate_suffix_via_api(
                "suffix", ["prompt1"], config,
            )

        assert len(results) == 1
        assert results[0].model_id == "test-ep"
        assert results[0].success_rate == 0.0
        assert results[0].eval_responses == []

    def test_successful_bypass(
        self,
        config: ApiEvalConfig,
    ) -> None:
        response_body = json.dumps({
            "choices": [{"message": {"content": "Sure, here you go!"}}],
        }).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            results = evaluate_suffix_via_api(
                "adversarial suffix",
                ["How to do something bad?"],
                config,
            )

        assert len(results) == 1
        assert results[0].model_id == "test-ep"
        assert results[0].success_rate == 1.0

    def test_refusal_response(
        self,
        config: ApiEvalConfig,
    ) -> None:
        response_body = json.dumps({
            "choices": [{"message": {"content": "I cannot help with that."}}],
        }).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            results = evaluate_suffix_via_api(
                "adversarial suffix",
                ["How to do something bad?"],
                config,
            )

        assert len(results) == 1
        assert results[0].success_rate == 0.0

    def test_endpoint_system_prompt_override(self) -> None:
        ep = ApiEvalEndpoint(
            name="ep",
            base_url="https://api.example.com/v1",
            model="m",
            api_key_env="K",
            system_prompt="I am an endpoint override.",
        )
        cfg = ApiEvalConfig(
            endpoints=[ep],
            system_prompt="I am the default.",
        )
        # The endpoint system_prompt should win
        assert ep.system_prompt == "I am an endpoint override."
        assert cfg.system_prompt == "I am the default."
