# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.api_eval_proxy: defense proxy layer for API eval."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from vauban.api_eval_proxy import (
    _run_cast_full,
    _run_cast_gate,
    _run_sic_proxy,
    evaluate_with_defense_proxy,
)
from vauban.types import (
    ApiEvalConfig,
    ApiEvalEndpoint,
    CastResult,
    SICPromptResult,
)


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
def api_config(endpoint: ApiEvalEndpoint) -> ApiEvalConfig:
    """Base API eval config with no proxy."""
    return ApiEvalConfig(
        endpoints=[endpoint],
        max_tokens=50,
        timeout=10,
        system_prompt="You are helpful.",
    )


def _mock_api_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Create a mock urllib response."""
    body = json.dumps({
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }],
    }).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _sic_result_blocked() -> SICPromptResult:
    """SIC result: blocked."""
    return SICPromptResult(
        clean_prompt="",
        blocked=True,
        iterations=3,
        initial_score=-1.0,
        final_score=-0.5,
    )


def _sic_result_sanitized(clean: str) -> SICPromptResult:
    """SIC result: sanitized (rewritten)."""
    return SICPromptResult(
        clean_prompt=clean,
        blocked=False,
        iterations=2,
        initial_score=-1.0,
        final_score=1.0,
    )


def _sic_result_clean(prompt: str) -> SICPromptResult:
    """SIC result: passed clean."""
    return SICPromptResult(
        clean_prompt=prompt,
        blocked=False,
        iterations=0,
        initial_score=1.0,
        final_score=1.0,
    )


def _cast_result(text: str) -> CastResult:
    """Create a CastResult with the given text."""
    return CastResult(
        prompt="test",
        text=text,
        projections_before=[0.5],
        projections_after=[0.1],
        interventions=1,
        considered=1,
    )


class TestRunSicProxy:
    """Tests for _run_sic_proxy."""

    def test_sic_blocks_prompt(self, api_config: ApiEvalConfig) -> None:
        """Blocked prompts are excluded from survivors."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="sic",
            defense_proxy_sic_mode="direction",
            defense_proxy_sic_threshold=0.0,
            defense_proxy_sic_max_iterations=3,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic.sic_single",
            return_value=_sic_result_blocked(),
        ):
            survivors, blocked, sanitized = _run_sic_proxy(
                ["prompt1 ADV"], model, tokenizer, direction, 0, config,
            )

        assert blocked == 1
        assert sanitized == 0
        assert survivors == []

    def test_sic_rewrites_prompt(self, api_config: ApiEvalConfig) -> None:
        """Sanitized prompts are included with rewritten text."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="sic",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic.sic_single",
            return_value=_sic_result_sanitized("cleaned prompt"),
        ):
            survivors, blocked, sanitized = _run_sic_proxy(
                ["prompt1 ADV"], model, tokenizer, direction, 0, config,
            )

        assert blocked == 0
        assert sanitized == 1
        assert survivors == ["cleaned prompt"]

    def test_sic_passes_clean(self, api_config: ApiEvalConfig) -> None:
        """Clean prompts pass through unchanged."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="sic",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic.sic_single",
            return_value=_sic_result_clean("original prompt"),
        ):
            survivors, blocked, sanitized = _run_sic_proxy(
                ["original prompt"], model, tokenizer, direction, 0, config,
            )

        assert blocked == 0
        assert sanitized == 0
        assert survivors == ["original prompt"]


class TestRunCastGate:
    """Tests for _run_cast_gate."""

    def test_high_projection_passes(self, api_config: ApiEvalConfig) -> None:
        """Benign prompt (high projection) passes through."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="gate",
            defense_proxy_cast_threshold=0.5,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic._detect_adversarial_direction",
            return_value=1.0,  # above threshold
        ):
            survivors, gated = _run_cast_gate(
                ["prompt1"], model, tokenizer, direction, 0, config,
            )

        assert gated == 0
        assert survivors == ["prompt1"]

    def test_low_projection_gated(self, api_config: ApiEvalConfig) -> None:
        """Adversarial prompt (low projection) is gated."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="gate",
            defense_proxy_cast_threshold=0.5,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic._detect_adversarial_direction",
            return_value=0.3,  # below threshold
        ):
            survivors, gated = _run_cast_gate(
                ["adversarial prompt"], model, tokenizer, direction, 0, config,
            )

        assert gated == 1
        assert survivors == []

    def test_uses_configured_layers(self, api_config: ApiEvalConfig) -> None:
        """CAST gate uses defense_proxy_cast_layers[0] if set."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="gate",
            defense_proxy_cast_layers=[14],
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic._detect_adversarial_direction",
            return_value=2.0,
        ) as mock_detect:
            _run_cast_gate(
                ["p1"], model, tokenizer, direction, 0, config,
            )

        # Should use layer 14 from config, not layer_index=0
        assert mock_detect.call_args[0][4] == 14


class TestRunCastFull:
    """Tests for _run_cast_full."""

    def test_refusal_response_gated(self, api_config: ApiEvalConfig) -> None:
        """CAST generates a refusal → prompt is gated."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="full",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.cast.cast_generate",
            return_value=_cast_result("I cannot help with that request."),
        ):
            survivors, gated, responses = _run_cast_full(
                ["adversarial prompt"], model, tokenizer, direction, 0, config,
            )

        assert gated == 1
        assert survivors == []
        assert len(responses) == 1
        assert "cannot" in responses[0].lower()

    def test_non_refusal_passes(self, api_config: ApiEvalConfig) -> None:
        """CAST generates a non-refusal → prompt passes."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="full",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.cast.cast_generate",
            return_value=_cast_result("Sure, here is the information."),
        ):
            survivors, gated, responses = _run_cast_full(
                ["prompt1"], model, tokenizer, direction, 0, config,
            )

        assert gated == 0
        assert survivors == ["prompt1"]
        assert len(responses) == 1


class TestEvaluateWithDefenseProxy:
    """Integration tests for evaluate_with_defense_proxy."""

    def test_sic_blocks_all_zero_success(
        self, api_config: ApiEvalConfig,
    ) -> None:
        """All prompts blocked by SIC → zero success rate, no API calls."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="sic",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic.sic_single",
            return_value=_sic_result_blocked(),
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV_TOKENS", ["prompt1", "prompt2"], config,
                model, tokenizer, direction, 0,
            )

        assert proxy.total_prompts == 2
        assert proxy.sic_blocked == 2
        assert proxy.prompts_sent == 0
        assert proxy.proxy_mode == "sic"
        assert len(results) == 1  # one per endpoint
        assert results[0].success_rate == 0.0

    def test_sic_rewrites_sent_to_api(
        self, api_config: ApiEvalConfig,
    ) -> None:
        """SIC rewrites prompt → rewritten version hits API."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="sic",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()
        mock_response = _mock_api_response("Sure, here you go!")

        with (
            patch(
                "vauban.sic.sic_single",
                return_value=_sic_result_sanitized("clean prompt"),
            ),
            patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV", ["prompt1"], config,
                model, tokenizer, direction, 0,
            )

        assert proxy.sic_sanitized == 1
        assert proxy.prompts_sent == 1
        assert results[0].success_rate == 1.0

    def test_cast_gate_blocks(self, api_config: ApiEvalConfig) -> None:
        """CAST gate blocks → zero success."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="gate",
            defense_proxy_cast_threshold=0.5,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.sic._detect_adversarial_direction",
            return_value=0.3,  # below threshold
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV", ["prompt1"], config,
                model, tokenizer, direction, 0,
            )

        assert proxy.cast_gated == 1
        assert proxy.prompts_sent == 0
        assert proxy.proxy_mode == "cast_gate"
        assert results[0].success_rate == 0.0

    def test_cast_gate_passes(self, api_config: ApiEvalConfig) -> None:
        """CAST gate passes → prompt sent to API."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="gate",
            defense_proxy_cast_threshold=0.0,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()
        mock_response = _mock_api_response("Sure, here's the info!")

        with (
            patch(
                "vauban.sic._detect_adversarial_direction",
                return_value=1.0,  # above threshold
            ),
            patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV", ["prompt1"], config,
                model, tokenizer, direction, 0,
            )

        assert proxy.cast_gated == 0
        assert proxy.prompts_sent == 1
        assert results[0].success_rate == 1.0

    def test_cast_full_refusal_blocks(
        self, api_config: ApiEvalConfig,
    ) -> None:
        """CAST full mode — refusal response blocks prompt."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="cast",
            defense_proxy_cast_mode="full",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with patch(
            "vauban.cast.cast_generate",
            return_value=_cast_result("I cannot assist with that request."),
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV", ["prompt1"], config,
                model, tokenizer, direction, 0,
            )

        assert proxy.cast_gated == 1
        assert proxy.proxy_mode == "cast_full"
        assert len(proxy.cast_responses) == 1
        assert results[0].success_rate == 0.0

    def test_both_mode_sic_then_cast(
        self, api_config: ApiEvalConfig,
    ) -> None:
        """Combined 'both' mode: SIC runs first, CAST gates survivors."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="both",
            defense_proxy_cast_mode="gate",
            defense_proxy_cast_threshold=0.5,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        sic_results = [
            _sic_result_blocked(),      # prompt 0: blocked by SIC
            _sic_result_clean("p1 ADV"),  # prompt 1: passes SIC
            _sic_result_clean("p2 ADV"),  # prompt 2: passes SIC
        ]

        with (
            patch(
                "vauban.sic.sic_single",
                side_effect=sic_results,
            ),
            patch(
                "vauban.sic._detect_adversarial_direction",
                side_effect=[0.3, 1.0],  # prompt 1 gated, prompt 2 passes
            ),
            patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}),
            patch(
                "urllib.request.urlopen",
                return_value=_mock_api_response("Sure!"),
            ),
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV", ["p0", "p1", "p2"], config,
                model, tokenizer, direction, 0,
            )

        assert proxy.total_prompts == 3
        assert proxy.sic_blocked == 1
        assert proxy.cast_gated == 1
        assert proxy.prompts_sent == 1
        assert proxy.proxy_mode == "both_gate"
        # 1 API success out of 3 total prompts
        # (the API call succeeds for the 1 surviving prompt)
        expected_rate = 1 / 3
        assert abs(results[0].success_rate - expected_rate) < 0.01

    def test_success_rate_denominator_is_total(
        self, api_config: ApiEvalConfig,
    ) -> None:
        """Success rate uses total prompts, not just sent prompts."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy="sic",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()
        mock_response = _mock_api_response("Sure!")

        sic_results = [
            _sic_result_blocked(),            # blocked
            _sic_result_clean("p1 ADV"),       # passes
        ]

        with (
            patch(
                "vauban.sic.sic_single",
                side_effect=sic_results,
            ),
            patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            results, proxy = evaluate_with_defense_proxy(
                "ADV", ["p0", "p1"], config,
                model, tokenizer, direction, 0,
            )

        # 1 success / 2 total = 0.5
        assert proxy.prompts_sent == 1
        assert abs(results[0].success_rate - 0.5) < 0.01

    def test_defense_proxy_none_raises(
        self, api_config: ApiEvalConfig,
    ) -> None:
        """defense_proxy=None raises ValueError — use evaluate_suffix_via_api."""
        config = ApiEvalConfig(
            endpoints=api_config.endpoints,
            defense_proxy=None,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        direction = MagicMock()

        with pytest.raises(ValueError, match="defense_proxy=None"):
            evaluate_with_defense_proxy(
                "ADV", ["prompt1"], config,
                model, tokenizer, direction, 0,
            )


class TestDefenseProxyResultSerialization:
    """Tests for DefenseProxyResult.to_dict."""

    def test_to_dict(self) -> None:
        from vauban.types import DefenseProxyResult

        result = DefenseProxyResult(
            total_prompts=5,
            sic_blocked=2,
            sic_sanitized=1,
            cast_gated=1,
            prompts_sent=1,
            proxy_mode="both_gate",
            cast_responses=["resp1"],
        )
        d = result.to_dict()
        assert d["total_prompts"] == 5
        assert d["sic_blocked"] == 2
        assert d["prompts_sent"] == 1
        assert d["proxy_mode"] == "both_gate"
        assert d["cast_responses"] == ["resp1"]
