# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra parser tests for vauban.config._parse_api_eval."""

from __future__ import annotations

from typing import cast

import pytest

from vauban.config._parse_api_eval import _parse_api_eval


def _minimal_endpoint(**overrides: object) -> dict[str, object]:
    endpoint: dict[str, object] = {
        "name": "test-ep",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key_env": "TEST_KEY",
    }
    endpoint.update(overrides)
    return endpoint


def _minimal_raw(**overrides: object) -> dict[str, object]:
    api_eval: dict[str, object] = {"endpoints": [_minimal_endpoint()]}
    api_eval.update(overrides)
    return {"api_eval": api_eval}


class TestParseApiEvalExtra:
    """Cover parser branches not exercised by the main API eval tests."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"token_position": 123}, "token_position"),
            ({"defense_proxy_sic_mode": 123}, "defense_proxy_sic_mode"),
            (
                {"defense_proxy_sic_threshold": "bad"},
                "defense_proxy_sic_threshold",
            ),
            (
                {"defense_proxy_sic_max_iterations": "bad"},
                "defense_proxy_sic_max_iterations",
            ),
            ({"defense_proxy_cast_mode": 123}, "defense_proxy_cast_mode"),
            (
                {"defense_proxy_cast_threshold": "bad"},
                "defense_proxy_cast_threshold",
            ),
            ({"defense_proxy_cast_layers": "bad"}, "defense_proxy_cast_layers"),
            ({"defense_proxy_cast_alpha": "bad"}, "defense_proxy_cast_alpha"),
            (
                {"defense_proxy_cast_max_tokens": "bad"},
                "defense_proxy_cast_max_tokens",
            ),
        ],
    )
    def test_top_level_type_errors(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(TypeError, match=match):
            _parse_api_eval(_minimal_raw(**overrides))

    @pytest.mark.parametrize(
        ("endpoint_overrides", "match"),
        [
            ({"name": ""}, "name"),
            ({"base_url": ""}, "base_url"),
            ({"model": ""}, "model"),
            ({"api_key_env": ""}, "api_key_env"),
        ],
    )
    def test_endpoint_empty_string_rejected(
        self,
        endpoint_overrides: dict[str, object],
        match: str,
    ) -> None:
        raw = _minimal_raw()
        api_eval = cast("dict[str, object]", raw["api_eval"])
        endpoints_obj = cast("list[dict[str, object]]", api_eval["endpoints"])
        endpoints_obj[0] = _minimal_endpoint(**endpoint_overrides)
        with pytest.raises(ValueError, match=match):
            _parse_api_eval(raw)

    def test_endpoint_system_prompt_type_error(self) -> None:
        raw = _minimal_raw()
        api_eval = cast("dict[str, object]", raw["api_eval"])
        endpoints_obj = cast("list[dict[str, object]]", api_eval["endpoints"])
        endpoints_obj[0] = _minimal_endpoint(system_prompt=42)
        with pytest.raises(TypeError, match="system_prompt"):
            _parse_api_eval(raw)

    def test_endpoint_auth_header_empty_string(self) -> None:
        raw = _minimal_raw()
        api_eval = cast("dict[str, object]", raw["api_eval"])
        endpoints_obj = cast("list[dict[str, object]]", api_eval["endpoints"])
        endpoints_obj[0] = _minimal_endpoint(auth_header="")
        with pytest.raises(ValueError, match="auth_header"):
            _parse_api_eval(raw)

    def test_endpoint_type_error_branches(self) -> None:
        raw = _minimal_raw(defense_proxy_cast_layers=[1, "two", 3])
        with pytest.raises(TypeError, match=r"defense_proxy_cast_layers\[1\]"):
            _parse_api_eval(raw)
