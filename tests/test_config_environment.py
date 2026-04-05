# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for [environment] config sub-parsers."""

import pytest

from vauban.config._parse_environment import (
    _parse_environment,
    _parse_policy,
    _parse_target,
    _parse_task,
    _parse_tools,
)


def _minimal_env() -> dict[str, object]:
    """Build a minimal valid [environment] section."""
    return {
        "environment": {
            "system_prompt": "You are an assistant.",
            "injection_surface": "calendar",
            "tools": [
                {
                    "name": "calendar",
                    "description": "Lookup events",
                    "parameters": {},
                },
                {
                    "name": "send_email",
                    "description": "Send email",
                    "parameters": {"to": "string", "body": "string"},
                },
            ],
            "target": {
                "function": "send_email",
                "required_args": ["to"],
                "arg_contains": {"to": "attacker@"},
            },
            "task": {"content": "Check my calendar."},
        },
    }


# ── _parse_environment (top-level) ───────────────────────────────────


class TestParseEnvironment:
    def test_absent_returns_none(self) -> None:
        assert _parse_environment({}) is None

    def test_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_environment({"environment": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_environment(_minimal_env())
        assert cfg is not None
        assert cfg.system_prompt == "You are an assistant."
        assert cfg.injection_surface == "calendar"
        assert cfg.injection_position == "suffix"
        assert cfg.benign_expected_tools == []
        assert cfg.max_turns == 6
        assert cfg.max_gen_tokens == 200
        assert cfg.rollout_top_n == 8
        assert cfg.rollout_every_n == 1
        assert cfg.temperature == 0.0
        assert len(cfg.tools) == 2
        assert cfg.target.function == "send_email"
        assert cfg.task.content == "Check my calendar."
        assert cfg.policy is None

    def test_missing_system_prompt(self) -> None:
        raw = _minimal_env()
        del raw["environment"]["system_prompt"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="system_prompt"):
            _parse_environment(raw)

    def test_missing_injection_surface(self) -> None:
        raw = _minimal_env()
        del raw["environment"]["injection_surface"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="injection_surface"):
            _parse_environment(raw)

    def test_max_turns_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["max_turns"] = "bad"  # type: ignore[index]
        with pytest.raises(TypeError, match="max_turns"):
            _parse_environment(raw)

    def test_max_turns_range(self) -> None:
        raw = _minimal_env()
        raw["environment"]["max_turns"] = 0  # type: ignore[index]
        with pytest.raises(ValueError, match="max_turns"):
            _parse_environment(raw)

    def test_injection_position_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["injection_position"] = 1  # type: ignore[index]
        with pytest.raises(TypeError, match="injection_position"):
            _parse_environment(raw)

    def test_injection_position_invalid(self) -> None:
        raw = _minimal_env()
        raw["environment"]["injection_position"] = "middle"  # type: ignore[index]
        with pytest.raises(ValueError, match="injection_position"):
            _parse_environment(raw)

    def test_benign_expected_tools_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["benign_expected_tools"] = "calendar"  # type: ignore[index]
        with pytest.raises(TypeError, match="benign_expected_tools"):
            _parse_environment(raw)

    def test_max_gen_tokens_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["max_gen_tokens"] = "bad"  # type: ignore[index]
        with pytest.raises(TypeError, match="max_gen_tokens"):
            _parse_environment(raw)

    def test_max_gen_tokens_range(self) -> None:
        raw = _minimal_env()
        raw["environment"]["max_gen_tokens"] = 0  # type: ignore[index]
        with pytest.raises(ValueError, match="max_gen_tokens"):
            _parse_environment(raw)

    def test_rollout_top_n_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["rollout_top_n"] = "bad"  # type: ignore[index]
        with pytest.raises(TypeError, match="rollout_top_n"):
            _parse_environment(raw)

    def test_rollout_top_n_range(self) -> None:
        raw = _minimal_env()
        raw["environment"]["rollout_top_n"] = 0  # type: ignore[index]
        with pytest.raises(ValueError, match="rollout_top_n"):
            _parse_environment(raw)

    def test_rollout_every_n_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["rollout_every_n"] = "bad"  # type: ignore[index]
        with pytest.raises(TypeError, match="rollout_every_n"):
            _parse_environment(raw)

    def test_rollout_every_n_range(self) -> None:
        raw = _minimal_env()
        raw["environment"]["rollout_every_n"] = 0  # type: ignore[index]
        with pytest.raises(ValueError, match="rollout_every_n"):
            _parse_environment(raw)

    def test_temperature_type_error(self) -> None:
        raw = _minimal_env()
        raw["environment"]["temperature"] = "bad"  # type: ignore[index]
        with pytest.raises(TypeError, match="temperature"):
            _parse_environment(raw)

    def test_temperature_negative(self) -> None:
        raw = _minimal_env()
        raw["environment"]["temperature"] = -1.0  # type: ignore[index]
        with pytest.raises(ValueError, match="temperature"):
            _parse_environment(raw)

    def test_temperature_int_cast(self) -> None:
        raw = _minimal_env()
        raw["environment"]["temperature"] = 1  # type: ignore[index]
        cfg = _parse_environment(raw)
        assert cfg is not None
        assert isinstance(cfg.temperature, float)
        assert cfg.temperature == 1.0

    def test_invalid_injection_surface_ref(self) -> None:
        raw = _minimal_env()
        raw["environment"]["injection_surface"] = "nonexistent"  # type: ignore[index]
        with pytest.raises(ValueError, match="nonexistent"):
            _parse_environment(raw)

    def test_invalid_target_function_ref(self) -> None:
        raw = _minimal_env()
        raw["environment"]["target"] = {  # type: ignore[index]
            "function": "nonexistent",
        }
        with pytest.raises(ValueError, match="nonexistent"):
            _parse_environment(raw)

    def test_missing_tools(self) -> None:
        raw = _minimal_env()
        del raw["environment"]["tools"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="tools"):
            _parse_environment(raw)

    def test_missing_target(self) -> None:
        raw = _minimal_env()
        del raw["environment"]["target"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="target"):
            _parse_environment(raw)

    def test_missing_task(self) -> None:
        raw = _minimal_env()
        del raw["environment"]["task"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="task"):
            _parse_environment(raw)

    def test_with_policy(self) -> None:
        raw = _minimal_env()
        raw["environment"]["policy"] = {  # type: ignore[index]
            "blocked_functions": ["dangerous"],
        }
        cfg = _parse_environment(raw)
        assert cfg is not None
        assert cfg.policy is not None
        assert cfg.policy.blocked_functions == ["dangerous"]

    def test_policy_not_table(self) -> None:
        raw = _minimal_env()
        raw["environment"]["policy"] = "bad"  # type: ignore[index]
        with pytest.raises(TypeError, match="policy"):
            _parse_environment(raw)

    def test_custom_scalars(self) -> None:
        raw = _minimal_env()
        raw["environment"]["injection_position"] = "infix"  # type: ignore[index]
        raw["environment"]["benign_expected_tools"] = ["calendar"]  # type: ignore[index]
        raw["environment"]["max_turns"] = 10  # type: ignore[index]
        raw["environment"]["max_gen_tokens"] = 300  # type: ignore[index]
        raw["environment"]["rollout_top_n"] = 16  # type: ignore[index]
        raw["environment"]["rollout_every_n"] = 3  # type: ignore[index]
        raw["environment"]["temperature"] = 0.5  # type: ignore[index]
        cfg = _parse_environment(raw)
        assert cfg is not None
        assert cfg.injection_position == "infix"
        assert cfg.benign_expected_tools == ["calendar"]
        assert cfg.max_turns == 10
        assert cfg.max_gen_tokens == 300
        assert cfg.rollout_top_n == 16
        assert cfg.rollout_every_n == 3
        assert cfg.temperature == 0.5

    def test_scenario_seeds_environment_defaults(self) -> None:
        cfg = _parse_environment({"environment": {"scenario": "share_doc"}})

        assert cfg is not None
        assert cfg.scenario == "share_doc"
        assert cfg.injection_surface == "read_document_content"
        assert cfg.injection_position == "infix"
        assert cfg.target.function == "share_drive_file"
        assert cfg.task.content

    def test_scenario_allows_scalar_overrides(self) -> None:
        cfg = _parse_environment({
            "environment": {
                "scenario": "share_doc",
                "max_turns": 9,
                "rollout_every_n": 3,
                "temperature": 0.2,
            },
        })

        assert cfg is not None
        assert cfg.scenario == "share_doc"
        assert cfg.max_turns == 9
        assert cfg.rollout_every_n == 3
        assert cfg.temperature == 0.2
        assert cfg.target.function == "share_drive_file"

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown environment scenario"):
            _parse_environment({"environment": {"scenario": "missing"}})

    def test_scenario_type_error(self) -> None:
        with pytest.raises(TypeError, match="scenario"):
            _parse_environment({"environment": {"scenario": 3}})


# ── _parse_tools ─────────────────────────────────────────────────────


class TestParseTools:
    def test_valid_tool(self) -> None:
        tools = _parse_tools([
            {"name": "t1", "description": "desc", "parameters": {"k": "str"}},
        ])
        assert len(tools) == 1
        assert tools[0].name == "t1"
        assert tools[0].description == "desc"
        assert tools[0].parameters == {"k": "str"}
        assert tools[0].result is None

    def test_tool_with_result(self) -> None:
        tools = _parse_tools([
            {
                "name": "t1",
                "description": "desc",
                "parameters": {},
                "result": "string",
            },
        ])
        assert tools[0].result == "string"

    def test_tool_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_tools(["not-a-dict"])

    def test_tool_missing_name(self) -> None:
        with pytest.raises(TypeError, match="name"):
            _parse_tools([{"description": "d", "parameters": {}}])

    def test_tool_description_defaults_empty(self) -> None:
        tools = _parse_tools([{"name": "t", "parameters": {}}])
        assert tools[0].description == ""

    def test_tool_parameters_not_table(self) -> None:
        with pytest.raises(TypeError, match="parameters"):
            _parse_tools([{"name": "t", "parameters": "bad"}])

    def test_tool_result_type_error(self) -> None:
        with pytest.raises(TypeError, match="result"):
            _parse_tools([{"name": "t", "parameters": {}, "result": 42}])


# ── _parse_target ────────────────────────────────────────────────────


class TestParseTarget:
    def test_minimal(self) -> None:
        target = _parse_target({"function": "send_email"})
        assert target.function == "send_email"
        assert target.required_args == []
        assert target.arg_contains == {}

    def test_full(self) -> None:
        target = _parse_target({
            "function": "send_email",
            "required_args": ["to", "body"],
            "arg_contains": {"to": "victim@"},
        })
        assert target.required_args == ["to", "body"]
        assert target.arg_contains == {"to": "victim@"}

    def test_missing_function(self) -> None:
        with pytest.raises(TypeError, match="function"):
            _parse_target({})

    def test_required_args_not_list(self) -> None:
        with pytest.raises(TypeError, match="required_args"):
            _parse_target({"function": "f", "required_args": "bad"})

    def test_arg_contains_not_table(self) -> None:
        with pytest.raises(TypeError, match="arg_contains"):
            _parse_target({"function": "f", "arg_contains": "bad"})


# ── _parse_task ──────────────────────────────────────────────────────


class TestParseTask:
    def test_valid(self) -> None:
        task = _parse_task({"content": "Do the thing"})
        assert task.content == "Do the thing"

    def test_missing_content(self) -> None:
        with pytest.raises(TypeError, match="content"):
            _parse_task({})


# ── _parse_policy (environment.policy) ───────────────────────────────


class TestParseEnvironmentPolicy:
    def test_valid(self) -> None:
        policy = _parse_policy({
            "blocked_functions": ["dangerous"],
            "require_confirmation": ["sensitive"],
            "arg_blocklist": {"email": ["evil.com"]},
        })
        assert policy.blocked_functions == ["dangerous"]
        assert policy.require_confirmation == ["sensitive"]
        assert policy.arg_blocklist == {"email": ["evil.com"]}

    def test_defaults(self) -> None:
        policy = _parse_policy({})
        assert policy.blocked_functions == []
        assert policy.require_confirmation == []
        assert policy.arg_blocklist == {}

    def test_blocked_not_list(self) -> None:
        with pytest.raises(TypeError, match="blocked_functions"):
            _parse_policy({"blocked_functions": "bad"})

    def test_confirm_not_list(self) -> None:
        with pytest.raises(TypeError, match="require_confirmation"):
            _parse_policy({"require_confirmation": "bad"})

    def test_arg_blocklist_not_table(self) -> None:
        with pytest.raises(TypeError, match="arg_blocklist"):
            _parse_policy({"arg_blocklist": "bad"})

    def test_arg_blocklist_value_not_list(self) -> None:
        with pytest.raises(TypeError, match="arg_blocklist"):
            _parse_policy({"arg_blocklist": {"key": "not-a-list"}})
