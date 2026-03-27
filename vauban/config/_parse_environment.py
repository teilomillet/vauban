# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [environment] section of a TOML config."""

from typing import cast

from vauban.config._types import TomlDict
from vauban.types import (
    EnvironmentConfig,
    EnvironmentTarget,
    EnvironmentTask,
    ToolCallPolicy,
    ToolSchema,
)


def _parse_environment(raw: TomlDict) -> EnvironmentConfig | None:
    """Parse the optional [environment] section into an EnvironmentConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("environment")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[environment] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    env = cast("dict[str, object]", sec)

    # -- system_prompt (required) --
    system_prompt_raw = env.get("system_prompt")
    if not isinstance(system_prompt_raw, str):
        msg = "[environment].system_prompt is required and must be a string"
        raise TypeError(msg)

    # -- injection_surface (required) --
    injection_surface_raw = env.get("injection_surface")
    if not isinstance(injection_surface_raw, str):
        msg = (
            "[environment].injection_surface is required"
            " and must be a string"
        )
        raise TypeError(msg)

    # -- max_turns --
    max_turns_raw = env.get("max_turns", 6)
    if not isinstance(max_turns_raw, int):
        msg = (
            f"[environment].max_turns must be an integer,"
            f" got {type(max_turns_raw).__name__}"
        )
        raise TypeError(msg)
    if max_turns_raw < 1:
        msg = f"[environment].max_turns must be >= 1, got {max_turns_raw}"
        raise ValueError(msg)

    # -- max_gen_tokens --
    max_gen_tokens_raw = env.get("max_gen_tokens", 200)
    if not isinstance(max_gen_tokens_raw, int):
        msg = (
            f"[environment].max_gen_tokens must be an integer,"
            f" got {type(max_gen_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if max_gen_tokens_raw < 1:
        msg = (
            f"[environment].max_gen_tokens must be >= 1,"
            f" got {max_gen_tokens_raw}"
        )
        raise ValueError(msg)

    # -- rollout_top_n --
    rollout_top_n_raw = env.get("rollout_top_n", 8)
    if not isinstance(rollout_top_n_raw, int):
        msg = (
            f"[environment].rollout_top_n must be an integer,"
            f" got {type(rollout_top_n_raw).__name__}"
        )
        raise TypeError(msg)
    if rollout_top_n_raw < 1:
        msg = (
            f"[environment].rollout_top_n must be >= 1,"
            f" got {rollout_top_n_raw}"
        )
        raise ValueError(msg)

    # -- temperature --
    temperature_raw = env.get("temperature", 0.0)
    if not isinstance(temperature_raw, (int, float)):
        msg = (
            f"[environment].temperature must be a number,"
            f" got {type(temperature_raw).__name__}"
        )
        raise TypeError(msg)
    temperature_val = float(temperature_raw)
    if temperature_val < 0.0:
        msg = (
            f"[environment].temperature must be >= 0.0,"
            f" got {temperature_val}"
        )
        raise ValueError(msg)

    # -- tools (required, list of tables) --
    tools_raw = env.get("tools")
    if not isinstance(tools_raw, list):
        msg = "[environment].tools is required and must be a list of tables"
        raise TypeError(msg)
    tools_list: list[object] = list(tools_raw)
    tools = _parse_tools(tools_list)

    # -- target (required, table) --
    target_raw = env.get("target")
    if not isinstance(target_raw, dict):
        msg = "[environment].target is required and must be a table"
        raise TypeError(msg)
    target = _parse_target(cast("dict[str, object]", target_raw))

    # -- task (required, table) --
    task_raw = env.get("task")
    if not isinstance(task_raw, dict):
        msg = "[environment].task is required and must be a table"
        raise TypeError(msg)
    task = _parse_task(cast("dict[str, object]", task_raw))

    # -- policy (optional, table) --
    policy_raw = env.get("policy")
    policy: ToolCallPolicy | None = None
    if policy_raw is not None:
        if not isinstance(policy_raw, dict):
            msg = "[environment].policy must be a table"
            raise TypeError(msg)
        policy = _parse_policy(cast("dict[str, object]", policy_raw))

    # -- Cross-field validation --
    tool_names = {t.name for t in tools}
    if injection_surface_raw not in tool_names:
        msg = (
            f"[environment].injection_surface = {injection_surface_raw!r}"
            f" does not reference a defined tool."
            f" Available: {sorted(tool_names)}"
        )
        raise ValueError(msg)

    if target.function not in tool_names:
        msg = (
            f"[environment.target].function = {target.function!r}"
            f" does not reference a defined tool."
            f" Available: {sorted(tool_names)}"
        )
        raise ValueError(msg)

    return EnvironmentConfig(
        system_prompt=system_prompt_raw,
        tools=tools,
        target=target,
        task=task,
        injection_surface=injection_surface_raw,
        max_turns=max_turns_raw,
        max_gen_tokens=max_gen_tokens_raw,
        policy=policy,
        rollout_top_n=rollout_top_n_raw,
        temperature=temperature_val,
    )


def _parse_tools(tools_raw: list[object]) -> list[ToolSchema]:
    """Parse the [[environment.tools]] array."""
    tools: list[ToolSchema] = []
    for i, entry in enumerate(tools_raw):
        if not isinstance(entry, dict):
            msg = f"[[environment.tools]][{i}] must be a table"
            raise TypeError(msg)
        t = cast("dict[str, object]", entry)

        name = t.get("name")
        if not isinstance(name, str):
            msg = f"[[environment.tools]][{i}].name must be a string"
            raise TypeError(msg)

        description = t.get("description", "")
        if not isinstance(description, str):
            msg = f"[[environment.tools]][{i}].description must be a string"
            raise TypeError(msg)

        parameters_raw = t.get("parameters", {})
        if not isinstance(parameters_raw, dict):
            msg = f"[[environment.tools]][{i}].parameters must be a table"
            raise TypeError(msg)
        parameters: dict[str, str] = {
            str(k): str(v) for k, v in parameters_raw.items()
        }

        result_raw = t.get("result")
        result: str | None = None
        if result_raw is not None:
            if not isinstance(result_raw, str):
                msg = f"[[environment.tools]][{i}].result must be a string"
                raise TypeError(msg)
            result = result_raw

        tools.append(ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            result=result,
        ))

    return tools


def _parse_target(raw: dict[str, object]) -> EnvironmentTarget:
    """Parse the [environment.target] table."""
    function = raw.get("function")
    if not isinstance(function, str):
        msg = "[environment.target].function must be a string"
        raise TypeError(msg)

    required_args_raw = raw.get("required_args", [])
    if not isinstance(required_args_raw, list):
        msg = "[environment.target].required_args must be a list"
        raise TypeError(msg)
    required_args: list[str] = [str(a) for a in required_args_raw]

    arg_contains_raw = raw.get("arg_contains", {})
    if not isinstance(arg_contains_raw, dict):
        msg = "[environment.target].arg_contains must be a table"
        raise TypeError(msg)
    arg_contains: dict[str, str] = {
        str(k): str(v) for k, v in arg_contains_raw.items()
    }

    return EnvironmentTarget(
        function=function,
        required_args=required_args,
        arg_contains=arg_contains,
    )


def _parse_task(raw: dict[str, object]) -> EnvironmentTask:
    """Parse the [environment.task] table."""
    content = raw.get("content")
    if not isinstance(content, str):
        msg = "[environment.task].content must be a string"
        raise TypeError(msg)
    return EnvironmentTask(content=content)


def _parse_policy(raw: dict[str, object]) -> ToolCallPolicy:
    """Parse the [environment.policy] table."""
    blocked_raw = raw.get("blocked_functions", [])
    if not isinstance(blocked_raw, list):
        msg = "[environment.policy].blocked_functions must be a list"
        raise TypeError(msg)
    blocked: list[str] = [str(f) for f in blocked_raw]

    confirm_raw = raw.get("require_confirmation", [])
    if not isinstance(confirm_raw, list):
        msg = "[environment.policy].require_confirmation must be a list"
        raise TypeError(msg)
    confirm: list[str] = [str(f) for f in confirm_raw]

    blocklist_raw = raw.get("arg_blocklist", {})
    if not isinstance(blocklist_raw, dict):
        msg = "[environment.policy].arg_blocklist must be a table"
        raise TypeError(msg)
    arg_blocklist: dict[str, list[str]] = {}
    for key, val in blocklist_raw.items():
        if not isinstance(val, list):
            msg = f"[environment.policy].arg_blocklist.{key} must be a list"
            raise TypeError(msg)
        arg_blocklist[str(key)] = [str(v) for v in val]

    return ToolCallPolicy(
        blocked_functions=blocked,
        require_confirmation=confirm,
        arg_blocklist=arg_blocklist,
    )
