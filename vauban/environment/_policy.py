"""ToolCallPolicy enforcement for the environment harness."""

from vauban.types import ToolCall, ToolCallPolicy


def check_policy(
    tool_call: ToolCall,
    policy: ToolCallPolicy | None,
) -> str | None:
    """Check a tool call against the environment policy.

    Args:
        tool_call: The tool call to check.
        policy: Policy rules (None = allow all).

    Returns:
        None if allowed, or a string reason for blocking.
    """
    if policy is None:
        return None

    # Blocked functions
    if tool_call.function in policy.blocked_functions:
        return f"Function {tool_call.function!r} is blocked by policy"

    # Argument blocklist
    if tool_call.function in policy.arg_blocklist:
        blocked_values = policy.arg_blocklist[tool_call.function]
        for arg_value in tool_call.arguments.values():
            for blocked in blocked_values:
                if blocked in arg_value:
                    return (
                        f"Argument value contains blocked substring"
                        f" {blocked!r} for {tool_call.function!r}"
                    )

    # Require confirmation (in the harness, treated as "allow with log")
    if tool_call.function in policy.require_confirmation:
        return None  # harness doesn't block, just flags

    return None
