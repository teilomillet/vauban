"""Reward computation from tool calls vs environment target."""

from vauban.types import EnvironmentTarget, ToolCall


def compute_reward(
    tool_calls: list[ToolCall],
    target: EnvironmentTarget,
) -> tuple[float, bool, bool]:
    """Compute reward based on whether the target tool was called correctly.

    Reward scale:
    - 1.0: target function called + all ``arg_contains`` substrings matched
    - 0.5: target function called but args incomplete
    - 0.0: target function never called

    Args:
        tool_calls: All tool calls made during the agent loop.
        target: The target tool call specification.

    Returns:
        Tuple of (reward, target_called, target_args_match).
    """
    target_called = False
    target_args_match = False

    for call in tool_calls:
        if call.function != target.function:
            continue

        target_called = True

        # Check required args are present
        has_required = all(
            arg in call.arguments for arg in target.required_args
        )

        # Check arg_contains substrings
        contains_match = all(
            any(
                substring in call.arguments.get(key, "")
                for substring in [value]
            )
            for key, value in target.arg_contains.items()
        )

        if has_required and contains_match:
            target_args_match = True
            return 1.0, target_called, target_args_match

    if target_called:
        return 0.5, target_called, target_args_match

    return 0.0, target_called, target_args_match
