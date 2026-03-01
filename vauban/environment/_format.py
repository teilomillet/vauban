"""Format tool definitions into system prompt suffix."""

from vauban.types import ToolSchema


def format_tools_for_prompt(tools: list[ToolSchema]) -> str:
    """Format tool schemas into a system prompt suffix.

    Produces a structured text block describing available tools that
    can be appended to the agent's system prompt.

    Args:
        tools: List of tool schemas to format.

    Returns:
        Formatted string describing all available tools.
    """
    if not tools:
        return ""

    lines: list[str] = ["\n\nYou have access to the following tools:\n"]
    for tool in tools:
        lines.append(f"### {tool.name}")
        lines.append(f"{tool.description}")
        if tool.parameters:
            lines.append("Parameters:")
            for param_name, param_type in tool.parameters.items():
                lines.append(f"  - {param_name}: {param_type}")
        else:
            lines.append("Parameters: none")
        lines.append("")

    lines.append(
        "To call a tool, respond with a JSON object in a <tool_call> tag:"
    )
    lines.append(
        '<tool_call>{"name": "tool_name", "arguments": {"key": "value"}}'
        "</tool_call>",
    )
    return "\n".join(lines)
