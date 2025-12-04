from typing import Optional, Tuple
import os


def resolve_api_config(
    api_key: Optional[str] = None, base_url: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Resolve API key / base URL with sensible defaults:
    - prefer explicit args
    - fall back to OPENROUTER_API_KEY + its default URL if present
    - otherwise use OPENAI_API_KEY and OPENAI_BASE_URL (or OpenRouter as default)
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if base_url:
        url = base_url
    elif os.getenv("OPENROUTER_API_KEY"):
        url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    else:
        url = os.getenv("OPENAI_BASE_URL")

    return key, url


def get_smart_default_model() -> str:
    """
    Determine the best default model based on available API keys.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return "anthropic/claude-3.5-sonnet"
    return "gpt-4o"


# Centralized Defaults
DEFAULT_ATTACKER_MODEL = get_smart_default_model()
DEFAULT_STRATEGY_MODEL = DEFAULT_ATTACKER_MODEL
DEFAULT_STEALTH_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Pricing (per 1M tokens)
PRICING_INPUT_PER_1M = 2.50
PRICING_OUTPUT_PER_1M = 10.00
