"""API-based suffix evaluation against OpenAI-compatible endpoints."""

import json
import logging
import os
import urllib.error
import urllib.request

from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import ApiEvalConfig, ApiEvalEndpoint, TransferEvalResult

logger = logging.getLogger(__name__)


def evaluate_suffix_via_api(
    suffix_text: str,
    prompts: list[str],
    config: ApiEvalConfig,
    fallback_system_prompt: str | None = None,
) -> list[TransferEvalResult]:
    """Evaluate an adversarial suffix against API endpoints.

    For each endpoint and each prompt, sends ``prompt + " " + suffix_text``
    to the endpoint's ``/chat/completions`` API and checks whether the
    response contains refusal phrases.

    Args:
        suffix_text: The optimized adversarial suffix.
        prompts: Attack prompts to test.
        config: API evaluation configuration.
        fallback_system_prompt: Used when neither the endpoint nor the
            config-level system_prompt is set.

    Returns:
        One ``TransferEvalResult`` per endpoint.
    """
    results: list[TransferEvalResult] = []
    for endpoint in config.endpoints:
        result = _evaluate_endpoint(
            endpoint, suffix_text, prompts, config, fallback_system_prompt,
        )
        results.append(result)
    return results


def _evaluate_endpoint(
    endpoint: ApiEvalEndpoint,
    suffix_text: str,
    prompts: list[str],
    config: ApiEvalConfig,
    fallback_system_prompt: str | None,
) -> TransferEvalResult:
    """Evaluate a suffix against a single API endpoint."""
    # Resolve system prompt: endpoint > config > fallback
    system_prompt = (
        endpoint.system_prompt
        or config.system_prompt
        or fallback_system_prompt
    )

    # Resolve API key from environment
    api_key = os.environ.get(endpoint.api_key_env)
    if not api_key:
        logger.warning(
            "API key env var %s not set for endpoint %s — skipping",
            endpoint.api_key_env,
            endpoint.name,
        )
        return TransferEvalResult(
            model_id=endpoint.name,
            success_rate=0.0,
            eval_responses=[],
        )

    successes = 0
    responses: list[str] = []

    for prompt in prompts:
        response_text = _call_chat_api(
            endpoint=endpoint,
            api_key=api_key,
            prompt=f"{prompt} {suffix_text}",
            system_prompt=system_prompt,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        responses.append(response_text)
        if not _is_refusal(response_text):
            successes += 1

    success_rate = successes / len(prompts) if prompts else 0.0
    return TransferEvalResult(
        model_id=endpoint.name,
        success_rate=success_rate,
        eval_responses=responses,
    )


def _call_chat_api(
    endpoint: ApiEvalEndpoint,
    api_key: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    timeout: int,
) -> str:
    """Send a single chat completion request and return the response text.

    On any error (timeout, auth, rate limit, malformed response), logs a
    warning and returns a refusal-like string so the prompt counts as
    refused (conservative).
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": endpoint.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        choices = data.get("choices", [])
        if not choices:
            logger.warning(
                "Empty choices from %s — treating as refusal",
                endpoint.name,
            )
            return "I cannot assist with that request."
        return str(choices[0].get("message", {}).get("content", ""))
    except urllib.error.HTTPError as exc:
        logger.warning(
            "HTTP %d from %s: %s — treating as refusal",
            exc.code,
            endpoint.name,
            exc.reason,
        )
        return "I cannot assist with that request."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning(
            "Request to %s failed: %s — treating as refusal",
            endpoint.name,
            exc,
        )
        return "I cannot assist with that request."
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning(
            "Malformed response from %s: %s — treating as refusal",
            endpoint.name,
            exc,
        )
        return "I cannot assist with that request."


def _is_refusal(text: str) -> bool:
    """Check if the response contains refusal phrases."""
    lower = text.lower()
    return any(phrase.lower() in lower for phrase in DEFAULT_REFUSAL_PHRASES)
