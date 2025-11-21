"""
Lightweight LLM client around OpenAI chat completions.
Keeps a minimal API that's compatible with the few methods we used from pydantic-ai:
- async run(...) -> result with .data and .usage()

Uses Weave Tracing for observability (inputs/outputs automatically captured).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from openai import AsyncOpenAI

from vauban.config import resolve_api_config
from vauban.tracing import trace, init_weave


@dataclass
class TokenUsage:
    request_tokens: int = 0
    response_tokens: int = 0


class MiniResult:
    def __init__(self, data: Any, usage: TokenUsage):
        self.data = data
        self._usage = usage

    def usage(self) -> TokenUsage:
        return self._usage


Parser = Callable[[str], Any]


def _default_parser(text: str) -> str:
    return text


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(text: str) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        # Grab first number in the string
        import re

        match = re.search(r"[-+]?[0-9]*\\.?[0-9]+", str(text))
        if match:
            return float(match.group())
        raise


class LLM:
    """
    Minimal async LLM client that wraps OpenAI chat completions.
    - Uses response_format=json_object when force_json is True so parsing stays deterministic.
    - Accepts a parser callable to turn the raw content into a domain object.
    - Traced via Weave.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        parser: Parser = _default_parser,
        force_json: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        weave_project: Optional[str] = None,
    ):
        self.model_name = model_name.replace("openai:", "")
        self.system_prompt = system_prompt.strip()
        self.parser = parser
        self.force_json = force_json
        self.api_key, self.base_url = resolve_api_config(api_key, base_url)
        self._client: Optional[AsyncOpenAI] = None

        if weave_project:
            init_weave(weave_project)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @trace
    async def run(self, prompt: str) -> MiniResult:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.force_json:
            params["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**params)
        msg = response.choices[0].message
        content = msg.content or ""

        usage = TokenUsage(
            request_tokens=getattr(response.usage, "prompt_tokens", 0),
            response_tokens=getattr(response.usage, "completion_tokens", 0),
        )

        data = self._parse_content(content)
        # No manual logging here - @trace handles inputs (prompt) and output (MiniResult)
        return MiniResult(data=data, usage=usage)

    def _parse_content(self, content: str) -> Any:
        if not self.force_json:
            return self.parser(content)

        try:
            payload = json.loads(content)
        except Exception:
            # If model echoed text around JSON, extract first JSON object
            import re

            match = re.search(r"\\{.*\\}", content, flags=re.S)
            payload = json.loads(match.group()) if match else {}
        return self.parser(payload)  # parser can accept dict as well

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

