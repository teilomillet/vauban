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


from pydantic import PrivateAttr, Field

from vauban.config import resolve_api_config
from vauban.tracing import trace, init_weave, WeaveModel


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


def parse_float(text: str, default: Optional[float] = None) -> float:
    """Parse a float from arbitrary model output with a safe fallback."""

    source = str(text).strip()
    try:
        return float(source)
    except Exception:
        # Grab first number in the string
        import re

        match = re.search(r"[-+]?[0-9]*\\.?[0-9]+", source)
        if match:
            return float(match.group())
        if default is not None:
            return default
        raise ValueError(f"Could not parse float from: {text}")


class LLM(WeaveModel):
    """
    Minimal async LLM client that wraps OpenAI chat completions.
    - Uses response_format=json_object when force_json is True so parsing stays deterministic.
    - Accepts a parser callable to turn the raw content into a domain object.
    - Traced via Weave.
    """
    model_name: str
    system_prompt: str = ""
    force_json: bool = False
    api_key: Optional[str] = Field(default=None, exclude=True)
    base_url: Optional[str] = None
    weave_project: Optional[str] = Field(default=None, exclude=True)
    trace_label: Optional[str] = None
    
    # Exclude parser from serialization as it's a callable
    parser: Parser = Field(default=_default_parser, exclude=True)
    
    _client: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.model_name = self.model_name.replace("openai:", "")
        self.system_prompt = self.system_prompt.strip()
        self.api_key, self.base_url = resolve_api_config(self.api_key, self.base_url)
        
        if self.weave_project:
            init_weave(self.weave_project)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def _run_impl(self, prompt: str) -> MiniResult:
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
        
        if not response.choices:
            # Handle cases where provider returns None or empty list for choices
            raise ValueError(f"LLM returned no choices. Model: {self.model_name}. Response: {response}")
            
        msg = response.choices[0].message
        content = msg.content or ""

        usage = TokenUsage(
            request_tokens=getattr(response.usage, "prompt_tokens", 0),
            response_tokens=getattr(response.usage, "completion_tokens", 0),
        )

        data = self._parse_content(content)
        # No manual logging here - tracing wrapper handles inputs/outputs
        return MiniResult(data=data, usage=usage)

    @trace(name="LLM.Run")
    async def run(self, prompt: str) -> MiniResult:
        # We call the implementation directly. 
        # Since 'run' is decorated with @trace, it will create the span.
        # 'self' is captured as an input, so model_name and trace_label are visible in Weave.
        return await self._run_impl(prompt)

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
