from typing import Optional, List, Dict, Any, Union
import json
from openai import AsyncOpenAI, OpenAI

from vauban.interfaces import Target, TargetResponse
from vauban.tools import RISKY_TOOLS
from vauban.config import resolve_api_config
from vauban.tracing import trace


class ModelTarget(Target):
    """
    Default target for any OpenAI-compatible API (defaults to OpenRouter via resolve_api_config).
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.api_key, self.base_url = resolve_api_config(api_key, base_url)
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None
        self._aclient: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._aclient

    def invoke(self, prompt: str) -> str:
        """Synchronous invocation (legacy support)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"

    @trace
    async def invoke_async(self, prompt: str) -> Union[str, TargetResponse]:
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"


# Alias kept so existing imports of OpenAITarget remain functional while the class is vendor-neutral.
OpenAITarget = ModelTarget


class MockAgentTarget(Target):
    """
    Target that simulates an Agent with Tool Use capabilities.
    """

    def __init__(
        self,
        model_name: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        injection_payload: Optional[str] = None,
        mock_behaviors: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.tools = tools if tools is not None else RISKY_TOOLS
        self.system_prompt = system_prompt
        self.injection_payload = injection_payload
        self.mock_behaviors = mock_behaviors or {}
        self.api_key, self.base_url = resolve_api_config(api_key, base_url)
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None
        self._aclient: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._aclient

    def _resolve_mock_response(self, tool_name: str, tool_args: str) -> str:
        """
        Resolve the mock response for a given tool call.
        """
        if tool_name not in self.mock_behaviors:
            return json.dumps(
                {"status": "error", "message": f"Tool {tool_name} not mocked."}
            )

        template = self.mock_behaviors[tool_name]

        # Parse args
        try:
            args = json.loads(tool_args)
        except Exception:
            args = {}

        # 1. Replace injection payload if present
        if self.injection_payload:
            template = template.replace("{fill}", self.injection_payload)

        # 2. Simple template substitution for arguments
        for key, value in args.items():
            placeholder = f"{{{key}}}"
            if placeholder in template:
                template = template.replace(placeholder, str(value))

        return template

    async def _handle_tool_calls_async(
        self, messages: List[Dict[str, str]], tool_calls: List[Any]
    ) -> Optional[Any]:
        """
        Simulate tool execution loop (Async).
        """
        messages = messages.copy()

        messages.append({"role": "assistant", "tool_calls": tool_calls})

        has_mocked_tool = False
        for tool_call in tool_calls:
            name = tool_call.function.name
            if name in self.mock_behaviors:
                has_mocked_tool = True
                content = self._resolve_mock_response(
                    name, tool_call.function.arguments
                )

                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": content}
                )

        if has_mocked_tool:
            try:
                response = await self.aclient.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message
            except Exception:
                return None

        return None

    @trace
    async def invoke_async(self, prompt: str) -> TargetResponse:
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
            )
            msg = response.choices[0].message

            tool_calls_data = []
            if msg.tool_calls:
                final_msg = await self._handle_tool_calls_async(
                    messages, msg.tool_calls
                )
                if final_msg:
                    msg = final_msg

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_data.append(
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                                "type": tc.type,
                            }
                        )

            return TargetResponse(content=msg.content, tool_calls=tool_calls_data)
        except Exception as e:
            return TargetResponse(content=f"ERROR: {e}")
