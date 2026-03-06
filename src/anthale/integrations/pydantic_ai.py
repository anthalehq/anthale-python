"""
Anthale middleware and utilities for PydanticAI models.
"""

from __future__ import annotations

from typing import Any, Mapping, AsyncIterator
from contextlib import asynccontextmanager
from importlib.util import find_spec

if find_spec(name="pydantic_ai") is None:
    raise ImportError("PydanticAI is not installed. Please install it using 'pip install anthale pydantic-ai'.")

from pydantic_ai.models import ModelSettings, StreamedResponse, ModelRequestParameters
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models.wrapper import WrapperModel

from anthale.types.organizations.policy_enforce_params import Message

from .core import AsyncPolicyEnforcer, build_enforcers
from ._messages import stringify

__all__ = (
    "AnthalePydanticAIModel",
    "guard_pydantic_ai_model",
    "guard_model",
)


def _extract_content(*, value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:  # type: ignore
            text = _extract_content(value=item)
            if text:
                parts.append(text)
        return "\n".join(parts)

    if isinstance(value, Mapping):
        if value.get("content") is not None:  # type: ignore
            return _extract_content(value=value.get("content"))  # type: ignore

        if value.get("text") is not None:  # type: ignore
            return _extract_content(value=value.get("text"))  # type: ignore

        return ""

    return stringify(value=value)


def _extract_messages_from_model_request(*, request: Any) -> list[Message]:
    out: list[Message] = []
    for part in getattr(request, "parts", []):
        part_kind = str(getattr(part, "part_kind", ""))
        content = _extract_content(value=getattr(part, "content", None))
        if not content or content.strip() in ("", "None"):
            continue

        if part_kind == "system-prompt":
            out.append(Message(role="system", content=content))
            continue

        if part_kind in ("user-prompt", "retry-prompt"):
            out.append(Message(role="user", content=content))
            continue

    return out


def _extract_messages_from_model_response(*, response: Any) -> list[Message]:
    out: list[Message] = []
    for part in getattr(response, "parts", []):
        part_kind = str(getattr(part, "part_kind", ""))
        if part_kind not in ("text", "thinking"):
            continue

        content = _extract_content(value=getattr(part, "content", None))
        if not content or content.strip() in ("", "None"):
            continue

        out.append(Message(role="assistant", content=content))

    return out


def _extract_messages_from_model_messages(*, messages: list[ModelMessage]) -> list[Message]:
    out: list[Message] = []
    for message in messages:
        if isinstance(message, ModelRequest) or getattr(message, "kind", None) == "request":
            out.extend(_extract_messages_from_model_request(request=message))
            continue

        if isinstance(message, ModelResponse) or getattr(message, "kind", None) == "response":
            out.extend(_extract_messages_from_model_response(response=message))

    return out


class _AnthaleStreamedResponse:
    _stream: StreamedResponse
    _enforcer: AsyncPolicyEnforcer
    _request_messages: list[Message]
    _enforced: bool

    def __init__(
        self,
        *,
        stream: StreamedResponse,
        enforcer: AsyncPolicyEnforcer,
        request_messages: list[Message],
    ) -> None:
        self._stream = stream
        self._enforcer = enforcer
        self._request_messages = request_messages
        self._enforced = False

    async def _ensure_enforced(self) -> None:
        if self._enforced:
            return

        self._enforced = True
        try:
            response = self._stream.get()
        except Exception:
            return

        response_messages = _extract_messages_from_model_response(response=response)
        if response_messages:
            await self._enforcer.enforce(
                direction="output",
                messages=self._request_messages + response_messages,
            )

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _iterator() -> AsyncIterator[Any]:
            async for event in self._stream:
                yield event
            await self._ensure_enforced()

        return _iterator()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)


class AnthalePydanticAIModel(WrapperModel):
    """
    Wrapper model that enforces Anthale policy on PydanticAI model requests and responses.

    Example:
    ```python
    from os import environ
    from pydantic_ai import Agent
    from anthale.integrations.pydantic_ai import AnthalePydanticAIModel

    model = AnthalePydanticAIModel(
        "openai:gpt-5-nano",
        policy_id="<your-policy-identifier>",
        api_key=environ["ANTHALE_API_KEY"],
    )
    agent = Agent(model=model)
    ```
    """

    _async_enforcer: AsyncPolicyEnforcer

    def __init__(
        self,
        wrapped: Any,
        *,
        policy_id: str,
        api_key: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(wrapped)

        _sync_enforcer, async_enforcer = build_enforcers(
            policy_id=policy_id,
            api_key=api_key,
            client=client,
            async_client=async_client,
            metadata=metadata,
        )
        if async_enforcer is None:
            raise ValueError("PydanticAI integration requires an async Anthale enforcer.")

        self._async_enforcer = async_enforcer

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        request_messages = _extract_messages_from_model_messages(messages=messages)
        if request_messages:
            await self._async_enforcer.enforce(direction="input", messages=request_messages)

        response = await self.wrapped.request(messages, model_settings, model_request_parameters)

        response_messages = _extract_messages_from_model_response(response=response)
        if response_messages:
            await self._async_enforcer.enforce(
                direction="output",
                messages=request_messages + response_messages,
            )

        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        request_messages = _extract_messages_from_model_messages(messages=messages)
        if request_messages:
            await self._async_enforcer.enforce(direction="input", messages=request_messages)

        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response_stream:
            wrapped_stream = _AnthaleStreamedResponse(
                stream=response_stream,
                enforcer=self._async_enforcer,
                request_messages=request_messages,
            )
            try:
                yield wrapped_stream  # type: ignore[misc]
            finally:
                await wrapped_stream._ensure_enforced()


def guard_pydantic_ai_model(
    model: Any,
    *,
    policy_id: str,
    api_key: str | None = None,
    client: Any | None = None,
    async_client: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AnthalePydanticAIModel:
    """
    Wrap a PydanticAI model with Anthale policy enforcement.

    Args:
        model (Any): Model instance or known model name accepted by `WrapperModel`.
        policy_id (str): Anthale policy identifier.
        api_key (str | None): Anthale API key if Anthale clients are not explicitly provided.
        client (Any | None): Optional prebuilt sync Anthale client.
        async_client (Any | None): Optional prebuilt async Anthale client.
        metadata (Mapping[str, Any] | None): Optional metadata sent with each enforcement request.

    Returns:
        AnthalePydanticAIModel: Guarded wrapper model.
    """
    return AnthalePydanticAIModel(
        model,
        policy_id=policy_id,
        api_key=api_key,
        client=client,
        async_client=async_client,
        metadata=metadata,
    )


guard_model = guard_pydantic_ai_model
