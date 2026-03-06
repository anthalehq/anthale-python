"""
Anthale middleware and utilities for Anthropic Python SDK.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec(name="anthropic") is None:
    raise ImportError("Anthropic is not installed. Please install it using 'pip install \"anthale[anthropic]\"'.")

from json import loads
from typing import Any, Mapping, TypeVar
from inspect import iscoroutinefunction
from warnings import warn

from httpx import Client, Request, Response, AsyncClient
from anthropic import Anthropic, AsyncAnthropic, APIConnectionError

from anthale.types.organizations.policy_enforce_params import Message

from .core import SyncPolicyEnforcer, AsyncPolicyEnforcer, AnthalePolicyViolationError, build_enforcers
from ._messages import stringify, normalize_role


class _AnthalePolicyViolationSignal(BaseException):
    """
    Private BaseException wrapper used to propagate AnthalePolicyViolationError through
    Anthropic's internal retry loop without triggering retries.
    """

    _error: AnthalePolicyViolationError

    def __init__(self, *, error: AnthalePolicyViolationError) -> None:
        self._error = error
        super().__init__(str(error))


__all__ = (
    "guard_anthropic_client",
    "guard_client",
)


_AnthropicClient = TypeVar("_AnthropicClient", Anthropic, AsyncAnthropic)
_stream_warning_issued: bool = False


def _extract_content(*, raw: Any) -> str:
    """
    Normalize Anthropic message content into plain text.
    """
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:  # type: ignore
            text = _extract_content(raw=item)
            if text:
                parts.append(text)

        return "\n".join(parts)

    if isinstance(raw, Mapping):
        block_type = str(raw.get("type", "")).lower()  # type: ignore
        if "tool" in block_type:
            return ""

        if raw.get("text") is not None:  # type: ignore
            return _extract_content(raw=raw.get("text"))  # type: ignore

        if raw.get("input_text") is not None:  # type: ignore
            return _extract_content(raw=raw.get("input_text"))  # type: ignore

        if raw.get("output_text") is not None:  # type: ignore
            return _extract_content(raw=raw.get("output_text"))  # type: ignore

        if raw.get("content") is not None:  # type: ignore
            return _extract_content(raw=raw.get("content"))  # type: ignore

        if raw.get("delta") is not None:  # type: ignore
            return _extract_content(raw=raw.get("delta"))  # type: ignore

        return ""

    return stringify(value=raw)


def _to_message(*, value: Mapping[str, Any], default_role: str) -> Message | None:
    """
    Convert a mapping into an Anthale Message.
    """
    role = str(value.get("role", default_role))  # type: ignore
    content = _extract_content(
        raw=value.get("content")  # type: ignore
        or value.get("text")  # type: ignore
        or value.get("input_text")  # type: ignore
        or value.get("output_text")  # type: ignore
    )

    if not content or content.strip() in ("", "None"):
        return None

    return Message(role=normalize_role(value=role), content=content)


def _messages_from_value(*, value: Any, default_role: str) -> list[Message]:
    """
    Recursively convert values into Anthale Message objects.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        out: list[Message] = []
        for item in value:  # type: ignore
            out.extend(_messages_from_value(value=item, default_role=default_role))
        return out

    if isinstance(value, Mapping):
        message = _to_message(value=value, default_role=default_role)  # type: ignore
        return [] if message is None else [message]

    content = value if isinstance(value, str) else stringify(value=value)
    if not content or content.strip() in ("", "None"):
        return []

    return [Message(role=normalize_role(value=default_role), content=content)]


def _extract_request_json(*, request: Request) -> Mapping[str, Any] | None:
    content = getattr(request, "content", None)
    if not content:
        return None

    raw = content.decode("utf-8", "replace") if isinstance(content, bytes) else str(content)
    try:
        parsed = loads(raw)
    except Exception:
        return None

    return parsed if isinstance(parsed, Mapping) else None  # type: ignore


def _extract_response_json(*, response: Response) -> Mapping[str, Any] | None:
    try:
        parsed = response.json()
    except Exception:
        return None

    return parsed if isinstance(parsed, Mapping) else None  # type: ignore


def _extract_messages_from_request(*, request: Request) -> list[Message]:
    payload = _extract_request_json(request=request)
    if payload is None:
        return []

    path = str(getattr(getattr(request, "url", None), "path", ""))
    if not path.endswith("/messages"):
        return []

    messages: list[Message] = []
    if payload.get("system") is not None:
        messages.extend(_messages_from_value(value=payload.get("system"), default_role="system"))

    messages.extend(_messages_from_value(value=payload.get("messages"), default_role="user"))
    return messages


def _extract_messages_from_response(*, request_path: str, response: Response) -> list[Message]:
    payload = _extract_response_json(response=response)
    if payload is None:
        return []

    if not request_path.endswith("/messages"):
        return []

    role = str(payload.get("role", "assistant"))  # type: ignore
    return _messages_from_value(value={"role": role, "content": payload.get("content")}, default_role=role)


def _parse_sse_payloads(*, raw: str) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    data_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "":
            if data_lines:
                data = "\n".join(data_lines).strip()
                data_lines = []
                if data == "[DONE]":
                    continue

                try:
                    parsed = loads(data)
                except Exception:
                    continue

                if isinstance(parsed, Mapping):
                    payloads.append(parsed)  # type: ignore

            continue

        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())

    if data_lines:
        data = "\n".join(data_lines).strip()
        if data != "[DONE]":
            try:
                parsed = loads(data)
            except Exception:
                parsed = None

            if isinstance(parsed, Mapping):
                payloads.append(parsed)  # type: ignore

    return payloads


def _extract_messages_from_stream_payloads(*, request_path: str, payloads: list[Mapping[str, Any]]) -> list[Message]:
    if not request_path.endswith("/messages"):
        return []

    content_parts: list[str] = []
    for payload in payloads:
        event_type = str(payload.get("type", ""))
        if event_type == "content_block_start":
            text = _extract_content(raw=payload.get("content_block"))
            if text:
                content_parts.append(text)
            continue

        if event_type == "content_block_delta":
            text = _extract_content(raw=payload.get("delta"))
            if text:
                content_parts.append(text)

    if not content_parts:
        return []

    return [Message(role="assistant", content="".join(content_parts))]


def _extract_messages_from_streaming_response(*, request_path: str, response: Response) -> list[Message]:
    try:
        raw = response.text
    except Exception:
        return []

    payloads = _parse_sse_payloads(raw=raw)
    if not payloads:
        return []

    return _extract_messages_from_stream_payloads(request_path=request_path, payloads=payloads)


class _AnthaleSyncHTTPClient:
    _inner: Client
    _enforcer: SyncPolicyEnforcer
    _anthale_guarded: bool

    def __init__(self, *, inner: Client, enforcer: SyncPolicyEnforcer) -> None:
        self._inner = inner
        self._enforcer = enforcer
        self._anthale_guarded = True

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def send(self, request: Request, **kwargs: Any) -> Any:
        request_messages = _extract_messages_from_request(request=request)
        if request_messages:
            try:
                self._enforcer.enforce(direction="input", messages=request_messages)
            except AnthalePolicyViolationError as error:
                raise _AnthalePolicyViolationSignal(error=error) from error

        path = str(getattr(getattr(request, "url", None), "path", ""))
        response = self._inner.send(request, **kwargs)

        if kwargs.get("stream"):
            global _stream_warning_issued
            if not _stream_warning_issued:
                warn(
                    message="Anthale does not support real-time stream analysis. Anthropic stream outputs are buffered and analyzed once the stream completes.",
                    category=UserWarning,
                    stacklevel=3,
                )
                _stream_warning_issued = True

            try:
                response.read()
            except Exception:
                return response

            response_messages = _extract_messages_from_streaming_response(request_path=path, response=response)
            if response_messages:
                try:
                    self._enforcer.enforce(direction="output", messages=request_messages + response_messages)
                except AnthalePolicyViolationError as error:
                    raise _AnthalePolicyViolationSignal(error=error) from error

            return response

        response_messages = _extract_messages_from_response(request_path=path, response=response)
        if response_messages:
            try:
                self._enforcer.enforce(direction="output", messages=request_messages + response_messages)
            except AnthalePolicyViolationError as error:
                raise _AnthalePolicyViolationSignal(error=error) from error

        return response


class _AnthaleAsyncHTTPClient:
    _inner: AsyncClient
    _enforcer: AsyncPolicyEnforcer
    _anthale_guarded: bool

    def __init__(self, *, inner: AsyncClient, enforcer: AsyncPolicyEnforcer) -> None:
        self._inner = inner
        self._enforcer = enforcer
        self._anthale_guarded = True

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def send(self, request: Request, **kwargs: Any) -> Any:
        request_messages = _extract_messages_from_request(request=request)
        if request_messages:
            try:
                await self._enforcer.enforce(direction="input", messages=request_messages)
            except AnthalePolicyViolationError as error:
                raise _AnthalePolicyViolationSignal(error=error) from error

        path = str(getattr(getattr(request, "url", None), "path", ""))
        response = await self._inner.send(request, **kwargs)

        if kwargs.get("stream"):
            global _stream_warning_issued
            if not _stream_warning_issued:
                warn(
                    message="Anthale does not support real-time stream analysis. Anthropic stream outputs are buffered and analyzed once the stream completes.",
                    category=UserWarning,
                    stacklevel=3,
                )
                _stream_warning_issued = True

            try:
                aread = getattr(response, "aread", None)
                if callable(aread):
                    await aread()
                else:
                    response.read()
            except Exception:
                return response

            response_messages = _extract_messages_from_streaming_response(request_path=path, response=response)
            if response_messages:
                try:
                    await self._enforcer.enforce(direction="output", messages=request_messages + response_messages)
                except AnthalePolicyViolationError as error:
                    raise _AnthalePolicyViolationSignal(error=error) from error

            return response

        response_messages = _extract_messages_from_response(request_path=path, response=response)
        if response_messages:
            try:
                await self._enforcer.enforce(direction="output", messages=request_messages + response_messages)
            except AnthalePolicyViolationError as error:
                raise _AnthalePolicyViolationSignal(error=error) from error

        return response


def _patch_sync_request(*, anthropic_client: Anthropic) -> None:
    original_request = anthropic_client.request

    def patched_request(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_request(*args, **kwargs)  # type: ignore
        except _AnthalePolicyViolationSignal as signal:
            raise signal._error from signal._error.__cause__
        except APIConnectionError as error:
            if isinstance(error.__cause__, AnthalePolicyViolationError):
                raise error.__cause__ from error.__cause__.__cause__
            raise

    anthropic_client.request = patched_request  # type: ignore[method-assign]


def _patch_async_request(*, anthropic_client: AsyncAnthropic) -> None:
    original_request = anthropic_client.request

    async def patched_request(*args: Any, **kwargs: Any) -> Any:
        try:
            return await original_request(*args, **kwargs)
        except _AnthalePolicyViolationSignal as signal:
            raise signal._error from signal._error.__cause__
        except APIConnectionError as error:
            if isinstance(error.__cause__, AnthalePolicyViolationError):
                raise error.__cause__ from error.__cause__.__cause__
            raise

    anthropic_client.request = patched_request  # type: ignore[method-assign]


def guard_anthropic_client(
    anthropic_client: _AnthropicClient,
    *,
    policy_id: str,
    api_key: str | None = None,
    client: Any | None = None,
    async_client: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> _AnthropicClient:
    """
    Guard an existing Anthropic client and return the same instance.

    Args:
        anthropic_client (_AnthropicClient): Anthropic `Anthropic` or `AsyncAnthropic` client instance.
        policy_id (str): Anthale policy identifier.
        api_key (str | None): Anthale API key if Anthale clients are not explicitly provided.
        client (Any | None): Optional prebuilt sync Anthale client.
        async_client (Any | None): Optional prebuilt async Anthale client.
        metadata (Mapping[str, Any] | None): Optional metadata sent with each enforcement request.

    Returns:
        _AnthropicClient: The same `anthropic_client` instance, now instrumented with Anthale policy enforcement.

    Example:
    ```python
    from os import environ
    from anthropic import Anthropic
    from anthale.integrations.anthropic import guard_anthropic_client

    client = Anthropic(api_key=environ["ANTHROPIC_API_KEY"])
    client = guard_anthropic_client(client, policy_id="<your-policy-identifier>", api_key=environ["ANTHALE_API_KEY"])
    ```
    """
    sync_enforcer, async_enforcer = build_enforcers(
        policy_id=policy_id,
        api_key=api_key,
        client=client,
        async_client=async_client,
        metadata=metadata,
    )

    inner_http = getattr(anthropic_client, "_client", None)
    if inner_http is None:
        raise TypeError("Anthropic client must expose an internal '_client' HTTP client.")

    if getattr(inner_http, "_anthale_guarded", False):
        return anthropic_client

    send = getattr(inner_http, "send", None)
    if send is None:
        raise TypeError("Anthropic client internal HTTP client must expose a 'send' method.")

    if iscoroutinefunction(send):
        if async_enforcer is None:
            raise ValueError("Async Anthropic client requires an async Anthale enforcer.")

        anthropic_client._client = _AnthaleAsyncHTTPClient(inner=inner_http, enforcer=async_enforcer)  # type: ignore
        _patch_async_request(anthropic_client=anthropic_client)  # type: ignore[arg-type]
        return anthropic_client

    if sync_enforcer is None:
        raise ValueError("Sync Anthropic client requires a sync Anthale enforcer.")

    anthropic_client._client = _AnthaleSyncHTTPClient(inner=inner_http, enforcer=sync_enforcer)  # type: ignore
    _patch_sync_request(anthropic_client=anthropic_client)  # type: ignore[arg-type]
    return anthropic_client


guard_client = guard_anthropic_client
