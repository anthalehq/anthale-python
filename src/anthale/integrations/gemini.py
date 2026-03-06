"""
Anthale middleware and utilities for Google Gen AI (Gemini) Python SDK.
"""

from __future__ import annotations

from importlib.util import find_spec

try:
    _google_genai_spec = find_spec(name="google.genai")
except ModuleNotFoundError:
    _google_genai_spec = None

if _google_genai_spec is None:
    raise ImportError("Google Gen AI SDK is not installed. Please install it using 'pip install anthale google-genai'.")

from json import loads
from typing import Any, Mapping, TypeVar
from warnings import warn

from httpx import URL, Client, Request, Response, AsyncClient

from anthale.types.organizations.policy_enforce_params import Message

from .core import SyncPolicyEnforcer, AsyncPolicyEnforcer, build_enforcers
from ._messages import stringify, normalize_role

__all__ = (
    "guard_gemini_client",
    "guard_client",
)

_GeminiClient = TypeVar("_GeminiClient")
_stream_warning_issued: bool = False


def _request_path_from_url(*, url: Any) -> str:
    try:
        return str(URL(str(url)).path)
    except Exception:
        return str(url)


def _extract_text_from_part(*, value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:  # type: ignore
            text = _extract_text_from_part(value=item)
            if text:
                parts.append(text)
        return "\n".join(parts)

    if isinstance(value, Mapping):
        if value.get("text") is not None:  # type: ignore
            return _extract_text_from_part(value=value.get("text"))  # type: ignore

        if value.get("parts") is not None:  # type: ignore
            return _extract_text_from_part(value=value.get("parts"))  # type: ignore

        return ""

    return stringify(value=value)


def _to_message_from_content(*, value: Any, default_role: str) -> Message | None:
    if isinstance(value, Mapping):
        role = normalize_role(value=str(value.get("role", default_role)))  # type: ignore
        content = _extract_text_from_part(value=value.get("parts"))  # type: ignore
    else:
        role = normalize_role(value=default_role)
        content = _extract_text_from_part(value=value)

    if not content or content.strip() in ("", "None"):
        return None

    return Message(role=role, content=content)


def _messages_from_content_value(*, value: Any, default_role: str) -> list[Message]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        out: list[Message] = []
        for item in value:  # type: ignore
            out.extend(_messages_from_content_value(value=item, default_role=default_role))
        return out

    message = _to_message_from_content(value=value, default_role=default_role)
    return [] if message is None else [message]


def _is_generate_content_path(*, path: str) -> bool:
    return (
        ":generateContent" in path
        or ":streamGenerateContent" in path
        or path.endswith("/generateContent")
        or path.endswith("/streamGenerateContent")
    )


def _extract_messages_from_generate_content_request(*, path: str, payload: Mapping[str, Any] | None) -> list[Message]:
    if payload is None or not _is_generate_content_path(path=path):
        return []

    messages: list[Message] = []

    system_instruction = payload.get("systemInstruction") or payload.get("system_instruction")
    if system_instruction is not None:
        messages.extend(_messages_from_content_value(value=system_instruction, default_role="system"))

    messages.extend(_messages_from_content_value(value=payload.get("contents"), default_role="user"))
    return messages


def _extract_messages_from_generate_content_response(*, path: str, payload: Mapping[str, Any] | None) -> list[Message]:
    if payload is None or not _is_generate_content_path(path=path):
        return []

    messages: list[Message] = []
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue

            content = candidate.get("content")
            message = _to_message_from_content(value=content, default_role="assistant")
            if message is not None:
                messages.append(message)

    return messages


def _load_mapping_from_payload(*, payload: Any) -> Mapping[str, Any] | None:
    if payload is None:
        return None

    if isinstance(payload, Mapping):
        return payload  # type: ignore

    raw = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else str(payload)
    try:
        parsed = loads(raw)
    except Exception:
        return None

    return parsed if isinstance(parsed, Mapping) else None  # type: ignore


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

                parsed = _load_mapping_from_payload(payload=data)
                if parsed is not None:
                    payloads.append(parsed)
            continue

        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
        else:
            # Gemini may emit raw JSON per-line depending on endpoint behavior.
            parsed = _load_mapping_from_payload(payload=stripped)
            if parsed is not None:
                payloads.append(parsed)

    if data_lines:
        data = "\n".join(data_lines).strip()
        if data != "[DONE]":
            parsed = _load_mapping_from_payload(payload=data)
            if parsed is not None:
                payloads.append(parsed)

    return payloads


def _extract_messages_from_streaming_response(*, path: str, response: Response) -> list[Message]:
    try:
        raw = response.text
    except Exception:
        return []

    payloads = _parse_sse_payloads(raw=raw)
    if not payloads:
        return []

    chunks: list[str] = []
    for payload in payloads:
        for message in _extract_messages_from_generate_content_response(path=path, payload=payload):
            if isinstance(message, Mapping):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)

            if isinstance(content, str) and content:
                chunks.append(content)

    if not chunks:
        return []

    return [Message(role="assistant", content="".join(chunks))]


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

    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        path = _request_path_from_url(url=url)
        request_payload = _load_mapping_from_payload(payload=kwargs.get("json") or kwargs.get("content"))
        request_messages = _extract_messages_from_generate_content_request(path=path, payload=request_payload)
        if request_messages:
            self._enforcer.enforce(direction="input", messages=request_messages)

        response = self._inner.request(method, url, **kwargs)
        response_messages = _extract_messages_from_generate_content_response(
            path=path, payload=_load_mapping_from_payload(payload=response.text)
        )
        if response_messages:
            self._enforcer.enforce(direction="output", messages=request_messages + response_messages)

        return response

    def send(self, request: Request, **kwargs: Any) -> Any:
        path = str(getattr(getattr(request, "url", None), "path", ""))
        request_messages = _extract_messages_from_generate_content_request(
            path=path,
            payload=_load_mapping_from_payload(payload=getattr(request, "content", None)),
        )
        if request_messages:
            self._enforcer.enforce(direction="input", messages=request_messages)

        response = self._inner.send(request, **kwargs)
        if kwargs.get("stream"):
            global _stream_warning_issued
            if not _stream_warning_issued:
                warn(
                    message="Anthale does not support real-time stream analysis. Gemini stream outputs are buffered and analyzed once the stream completes.",
                    category=UserWarning,
                    stacklevel=3,
                )
                _stream_warning_issued = True

            try:
                response.read()
            except Exception:
                return response

            response_messages = _extract_messages_from_streaming_response(path=path, response=response)
            if response_messages:
                self._enforcer.enforce(direction="output", messages=request_messages + response_messages)
            return response

        response_messages = _extract_messages_from_generate_content_response(
            path=path, payload=_load_mapping_from_payload(payload=response.text)
        )
        if response_messages:
            self._enforcer.enforce(direction="output", messages=request_messages + response_messages)

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

    async def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        path = _request_path_from_url(url=url)
        request_payload = _load_mapping_from_payload(payload=kwargs.get("json") or kwargs.get("content"))
        request_messages = _extract_messages_from_generate_content_request(path=path, payload=request_payload)
        if request_messages:
            await self._enforcer.enforce(direction="input", messages=request_messages)

        response = await self._inner.request(method, url, **kwargs)
        response_messages = _extract_messages_from_generate_content_response(
            path=path, payload=_load_mapping_from_payload(payload=response.text)
        )
        if response_messages:
            await self._enforcer.enforce(direction="output", messages=request_messages + response_messages)

        return response

    async def send(self, request: Request, **kwargs: Any) -> Any:
        path = str(getattr(getattr(request, "url", None), "path", ""))
        request_messages = _extract_messages_from_generate_content_request(
            path=path,
            payload=_load_mapping_from_payload(payload=getattr(request, "content", None)),
        )
        if request_messages:
            await self._enforcer.enforce(direction="input", messages=request_messages)

        response = await self._inner.send(request, **kwargs)
        if kwargs.get("stream"):
            global _stream_warning_issued
            if not _stream_warning_issued:
                warn(
                    message="Anthale does not support real-time stream analysis. Gemini stream outputs are buffered and analyzed once the stream completes.",
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

            response_messages = _extract_messages_from_streaming_response(path=path, response=response)
            if response_messages:
                await self._enforcer.enforce(direction="output", messages=request_messages + response_messages)
            return response

        response_messages = _extract_messages_from_generate_content_response(
            path=path, payload=_load_mapping_from_payload(payload=response.text)
        )
        if response_messages:
            await self._enforcer.enforce(direction="output", messages=request_messages + response_messages)

        return response


def guard_gemini_client(
    gemini_client: _GeminiClient,
    *,
    policy_id: str,
    api_key: str | None = None,
    client: Any | None = None,
    async_client: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> _GeminiClient:
    """
    Guard an existing Google Gen AI client and return the same instance.

    Args:
        gemini_client (_GeminiClient): Existing `google.genai.Client` instance.
        policy_id (str): Anthale policy identifier.
        api_key (str | None): Anthale API key if Anthale clients are not explicitly provided.
        client (Any | None): Optional prebuilt sync Anthale client.
        async_client (Any | None): Optional prebuilt async Anthale client.
        metadata (Mapping[str, Any] | None): Optional metadata sent with each enforcement request.

    Returns:
        _GeminiClient: The same `gemini_client` instance, now instrumented with Anthale policy enforcement.

    Example:
    ```python
    from os import environ
    from google import genai
    from anthale.integrations.gemini import guard_gemini_client

    client = genai.Client(api_key=environ["GOOGLE_API_KEY"])
    client = guard_gemini_client(client, policy_id="<your-policy-identifier>", api_key=environ["ANTHALE_API_KEY"])
    ```
    """
    sync_enforcer, async_enforcer = build_enforcers(
        policy_id=policy_id,
        api_key=api_key,
        client=client,
        async_client=async_client,
        metadata=metadata,
    )

    api_client = getattr(gemini_client, "_api_client", None)
    if api_client is None:
        raise TypeError("Gemini client must expose an internal '_api_client'.")

    sync_http = getattr(api_client, "_httpx_client", None)
    async_http = getattr(api_client, "_async_httpx_client", None)
    if sync_http is None and async_http is None:
        raise TypeError("Gemini client internal API client must expose '_httpx_client' or '_async_httpx_client'.")

    if sync_http is not None and not getattr(sync_http, "_anthale_guarded", False):
        if sync_enforcer is None:
            raise ValueError("Gemini sync client requires a sync Anthale enforcer.")
        api_client._httpx_client = _AnthaleSyncHTTPClient(inner=sync_http, enforcer=sync_enforcer)  # type: ignore

    if async_http is not None and not getattr(async_http, "_anthale_guarded", False):
        if async_enforcer is None:
            raise ValueError("Gemini async client requires an async Anthale enforcer.")
        api_client._async_httpx_client = _AnthaleAsyncHTTPClient(inner=async_http, enforcer=async_enforcer)  # type: ignore

    return gemini_client


guard_client = guard_gemini_client
