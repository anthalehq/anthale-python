from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Mapping

import httpx
import pytest
from openai import APIConnectionError

import anthale.integrations.openai as openai_integration
from anthale.integrations.core import AnthalePolicyViolationError
from anthale.integrations.openai import guard_client, guard_openai_client
from anthale.types.organizations.policy_enforce_response import PolicyEnforceResponse


def _policy_response(action: str = "allow") -> PolicyEnforceResponse:
    if hasattr(PolicyEnforceResponse, "construct"):
        return PolicyEnforceResponse.construct(  # type: ignore[reportDeprecated]
            action=action,
            enforcer_identifier="enf",
        )
    return PolicyEnforceResponse.model_construct(action=action, enforcer_identifier="enf")


class _Policies:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


class _BlockingPolicies:
    def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:  # noqa: ARG002
        raise AnthalePolicyViolationError(enforcement_identifier="enf_block")


class _AsyncPolicies:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


class _BlockingAsyncPolicies:
    async def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:  # noqa: ARG002
        raise AnthalePolicyViolationError(enforcement_identifier="enf_block")


class _SyncAnthaleClient:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.policies = _Policies(response)
        self.organizations = SimpleNamespace(policies=self.policies)


class _AsyncAnthaleClient:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.policies = _AsyncPolicies(response)
        self.organizations = SimpleNamespace(policies=self.policies)


class _SyncAnthaleBlockingClient:
    def __init__(self) -> None:
        self.policies = _BlockingPolicies()
        self.organizations = SimpleNamespace(policies=self.policies)


class _AsyncAnthaleBlockingClient:
    def __init__(self) -> None:
        self.policies = _BlockingAsyncPolicies()
        self.organizations = SimpleNamespace(policies=self.policies)


class _SyncHTTPClient:
    def __init__(
        self,
        response_json: dict[str, object] | None = None,
        *,
        response_content: bytes | None = None,
        headers: dict[str, str] | None = None,
        fail_on_read: bool = False,
    ) -> None:
        self.response_json = response_json
        self.response_content = response_content
        self.headers = headers or {}
        self.fail_on_read = fail_on_read

    def send(self, request: httpx.Request, **kwargs: object) -> httpx.Response:  # noqa: ARG002
        response = httpx.Response(
            200,
            request=request,
            content=self.response_content,
            headers=self.headers,
            json=self.response_json,
        )

        if self.fail_on_read:
            def _boom() -> bytes:
                raise RuntimeError("read failed")

            response.read = _boom  # type: ignore[method-assign]

        return response


class _AsyncHTTPClient:
    def __init__(
        self,
        response_json: dict[str, object] | None = None,
        *,
        response_content: bytes | None = None,
        headers: dict[str, str] | None = None,
        fail_on_read: bool = False,
    ) -> None:
        self.response_json = response_json
        self.response_content = response_content
        self.headers = headers or {}
        self.fail_on_read = fail_on_read

    async def send(self, request: httpx.Request, **kwargs: object) -> httpx.Response:  # noqa: ARG002
        response = httpx.Response(
            200,
            request=request,
            content=self.response_content,
            headers=self.headers,
            json=self.response_json,
        )

        if self.fail_on_read:
            def _boom() -> bytes:
                raise RuntimeError("read failed")

            response.read = _boom  # type: ignore[method-assign]

        return response


class _GuardedSyncHTTPClient(_SyncHTTPClient):
    _anthale_guarded = True


class _GuardedAsyncHTTPClient(_AsyncHTTPClient):
    _anthale_guarded = True


class _NoSendHTTPClient:
    pass


def _sync_request_stub(*args: object, **kwargs: object) -> object:  # noqa: ARG001
    return object()


async def _async_request_stub(*args: object, **kwargs: object) -> object:  # noqa: ARG001
    return object()


def _content_of_message(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("content", ""))
    return str(getattr(value, "content", ""))


def test_guard_client_alias() -> None:
    assert guard_client is guard_openai_client


def test_guard_openai_client_sync_returns_same_client_and_enforces_for_responses() -> None:
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_json={
                "output_text": "Hello!",
                "output": [{"role": "assistant", "arguments": {"x": 1}}],
            }
        ),
        request=_sync_request_stub,
    )

    guarded = guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)
    assert guarded is openai_client

    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        json={"instructions": "be brief", "input": [{"role": "user", "content": "Hello"}]},
    )
    _ = guarded._client.send(req)

    calls = anthale_client.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


def test_guard_openai_client_sync_ignores_tool_call_arguments_for_chat_completions() -> None:
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "tool_calls": [
                                {"function": {"arguments": '{"city":"Paris"}'}},
                            ],
                        }
                    }
                ]
            }
        ),
        request=_sync_request_stub,
    )

    guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)

    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "weather?"}]},
    )
    _ = openai_client._client.send(req)

    calls = anthale_client.policies.calls
    assert len(calls) == 2
    output_messages = calls[1]["kwargs"]["messages"]
    assert any(_content_of_message(message) == "ok" for message in output_messages)
    assert not any(_content_of_message(message) == '{"city":"Paris"}' for message in output_messages)


def test_guard_openai_client_sync_stream_warns_once_and_returns_response_each_time() -> None:
    openai_integration._stream_warning_issued = False
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_content=(
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        ),
        request=_sync_request_stub,
    )
    guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)

    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response1 = openai_client._client.send(req, stream=True)
        response2 = openai_client._client.send(req, stream=True)

    assert isinstance(response1, httpx.Response)
    assert isinstance(response2, httpx.Response)
    assert len(caught) == 1
    assert "real-time stream analysis" in str(caught[0].message)

    calls = anthale_client.policies.calls
    # 2 calls per request: input + output
    assert len(calls) == 4
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"
    assert calls[2]["kwargs"]["direction"] == "input"
    assert calls[3]["kwargs"]["direction"] == "output"


def test_guard_openai_client_sync_stream_response_endpoint_uses_completed_payload() -> None:
    openai_integration._stream_warning_issued = False
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_content=(
                b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
                b'data: {"type":"response.completed","response":{"output_text":"final"}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        ),
        request=_sync_request_stub,
    )
    guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)

    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        json={"input": "hello"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _ = openai_client._client.send(req, stream=True)

    output_messages = anthale_client.policies.calls[1]["kwargs"]["messages"]
    assert any(_content_of_message(message) == "final" for message in output_messages)
    assert not any(_content_of_message(message) == "partial" for message in output_messages)


def test_guard_openai_client_sync_stream_returns_response_even_if_buffering_fails() -> None:
    openai_integration._stream_warning_issued = False
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_content=b"data: [DONE]\\n\\n",
            headers={"content-type": "text/event-stream"},
            fail_on_read=True,
        ),
        request=_sync_request_stub,
    )

    guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)
    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        response = openai_client._client.send(req, stream=True)
    assert isinstance(response, httpx.Response)
    assert len(anthale_client.policies.calls) == 1


@pytest.mark.asyncio
async def test_guard_openai_client_async_returns_same_client_and_enforces() -> None:
    anthale_async_client = _AsyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_AsyncHTTPClient(
            response_json={
                "output_text": "Hi async",
            }
        ),
        request=_async_request_stub,
    )

    guarded = guard_openai_client(openai_client, policy_id="pol_123", async_client=anthale_async_client)
    assert guarded is openai_client

    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        json={"input": [{"role": "user", "content": "Hello async"}]},
    )
    _ = await guarded._client.send(req)

    calls = anthale_async_client.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


@pytest.mark.asyncio
async def test_guard_openai_client_async_stream_warns_once_and_returns_response_each_time() -> None:
    openai_integration._stream_warning_issued = False
    anthale_async_client = _AsyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_AsyncHTTPClient(
            response_content=(
                b'data: {"choices":[{"delta":{"content":"Async"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" stream"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        ),
        request=_async_request_stub,
    )
    guard_openai_client(openai_client, policy_id="pol_123", async_client=anthale_async_client)

    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response1 = await openai_client._client.send(req, stream=True)
        response2 = await openai_client._client.send(req, stream=True)

    assert isinstance(response1, httpx.Response)
    assert isinstance(response2, httpx.Response)
    assert len(caught) == 1

    calls = anthale_async_client.policies.calls
    assert len(calls) == 4
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


def test_guard_openai_client_returns_early_if_already_guarded() -> None:
    client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(_client=_GuardedSyncHTTPClient(), request=_sync_request_stub)

    guarded = guard_openai_client(openai_client, policy_id="pol_123", client=client)
    assert guarded is openai_client
    assert guarded._client is openai_client._client


@pytest.mark.asyncio
async def test_guard_openai_client_async_returns_early_if_already_guarded() -> None:
    client = _AsyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(_client=_GuardedAsyncHTTPClient(), request=_async_request_stub)

    guarded = guard_openai_client(openai_client, policy_id="pol_123", async_client=client)
    assert guarded is openai_client
    assert guarded._client is openai_client._client


def test_guard_openai_client_raises_for_missing_internal_client() -> None:
    with pytest.raises(TypeError, match="internal '_client'"):
        guard_openai_client(SimpleNamespace(request=_sync_request_stub), policy_id="pol_123", client=object())


def test_guard_openai_client_raises_for_missing_send_method() -> None:
    with pytest.raises(TypeError, match="'send' method"):
        guard_openai_client(
            SimpleNamespace(_client=_NoSendHTTPClient(), request=_sync_request_stub),
            policy_id="pol_123",
            client=object(),
        )


def test_guard_openai_client_sync_requires_sync_enforcer() -> None:
    openai_client = SimpleNamespace(_client=_SyncHTTPClient(response_json={}), request=_sync_request_stub)

    with pytest.raises(ValueError, match="Sync OpenAI client requires a sync Anthale enforcer"):
        guard_openai_client(openai_client, policy_id="pol_123", async_client=_AsyncAnthaleClient(_policy_response()))


@pytest.mark.asyncio
async def test_guard_openai_client_async_requires_async_enforcer() -> None:
    openai_client = SimpleNamespace(_client=_AsyncHTTPClient(response_json={}), request=_async_request_stub)

    with pytest.raises(ValueError, match="Async OpenAI client requires an async Anthale enforcer"):
        guard_openai_client(openai_client, policy_id="pol_123", client=_SyncAnthaleClient(_policy_response()))


class _RequestWrapperSyncClient:
    def __init__(self) -> None:
        self._client = _SyncHTTPClient(response_json={})
        self.request = self._request

    def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
        raise openai_integration._AnthalePolicyViolationSignal(
            error=AnthalePolicyViolationError(enforcement_identifier="enf_inner")
        )


class _RequestWrapperAsyncClient:
    def __init__(self) -> None:
        self._client = _AsyncHTTPClient(response_json={})
        self.request = self._request

    async def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
        raise openai_integration._AnthalePolicyViolationSignal(
            error=AnthalePolicyViolationError(enforcement_identifier="enf_inner")
        )


def test_patched_sync_request_unwraps_policy_violation_signal() -> None:
    openai_client = _RequestWrapperSyncClient()
    guard_openai_client(openai_client, policy_id="pol_123", client=_SyncAnthaleClient(_policy_response()))

    with pytest.raises(AnthalePolicyViolationError, match="enf_inner"):
        openai_client.request("GET", "/x")


@pytest.mark.asyncio
async def test_patched_async_request_unwraps_policy_violation_signal() -> None:
    openai_client = _RequestWrapperAsyncClient()
    guard_openai_client(openai_client, policy_id="pol_123", async_client=_AsyncAnthaleClient(_policy_response()))

    with pytest.raises(AnthalePolicyViolationError, match="enf_inner"):
        await openai_client.request("GET", "/x")


class _ApiConnectionErrorSyncClient:
    def __init__(self) -> None:
        self._client = _SyncHTTPClient(response_json={})
        self.request = self._request

    def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
        violation = AnthalePolicyViolationError(enforcement_identifier="enf_cause")
        request = httpx.Request("GET", "https://api.openai.com/v1/responses")
        raise APIConnectionError(message="network", request=request) from violation


class _ApiConnectionErrorAsyncClient:
    def __init__(self) -> None:
        self._client = _AsyncHTTPClient(response_json={})
        self.request = self._request

    async def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
        violation = AnthalePolicyViolationError(enforcement_identifier="enf_cause")
        request = httpx.Request("GET", "https://api.openai.com/v1/responses")
        raise APIConnectionError(message="network", request=request) from violation


def test_patched_sync_request_unwraps_api_connection_cause() -> None:
    openai_client = _ApiConnectionErrorSyncClient()
    guard_openai_client(openai_client, policy_id="pol_123", client=_SyncAnthaleClient(_policy_response()))

    with pytest.raises(AnthalePolicyViolationError, match="enf_cause"):
        openai_client.request("GET", "/x")


@pytest.mark.asyncio
async def test_patched_async_request_unwraps_api_connection_cause() -> None:
    openai_client = _ApiConnectionErrorAsyncClient()
    guard_openai_client(openai_client, policy_id="pol_123", async_client=_AsyncAnthaleClient(_policy_response()))

    with pytest.raises(AnthalePolicyViolationError, match="enf_cause"):
        await openai_client.request("GET", "/x")


def test_sync_enforcer_violation_in_send_is_wrapped() -> None:
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(response_json={"output_text": "ignored"}),
        request=_sync_request_stub,
    )
    guard_openai_client(openai_client, policy_id="pol_123", client=_SyncAnthaleBlockingClient())

    req = httpx.Request("POST", "https://api.openai.com/v1/responses", json={"input": "hello"})
    with pytest.raises(openai_integration._AnthalePolicyViolationSignal):
        openai_client._client.send(req)


@pytest.mark.asyncio
async def test_async_enforcer_violation_in_send_is_wrapped() -> None:
    openai_client = SimpleNamespace(
        _client=_AsyncHTTPClient(response_json={"output_text": "ignored"}),
        request=_async_request_stub,
    )
    guard_openai_client(openai_client, policy_id="pol_123", async_client=_AsyncAnthaleBlockingClient())

    req = httpx.Request("POST", "https://api.openai.com/v1/responses", json={"input": "hello"})
    with pytest.raises(openai_integration._AnthalePolicyViolationSignal):
        await openai_client._client.send(req)


def test_non_supported_endpoint_only_enforces_input_when_messages_present() -> None:
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(response_json={"ok": True}),
        request=_sync_request_stub,
    )
    guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)

    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings", json={"input": "hello"})
    _ = openai_client._client.send(req)

    # embeddings endpoint is intentionally unsupported by extractor -> no enforcement calls
    assert anthale_client.policies.calls == []


@pytest.mark.asyncio
async def test_async_stream_returns_response_even_if_buffering_fails() -> None:
    openai_integration._stream_warning_issued = False
    anthale_client = _AsyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_AsyncHTTPClient(
            response_content=b"data: [DONE]\\n\\n",
            headers={"content-type": "text/event-stream"},
            fail_on_read=True,
        ),
        request=_async_request_stub,
    )

    guard_openai_client(openai_client, policy_id="pol_123", async_client=anthale_client)
    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        response = await openai_client._client.send(req, stream=True)
    assert isinstance(response, httpx.Response)
    assert len(anthale_client.policies.calls) == 1


def test_sync_stream_response_ignores_tool_call_arguments_from_delta() -> None:
    openai_integration._stream_warning_issued = False
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    openai_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_content=(
                b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\\"a\\":1"}}]}}]}\n\n'
                b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":",\\"b\\":2}"}}]}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        ),
        request=_sync_request_stub,
    )

    guard_openai_client(openai_client, policy_id="pol_123", client=anthale_client)
    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _ = openai_client._client.send(req, stream=True)

    # Only the input check runs because stream chunks contained tool-call arguments but no assistant text content.
    assert len(anthale_client.policies.calls) == 1


@pytest.mark.asyncio
async def test_async_request_patch_leaves_non_policy_api_connection_error_intact() -> None:
    class _Client:
        def __init__(self) -> None:
            self._client = _AsyncHTTPClient(response_json={})
            self.request = self._request

        async def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
            request = httpx.Request("GET", "https://api.openai.com/v1/responses")
            raise APIConnectionError(message="network", request=request)

    openai_client = _Client()
    guard_openai_client(openai_client, policy_id="pol_123", async_client=_AsyncAnthaleClient(_policy_response()))

    with pytest.raises(APIConnectionError, match="network"):
        await openai_client.request("GET", "/x")


def test_sync_request_patch_leaves_non_policy_api_connection_error_intact() -> None:
    class _Client:
        def __init__(self) -> None:
            self._client = _SyncHTTPClient(response_json={})
            self.request = self._request

        def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
            request = httpx.Request("GET", "https://api.openai.com/v1/responses")
            raise APIConnectionError(message="network", request=request)

    openai_client = _Client()
    guard_openai_client(openai_client, policy_id="pol_123", client=_SyncAnthaleClient(_policy_response()))

    with pytest.raises(APIConnectionError, match="network"):
        openai_client.request("GET", "/x")
