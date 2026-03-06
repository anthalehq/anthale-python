from __future__ import annotations

import sys
import types
import warnings
import importlib
import importlib.machinery
from types import SimpleNamespace

import httpx
import pytest

from anthale.integrations.core import AnthalePolicyViolationError
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


class _AsyncPolicies:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


class _SyncAnthaleClient:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.policies = _Policies(response)
        self.organizations = SimpleNamespace(policies=self.policies)


class _AsyncAnthaleClient:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.policies = _AsyncPolicies(response)
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
            async def _aboom() -> bytes:
                raise RuntimeError("aread failed")

            response.aread = _aboom  # type: ignore[method-assign]
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


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
    return module


def _install_fake_anthropic_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class Anthropic:
        pass

    class AsyncAnthropic:
        pass

    class APIConnectionError(Exception):
        pass

    module = _new_module("anthropic")
    module.Anthropic = Anthropic
    module.AsyncAnthropic = AsyncAnthropic
    module.APIConnectionError = APIConnectionError
    monkeypatch.setitem(sys.modules, "anthropic", module)


@pytest.fixture
def anthropic_mod(monkeypatch: pytest.MonkeyPatch):
    _install_fake_anthropic_module(monkeypatch)
    import anthale.integrations.anthropic as mod

    return importlib.reload(mod)


def _content_of_messages(messages: object) -> list[str]:
    values: list[str] = []
    for message in messages if isinstance(messages, list) else []:  # type: ignore
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        values.append(str(content))
    return values


def test_guard_client_alias(anthropic_mod: object) -> None:
    assert anthropic_mod.guard_client is anthropic_mod.guard_anthropic_client


def test_guard_anthropic_client_sync_enforces_messages(anthropic_mod: object) -> None:
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    anthropic_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_json={"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
        ),
        request=_sync_request_stub,
    )

    guarded = anthropic_mod.guard_anthropic_client(anthropic_client, policy_id="pol_123", client=anthale_client)
    assert guarded is anthropic_client

    req = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json={"system": "rules", "messages": [{"role": "user", "content": "hello"}]},
    )
    _ = guarded._client.send(req)

    calls = anthale_client.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


def test_guard_anthropic_client_ignores_tool_blocks(anthropic_mod: object) -> None:
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    anthropic_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_json={
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "name": "search", "input": {"q": "secret"}},
                ],
            }
        ),
        request=_sync_request_stub,
    )

    anthropic_mod.guard_anthropic_client(anthropic_client, policy_id="pol_123", client=anthale_client)
    req = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    _ = anthropic_client._client.send(req)

    output_messages = anthale_client.policies.calls[1]["kwargs"]["messages"]
    contents = _content_of_messages(output_messages)
    assert any(content == "ok" for content in contents)
    assert not any("secret" in content for content in contents)


def test_guard_anthropic_client_sync_stream_warns_once(anthropic_mod: object) -> None:
    anthropic_mod._stream_warning_issued = False
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    anthropic_client = SimpleNamespace(
        _client=_SyncHTTPClient(
            response_content=(
                b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n'
                b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        ),
        request=_sync_request_stub,
    )
    anthropic_mod.guard_anthropic_client(anthropic_client, policy_id="pol_123", client=anthale_client)

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages", json={"messages": [{"role": "user", "content": "hello"}]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response1 = anthropic_client._client.send(req, stream=True)
        response2 = anthropic_client._client.send(req, stream=True)

    assert isinstance(response1, httpx.Response)
    assert isinstance(response2, httpx.Response)
    assert len(caught) == 1
    assert len(anthale_client.policies.calls) == 4


@pytest.mark.asyncio
async def test_guard_anthropic_client_async_enforces_messages(anthropic_mod: object) -> None:
    anthale_client = _AsyncAnthaleClient(_policy_response("allow"))
    anthropic_client = SimpleNamespace(
        _client=_AsyncHTTPClient(response_json={"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}),
        request=_async_request_stub,
    )

    guarded = anthropic_mod.guard_anthropic_client(anthropic_client, policy_id="pol_123", async_client=anthale_client)
    assert guarded is anthropic_client

    req = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json={"messages": [{"role": "user", "content": "Hello async"}]},
    )
    _ = await guarded._client.send(req)

    calls = anthale_client.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


def test_guard_anthropic_client_returns_early_if_already_guarded(anthropic_mod: object) -> None:
    anthale_client = _SyncAnthaleClient(_policy_response("allow"))
    anthropic_client = SimpleNamespace(_client=_GuardedSyncHTTPClient(), request=_sync_request_stub)
    guarded = anthropic_mod.guard_anthropic_client(anthropic_client, policy_id="pol_123", client=anthale_client)
    assert guarded is anthropic_client
    assert guarded._client is anthropic_client._client


@pytest.mark.asyncio
async def test_guard_anthropic_client_async_returns_early_if_already_guarded(anthropic_mod: object) -> None:
    anthale_client = _AsyncAnthaleClient(_policy_response("allow"))
    anthropic_client = SimpleNamespace(_client=_GuardedAsyncHTTPClient(), request=_async_request_stub)
    guarded = anthropic_mod.guard_anthropic_client(anthropic_client, policy_id="pol_123", async_client=anthale_client)
    assert guarded is anthropic_client
    assert guarded._client is anthropic_client._client


def test_guard_anthropic_client_raises_for_missing_internal_client(anthropic_mod: object) -> None:
    with pytest.raises(TypeError, match="internal '_client'"):
        anthropic_mod.guard_anthropic_client(SimpleNamespace(request=_sync_request_stub), policy_id="pol_123", client=object())


def test_guard_anthropic_client_raises_for_missing_send_method(anthropic_mod: object) -> None:
    with pytest.raises(TypeError, match="'send' method"):
        anthropic_mod.guard_anthropic_client(
            SimpleNamespace(_client=_NoSendHTTPClient(), request=_sync_request_stub),
            policy_id="pol_123",
            client=object(),
        )


def test_guard_anthropic_client_sync_requires_sync_enforcer(anthropic_mod: object) -> None:
    anthropic_client = SimpleNamespace(_client=_SyncHTTPClient(response_json={}), request=_sync_request_stub)
    with pytest.raises(ValueError, match="Sync Anthropic client requires a sync Anthale enforcer"):
        anthropic_mod.guard_anthropic_client(
            anthropic_client,
            policy_id="pol_123",
            async_client=_AsyncAnthaleClient(_policy_response()),
        )


@pytest.mark.asyncio
async def test_guard_anthropic_client_async_requires_async_enforcer(anthropic_mod: object) -> None:
    anthropic_client = SimpleNamespace(_client=_AsyncHTTPClient(response_json={}), request=_async_request_stub)
    with pytest.raises(ValueError, match="Async Anthropic client requires an async Anthale enforcer"):
        anthropic_mod.guard_anthropic_client(
            anthropic_client,
            policy_id="pol_123",
            client=_SyncAnthaleClient(_policy_response()),
        )


def test_patched_sync_request_unwraps_policy_violation_signal(anthropic_mod: object) -> None:
    class _Client:
        def __init__(self) -> None:
            self._client = _SyncHTTPClient(response_json={})
            self.request = self._request

        def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
            raise anthropic_mod._AnthalePolicyViolationSignal(
                error=AnthalePolicyViolationError(enforcement_identifier="enf_inner")
            )

    client = _Client()
    anthropic_mod.guard_anthropic_client(client, policy_id="pol_123", client=_SyncAnthaleClient(_policy_response()))

    with pytest.raises(AnthalePolicyViolationError, match="enf_inner"):
        client.request("POST", "/v1/messages")


@pytest.mark.asyncio
async def test_patched_async_request_unwraps_policy_violation_signal(anthropic_mod: object) -> None:
    class _Client:
        def __init__(self) -> None:
            self._client = _AsyncHTTPClient(response_json={})
            self.request = self._request

        async def _request(self, *args: object, **kwargs: object) -> object:  # noqa: ARG002
            raise anthropic_mod._AnthalePolicyViolationSignal(
                error=AnthalePolicyViolationError(enforcement_identifier="enf_inner")
            )

    client = _Client()
    anthropic_mod.guard_anthropic_client(client, policy_id="pol_123", async_client=_AsyncAnthaleClient(_policy_response()))

    with pytest.raises(AnthalePolicyViolationError, match="enf_inner"):
        await client.request("POST", "/v1/messages")
from typing import Mapping
