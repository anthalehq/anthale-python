from __future__ import annotations

import sys
import types
import warnings
import importlib
import importlib.machinery
from types import SimpleNamespace

import httpx
import pytest

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

    def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:  # noqa: ARG002
        request = httpx.Request(method, url)
        return httpx.Response(
            200,
            request=request,
            content=self.response_content,
            headers=self.headers,
            json=self.response_json,
        )

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

    def build_request(self, method: str, url: str, **kwargs: object) -> httpx.Request:
        return httpx.Request(method, url, **kwargs)


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

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:  # noqa: ARG002
        request = httpx.Request(method, url)
        return httpx.Response(
            200,
            request=request,
            content=self.response_content,
            headers=self.headers,
            json=self.response_json,
        )

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

    def build_request(self, method: str, url: str, **kwargs: object) -> httpx.Request:
        return httpx.Request(method, url, **kwargs)


class _GuardedSyncHTTPClient(_SyncHTTPClient):
    _anthale_guarded = True


class _GuardedAsyncHTTPClient(_AsyncHTTPClient):
    _anthale_guarded = True


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
    return module


def _install_fake_google_genai_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    google_mod = _new_module("google")
    genai_mod = _new_module("google.genai")
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


@pytest.fixture
def gemini_mod(monkeypatch: pytest.MonkeyPatch):
    _install_fake_google_genai_modules(monkeypatch)
    import anthale.integrations.gemini as mod

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


def test_guard_client_alias(gemini_mod: object) -> None:
    assert gemini_mod.guard_client is gemini_mod.guard_gemini_client


def test_guard_gemini_client_sync_enforces_generate_content(gemini_mod: object) -> None:
    anthale_sync = _SyncAnthaleClient(_policy_response("allow"))
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(
            _httpx_client=_SyncHTTPClient(
                response_json={
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "Hi there"}]}},
                    ]
                }
            ),
            _async_httpx_client=_AsyncHTTPClient(response_json={}),
        )
    )

    guarded = gemini_mod.guard_gemini_client(
        gemini_client,
        policy_id="pol_123",
        client=anthale_sync,
        async_client=anthale_async,
    )
    assert guarded is gemini_client

    _ = guarded._api_client._httpx_client.request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        content='{"systemInstruction":{"parts":[{"text":"rules"}]},"contents":[{"role":"user","parts":[{"text":"hello"}]}]}',
    )

    calls = anthale_sync.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


def test_guard_gemini_client_ignores_function_parts(gemini_mod: object) -> None:
    anthale_sync = _SyncAnthaleClient(_policy_response("allow"))
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(
            _httpx_client=_SyncHTTPClient(
                response_json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "ok"},
                                    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
                                ],
                            }
                        },
                    ]
                }
            ),
            _async_httpx_client=_AsyncHTTPClient(response_json={}),
        )
    )

    gemini_mod.guard_gemini_client(
        gemini_client,
        policy_id="pol_123",
        client=anthale_sync,
        async_client=anthale_async,
    )

    _ = gemini_client._api_client._httpx_client.request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        content='{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}',
    )

    output_messages = anthale_sync.policies.calls[1]["kwargs"]["messages"]
    contents = _content_of_messages(output_messages)
    assert any(content == "ok" for content in contents)
    assert not any("Paris" in content for content in contents)


def test_guard_gemini_client_sync_stream_warns_once(gemini_mod: object) -> None:
    gemini_mod._stream_warning_issued = False
    anthale_sync = _SyncAnthaleClient(_policy_response("allow"))
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(
            _httpx_client=_SyncHTTPClient(
                response_content=(
                    b'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n\n'
                    b'data: {"candidates":[{"content":{"parts":[{"text":" there"}]}}]}\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _async_httpx_client=_AsyncHTTPClient(response_json={}),
        )
    )

    gemini_mod.guard_gemini_client(
        gemini_client,
        policy_id="pol_123",
        client=anthale_sync,
        async_client=anthale_async,
    )

    req = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        content='{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}',
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response1 = gemini_client._api_client._httpx_client.send(req, stream=True)
        response2 = gemini_client._api_client._httpx_client.send(req, stream=True)

    assert isinstance(response1, httpx.Response)
    assert isinstance(response2, httpx.Response)
    assert len(caught) == 1
    assert len(anthale_sync.policies.calls) == 4


def test_guard_gemini_client_sync_stream_ignores_function_chunks(gemini_mod: object) -> None:
    gemini_mod._stream_warning_issued = False
    anthale_sync = _SyncAnthaleClient(_policy_response("allow"))
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(
            _httpx_client=_SyncHTTPClient(
                response_content=(
                    b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather"}}]}}]}\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _async_httpx_client=_AsyncHTTPClient(response_json={}),
        )
    )

    gemini_mod.guard_gemini_client(
        gemini_client,
        policy_id="pol_123",
        client=anthale_sync,
        async_client=anthale_async,
    )
    req = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        content='{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}',
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _ = gemini_client._api_client._httpx_client.send(req, stream=True)

    # No text in output chunks, so only input call happens.
    assert len(anthale_sync.policies.calls) == 1


@pytest.mark.asyncio
async def test_guard_gemini_client_async_enforces_generate_content(gemini_mod: object) -> None:
    anthale_sync = _SyncAnthaleClient(_policy_response("allow"))
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(
            _httpx_client=_SyncHTTPClient(response_json={}),
            _async_httpx_client=_AsyncHTTPClient(
                response_json={
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "Async hi"}]}},
                    ]
                }
            ),
        )
    )
    gemini_mod.guard_gemini_client(
        gemini_client,
        policy_id="pol_123",
        client=anthale_sync,
        async_client=anthale_async,
    )

    _ = await gemini_client._api_client._async_httpx_client.request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        content='{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}',
    )

    calls = anthale_async.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


def test_guard_gemini_client_returns_early_if_already_guarded(gemini_mod: object) -> None:
    anthale_sync = _SyncAnthaleClient(_policy_response("allow"))
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    sync_http = _GuardedSyncHTTPClient()
    async_http = _GuardedAsyncHTTPClient()
    gemini_client = SimpleNamespace(_api_client=SimpleNamespace(_httpx_client=sync_http, _async_httpx_client=async_http))

    guarded = gemini_mod.guard_gemini_client(
        gemini_client,
        policy_id="pol_123",
        client=anthale_sync,
        async_client=anthale_async,
    )
    assert guarded is gemini_client
    assert guarded._api_client._httpx_client is sync_http
    assert guarded._api_client._async_httpx_client is async_http


def test_guard_gemini_client_raises_for_missing_api_client(gemini_mod: object) -> None:
    with pytest.raises(TypeError, match="internal '_api_client'"):
        gemini_mod.guard_gemini_client(SimpleNamespace(), policy_id="pol_123", client=object(), async_client=object())


def test_guard_gemini_client_raises_for_missing_http_clients(gemini_mod: object) -> None:
    with pytest.raises(TypeError, match="'_httpx_client' or '_async_httpx_client'"):
        gemini_mod.guard_gemini_client(
            SimpleNamespace(_api_client=SimpleNamespace()),
            policy_id="pol_123",
            client=object(),
            async_client=object(),
        )


def test_guard_gemini_client_sync_requires_sync_enforcer(gemini_mod: object) -> None:
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(_httpx_client=_SyncHTTPClient(response_json={}), _async_httpx_client=_AsyncHTTPClient(response_json={}))
    )
    with pytest.raises(ValueError, match="Gemini sync client requires a sync Anthale enforcer"):
        gemini_mod.guard_gemini_client(
            gemini_client,
            policy_id="pol_123",
            async_client=_AsyncAnthaleClient(_policy_response()),
        )


def test_guard_gemini_client_async_requires_async_enforcer(gemini_mod: object) -> None:
    gemini_client = SimpleNamespace(
        _api_client=SimpleNamespace(_httpx_client=_SyncHTTPClient(response_json={}), _async_httpx_client=_AsyncHTTPClient(response_json={}))
    )
    with pytest.raises(ValueError, match="Gemini async client requires an async Anthale enforcer"):
        gemini_mod.guard_gemini_client(
            gemini_client,
            policy_id="pol_123",
            client=_SyncAnthaleClient(_policy_response()),
        )
from typing import Mapping
