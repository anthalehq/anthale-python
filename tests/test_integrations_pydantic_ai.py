from __future__ import annotations

import sys
import types
import importlib
import importlib.machinery
from types import SimpleNamespace
from typing import Any, Mapping, AsyncIterator
from contextlib import asynccontextmanager

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


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
    return module


def _install_fake_pydantic_ai_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    pydantic_ai_mod = _new_module("pydantic_ai")
    messages_mod = _new_module("pydantic_ai.messages")
    models_mod = _new_module("pydantic_ai.models")
    wrapper_mod = _new_module("pydantic_ai.models.wrapper")

    class ModelMessage:
        pass

    class ModelRequest(ModelMessage):
        def __init__(self, parts: list[object]) -> None:
            self.parts = parts
            self.kind = "request"

    class ModelResponse(ModelMessage):
        def __init__(self, parts: list[object]) -> None:
            self.parts = parts
            self.kind = "response"

    class StreamedResponse:
        def __init__(self, events: list[object], response: object, *, fail_get: bool = False) -> None:
            self._events = events
            self._response = response
            self._fail_get = fail_get
            self.model_name = "fake-model"
            self.provider_name = "fake"
            self.provider_url = None
            self.timestamp = "now"

        def __aiter__(self) -> AsyncIterator[object]:
            async def _generator() -> AsyncIterator[object]:
                for event in self._events:
                    yield event

            return _generator()

        def get(self) -> object:
            if self._fail_get:
                raise RuntimeError("failed to build response")
            return self._response

        def usage(self) -> dict[str, int]:
            return {"input_tokens": 1, "output_tokens": 1}

    class ModelSettings:
        pass

    class ModelRequestParameters:
        pass

    class WrapperModel:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        async def request(self, messages: list[ModelMessage], model_settings: Any, model_request_parameters: Any) -> Any:
            return await self.wrapped.request(messages, model_settings, model_request_parameters)

        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: Any,
            model_request_parameters: Any,
            run_context: Any = None,
        ) -> AsyncIterator[Any]:
            async with self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context) as response_stream:
                yield response_stream

    messages_mod.ModelMessage = ModelMessage
    messages_mod.ModelRequest = ModelRequest
    messages_mod.ModelResponse = ModelResponse

    models_mod.ModelSettings = ModelSettings
    models_mod.ModelRequestParameters = ModelRequestParameters
    models_mod.StreamedResponse = StreamedResponse

    wrapper_mod.WrapperModel = WrapperModel

    monkeypatch.setitem(sys.modules, "pydantic_ai", pydantic_ai_mod)
    monkeypatch.setitem(sys.modules, "pydantic_ai.messages", messages_mod)
    monkeypatch.setitem(sys.modules, "pydantic_ai.models", models_mod)
    monkeypatch.setitem(sys.modules, "pydantic_ai.models.wrapper", wrapper_mod)


@pytest.fixture
def pai(monkeypatch: pytest.MonkeyPatch):
    _install_fake_pydantic_ai_modules(monkeypatch)
    import anthale.integrations.pydantic_ai as mod

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


class _WrappedModel:
    def __init__(self, *, fail_stream_get: bool = False) -> None:
        self.fail_stream_get = fail_stream_get
        self.stream_events = ["chunk-1", "chunk-2"]
        self.response = self._build_response()

    def _build_response(self) -> Any:
        from pydantic_ai.messages import ModelResponse

        return ModelResponse(
            parts=[
                SimpleNamespace(part_kind="text", content="assistant reply"),
                SimpleNamespace(part_kind="tool-call", content='{"secret":"x"}'),
            ]
        )

    async def request(self, messages: list[Any], model_settings: Any, model_request_parameters: Any) -> Any:  # noqa: ARG002
        return self.response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Any],  # noqa: ARG002
        model_settings: Any,  # noqa: ARG002
        model_request_parameters: Any,  # noqa: ARG002
        run_context: Any = None,  # noqa: ARG002
    ) -> AsyncIterator[Any]:
        from pydantic_ai.models import StreamedResponse

        yield StreamedResponse(
            events=self.stream_events,
            response=self.response,
            fail_get=self.fail_stream_get,
        )


def _request_messages() -> list[Any]:
    from pydantic_ai.messages import ModelRequest

    return [
        ModelRequest(
            parts=[
                SimpleNamespace(part_kind="system-prompt", content="rules"),
                SimpleNamespace(part_kind="user-prompt", content="hello"),
                SimpleNamespace(part_kind="tool-return", content='{"private":"data"}'),
            ]
        )
    ]


def test_guard_model_alias(pai: Any) -> None:
    assert pai.guard_model is pai.guard_pydantic_ai_model


def test_guard_pydantic_ai_model_returns_wrapper(pai: Any) -> None:
    wrapped = _WrappedModel()
    model = pai.guard_pydantic_ai_model(
        wrapped,
        policy_id="pol_123",
        async_client=_AsyncAnthaleClient(_policy_response()),
    )
    assert isinstance(model, pai.AnthalePydanticAIModel)


@pytest.mark.asyncio
async def test_request_enforces_input_and_output(pai: Any) -> None:
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    wrapped = _WrappedModel()
    model = pai.AnthalePydanticAIModel(wrapped, policy_id="pol_123", async_client=anthale_async)

    response = await model.request(_request_messages(), None, object())
    assert getattr(response, "kind", None) == "response"

    calls = anthale_async.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"

    input_messages = calls[0]["kwargs"]["messages"]
    input_contents = _content_of_messages(input_messages)
    assert "rules" in input_contents
    assert "hello" in input_contents
    assert not any("private" in content for content in input_contents)

    output_messages = calls[1]["kwargs"]["messages"]
    output_contents = _content_of_messages(output_messages)
    assert any(content == "assistant reply" for content in output_contents)
    assert not any("secret" in content for content in output_contents)


@pytest.mark.asyncio
async def test_request_stream_enforces_on_iteration_completion(pai: Any) -> None:
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    wrapped = _WrappedModel()
    model = pai.AnthalePydanticAIModel(wrapped, policy_id="pol_123", async_client=anthale_async)

    events: list[str] = []
    async with model.request_stream(_request_messages(), None, object()) as stream:
        async for event in stream:
            events.append(str(event))

    assert events == ["chunk-1", "chunk-2"]
    calls = anthale_async.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


@pytest.mark.asyncio
async def test_request_stream_enforces_on_context_exit_without_iteration(pai: Any) -> None:
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    wrapped = _WrappedModel()
    model = pai.AnthalePydanticAIModel(wrapped, policy_id="pol_123", async_client=anthale_async)

    async with model.request_stream(_request_messages(), None, object()):
        pass

    calls = anthale_async.policies.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["direction"] == "input"
    assert calls[1]["kwargs"]["direction"] == "output"


@pytest.mark.asyncio
async def test_request_stream_skips_output_enforcement_when_stream_get_fails(pai: Any) -> None:
    anthale_async = _AsyncAnthaleClient(_policy_response("allow"))
    wrapped = _WrappedModel(fail_stream_get=True)
    model = pai.AnthalePydanticAIModel(wrapped, policy_id="pol_123", async_client=anthale_async)

    async with model.request_stream(_request_messages(), None, object()) as stream:
        async for _ in stream:
            pass

    # Input check runs; output check is skipped because stream.get() raised.
    assert len(anthale_async.policies.calls) == 1


def test_model_requires_async_enforcer(pai: Any) -> None:
    with pytest.raises(ValueError, match="requires an async Anthale enforcer"):
        pai.AnthalePydanticAIModel(
            _WrappedModel(),
            policy_id="pol_123",
            client=_SyncAnthaleClient(_policy_response()),
        )
