from __future__ import annotations

import sys
import types
import inspect
import warnings
import importlib
import importlib.machinery
from typing import Any
from collections.abc import Iterator, AsyncIterator

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
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:
        self.calls.append((args, kwargs))
        return self.response


class _AsyncPolicies:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def enforce(self, *args: object, **kwargs: object) -> PolicyEnforceResponse:
        self.calls.append((args, kwargs))
        return self.response


class _Client:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.organizations = types.SimpleNamespace(policies=_Policies(response))


class _AsyncClient:
    def __init__(self, response: PolicyEnforceResponse) -> None:
        self.organizations = types.SimpleNamespace(policies=_AsyncPolicies(response))


def _message_content(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(getattr(value, "content", ""))


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
    return module


def _install_fake_langchain_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    class BaseMessage:
        def __init__(self, type_: str, content: Any) -> None:
            self.type = type_
            self.content = content

    class Runnable:
        pass

    class _Pipeline:
        def __init__(self, steps: list[object]) -> None:
            self.steps = steps

        def __or__(self, other: object) -> object:
            return _Pipeline([*self.steps, other])

        def invoke(self, value: Any) -> Any:
            current = value
            for step in self.steps:
                if hasattr(step, "invoke"):
                    current = step.invoke(current)  # type: ignore[attr-defined]
                elif callable(step):
                    current = step(current)
                else:
                    current = step
            return current

        async def ainvoke(self, value: Any) -> Any:
            current = value
            for step in self.steps:
                if hasattr(step, "ainvoke"):
                    current = await step.ainvoke(current)  # type: ignore[attr-defined]
                    continue

                if hasattr(step, "invoke"):
                    result = step.invoke(current)  # type: ignore[attr-defined]
                    current = await result if inspect.isawaitable(result) else result
                    continue

                if callable(step):
                    result = step(current)
                    current = await result if inspect.isawaitable(result) else result
                    continue

                current = step
            return current

        def stream(self, value: Any) -> Iterator[Any]:
            iterable: Iterator[Any] = iter([value])
            for step in self.steps:
                if hasattr(step, "transform"):
                    iterable = step.transform(iterable)  # type: ignore[attr-defined]
                    continue

                def _invoke_iter(source: Iterator[Any], target: object) -> Iterator[Any]:
                    for item in source:
                        if hasattr(target, "invoke"):
                            yield target.invoke(item)  # type: ignore[attr-defined]
                        elif callable(target):
                            yield target(item)
                        else:
                            yield target

                iterable = _invoke_iter(iterable, step)

            yield from iterable

        async def astream(self, value: Any) -> AsyncIterator[Any]:
            stream: AsyncIterator[Any] = _single_async(value)
            for step in self.steps:
                if hasattr(step, "atransform"):
                    stream = step.atransform(stream)  # type: ignore[attr-defined]
                    continue

                async def _invoke_aiter(source: AsyncIterator[Any], target: object) -> AsyncIterator[Any]:
                    async for item in source:
                        if hasattr(target, "ainvoke"):
                            result = target.ainvoke(item)  # type: ignore[attr-defined]
                            yield await result if inspect.isawaitable(result) else result
                        elif hasattr(target, "invoke"):
                            result = target.invoke(item)  # type: ignore[attr-defined]
                            yield await result if inspect.isawaitable(result) else result
                        elif callable(target):
                            result = target(item)
                            yield await result if inspect.isawaitable(result) else result
                        else:
                            yield target

                stream = _invoke_aiter(stream, step)

            async for chunk in stream:
                yield chunk

    class RunnableLambda(Runnable):
        @classmethod
        def __class_getitem__(cls, _item: object) -> type["RunnableLambda"]:
            return cls

        def __init__(self, func: Any, afunc: Any | None = None) -> None:
            self.func = func
            self.afunc = afunc

        def __or__(self, other: object) -> object:
            return _Pipeline([self, other])

        def invoke(self, value: Any) -> Any:
            return self.func(value)

        async def ainvoke(self, value: Any) -> Any:
            if self.afunc is not None:
                result = self.afunc(value)
                return await result if inspect.isawaitable(result) else result

            result = self.func(value)
            return await result if inspect.isawaitable(result) else result

        def stream(self, value: Any) -> Iterator[Any]:
            yield self.invoke(value)

        async def astream(self, value: Any) -> AsyncIterator[Any]:
            yield await self.ainvoke(value)

        def transform(self, input: Iterator[Any], config: Any = None, **kwargs: Any) -> Iterator[Any]:  # noqa: ARG002
            for chunk in input:
                yield self.invoke(chunk)

        async def atransform(
            self,
            input: AsyncIterator[Any],
            config: Any = None,  # noqa: ARG002
            **kwargs: Any,  # noqa: ARG002
        ) -> AsyncIterator[Any]:
            async for chunk in input:
                yield await self.ainvoke(chunk)

    class AgentMiddleware:
        pass

    class ModelRequest:
        def __init__(self, messages: list[object], system_message: object | None = None) -> None:
            self.messages = messages
            self.system_message = system_message

    class ModelResponse:
        def __init__(self, result: object) -> None:
            self.result = result

    class ExtendedModelResponse:
        def __init__(self, model_response: object) -> None:
            self.model_response = model_response

    langchain_mod = _new_module("langchain")
    langchain_agents_mod = _new_module("langchain.agents")
    langchain_agents_middleware_mod = _new_module("langchain.agents.middleware")
    langchain_agents_middleware_mod.AgentMiddleware = AgentMiddleware
    langchain_agents_middleware_mod.ModelRequest = ModelRequest
    langchain_agents_middleware_mod.ModelResponse = ModelResponse
    langchain_agents_middleware_mod.ExtendedModelResponse = ExtendedModelResponse

    langchain_core_mod = _new_module("langchain_core")
    langchain_core_messages_mod = _new_module("langchain_core.messages")
    langchain_core_messages_mod.BaseMessage = BaseMessage
    langchain_core_runnables_mod = _new_module("langchain_core.runnables")
    langchain_core_runnables_mod.Runnable = Runnable
    langchain_core_runnables_mod.RunnableLambda = RunnableLambda

    monkeypatch.setitem(sys.modules, "langchain", langchain_mod)
    monkeypatch.setitem(sys.modules, "langchain.agents", langchain_agents_mod)
    monkeypatch.setitem(sys.modules, "langchain.agents.middleware", langchain_agents_middleware_mod)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", langchain_core_messages_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", langchain_core_runnables_mod)


async def _single_async(value: Any) -> AsyncIterator[Any]:
    yield value


@pytest.fixture
def lc(monkeypatch: pytest.MonkeyPatch):
    _install_fake_langchain_modules(monkeypatch)
    import anthale.integrations.langchain as lc_mod

    return importlib.reload(lc_mod)


def test_middleware_wrap_model_call_sync(lc: Any) -> None:
    client = _Client(_policy_response())
    middleware = lc.AnthaleLangchainMiddleware(policy_id="pol_123", client=client)

    request = types.SimpleNamespace(messages=[{"role": "user", "content": "hello"}], system_message=None)
    response = middleware.wrap_model_call(request, lambda _req: {"role": "assistant", "content": "ok"})

    assert response == {"role": "assistant", "content": "ok"}
    calls = client.organizations.policies.calls
    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[1][1]["direction"] == "output"


@pytest.mark.asyncio
async def test_middleware_awrap_model_call_async(lc: Any) -> None:
    async_client = _AsyncClient(_policy_response())
    middleware = lc.AnthaleLangchainMiddleware(policy_id="pol_123", async_client=async_client)

    request = types.SimpleNamespace(messages=[{"role": "user", "content": "hello"}], system_message=None)

    async def _handler(_: object) -> dict[str, str]:
        return {"role": "assistant", "content": "ok"}

    response = await middleware.awrap_model_call(request, _handler)
    assert response == {"role": "assistant", "content": "ok"}

    calls = async_client.organizations.policies.calls
    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[1][1]["direction"] == "output"


def test_middleware_wrap_tool_call_sync(lc: Any) -> None:
    client = _Client(_policy_response())
    middleware = lc.AnthaleLangchainMiddleware(policy_id="pol_123", client=client)

    request = types.SimpleNamespace(tool_call={"name": "search", "args": {"content": "hello"}})
    result = middleware.wrap_tool_call(request, lambda _req: "tool-ok")

    assert result == "tool-ok"
    assert client.organizations.policies.calls == []


@pytest.mark.asyncio
async def test_middleware_awrap_tool_call_async(lc: Any) -> None:
    async_client = _AsyncClient(_policy_response())
    middleware = lc.AnthaleLangchainMiddleware(policy_id="pol_123", async_client=async_client)

    request = types.SimpleNamespace(tool_call={"name": "search", "args": {"content": "hello"}})

    async def _handler(_: object) -> str:
        return "tool-ok"

    result = await middleware.awrap_tool_call(request, _handler)

    assert result == "tool-ok"
    assert async_client.organizations.policies.calls == []


def test_middleware_wrap_tool_call_no_args_no_enforcement(lc: Any) -> None:
    client = _Client(_policy_response())
    middleware = lc.AnthaleLangchainMiddleware(policy_id="pol_123", client=client)

    request = types.SimpleNamespace(tool_call={"name": "search"})
    result = middleware.wrap_tool_call(request, lambda _req: "ok")

    assert result == "ok"
    assert client.organizations.policies.calls == []


def test_extract_messages_from_model_request_model_response_and_extended(lc: Any) -> None:
    from langchain.agents.middleware import ModelRequest, ModelResponse, ExtendedModelResponse

    request = ModelRequest(
        messages=[{"role": "user", "content": "hi"}, "fallback"],
        system_message={"role": "system", "content": "rules"},
    )
    response = ModelResponse(result=[{"role": "assistant", "content": "ok"}])
    extended = ExtendedModelResponse(model_response=response)

    request_messages = lc._extract_messages(value=request)
    response_messages = lc._extract_messages(value=extended)

    assert [_message_content(message) for message in request_messages] == ["rules", "hi", "fallback"]
    assert [_message_content(message) for message in response_messages] == ["ok"]


def test_extract_messages_drops_empty_payloads(lc: Any) -> None:
    messages = lc._extract_messages(value=[{"role": "assistant", "content": "   "}, None, "ok"])
    assert [_message_content(message) for message in messages] == ["ok"]


def test_guard_chat_model_pipeline_invoke(lc: Any) -> None:
    from langchain_core.runnables import RunnableLambda

    client = _Client(_policy_response())
    model = RunnableLambda(func=lambda x: f"model:{x}")
    guarded = lc.guard_chat_model(model, policy_id="pol_123", client=client)

    assert guarded.invoke("hello") == "model:hello"

    calls = client.organizations.policies.calls
    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[1][1]["direction"] == "output"


@pytest.mark.asyncio
async def test_guard_chat_model_pipeline_ainvoke(lc: Any) -> None:
    from langchain_core.runnables import RunnableLambda

    async_client = _AsyncClient(_policy_response())

    async def _model(value: str) -> str:
        return f"model:{value}"

    model = RunnableLambda(func=lambda x: x, afunc=_model)
    guarded = lc.guard_chat_model(model, policy_id="pol_123", async_client=async_client)

    assert await guarded.ainvoke("hello") == "model:hello"

    calls = async_client.organizations.policies.calls
    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[1][1]["direction"] == "output"


def test_guard_chat_model_stream(lc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(lc, "_stream_warning_issued", False)
    client = _Client(_policy_response())
    model = RunnableLambda(func=lambda x: f"stream:{x}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        guarded = lc.guard_chat_model(model, policy_id="pol_123", client=client)
        chunks = list(guarded.stream("hello"))

    assert chunks == ["stream:hello"]
    assert len(caught) == 1
    assert "real-time stream analysis" in str(caught[0].message)

    calls = client.organizations.policies.calls
    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[1][1]["direction"] == "output"


@pytest.mark.asyncio
async def test_guard_chat_model_astream(lc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(lc, "_stream_warning_issued", False)
    async_client = _AsyncClient(_policy_response())

    async def _model(value: str) -> str:
        return f"stream:{value}"

    model = RunnableLambda(func=lambda x: x, afunc=_model)
    guarded = lc.guard_chat_model(model, policy_id="pol_123", async_client=async_client)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunks = [chunk async for chunk in guarded.astream("hello")]

    assert chunks == ["stream:hello"]
    assert len(caught) == 1

    calls = async_client.organizations.policies.calls
    assert len(calls) == 2
    assert calls[0][1]["direction"] == "input"
    assert calls[1][1]["direction"] == "output"


def test_guardrail_output_runnable_transform_combines_chunks(lc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "_stream_warning_issued", False)
    enforced: list[Any] = []

    def capture(value: Any) -> Any:
        enforced.append(value)
        return value

    runnable = lc._GuardrailOutputRunnable(func=capture)  # type: ignore[attr-defined]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = list(runnable.transform(iter(["a", "b", "c"])))

    assert result == ["a", "b", "c"]
    assert enforced == ["abc"]


def test_guardrail_output_runnable_transform_falls_back_on_type_error(lc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "_stream_warning_issued", False)
    enforced: list[Any] = []

    def capture(value: Any) -> Any:
        enforced.append(value)
        return value

    runnable = lc._GuardrailOutputRunnable(func=capture)  # type: ignore[attr-defined]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _ = list(runnable.transform(iter([{"a": 1}, {"b": 2}])))

    assert enforced == [{"b": 2}]


@pytest.mark.asyncio
async def test_guardrail_output_runnable_atransform_combines_chunks(lc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "_stream_warning_issued", False)
    enforced: list[Any] = []

    async def acapture(value: Any) -> Any:
        enforced.append(value)
        return value

    runnable = lc._GuardrailOutputRunnable(func=lambda v: v, afunc=acapture)  # type: ignore[attr-defined]

    async def _chunks() -> AsyncIterator[str]:
        for chunk in ["x", "y"]:
            yield chunk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = [chunk async for chunk in runnable.atransform(_chunks())]

    assert result == ["x", "y"]
    assert enforced == ["xy"]


def test_guard_model_alias(lc: Any) -> None:
    assert lc.guard_model is lc.guard_chat_model
