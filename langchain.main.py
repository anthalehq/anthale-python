"""Example chatbot using guard_chat_model (LCEL chain) with stream and async support."""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from src.anthale.integrations.core import AnthalePolicyViolationError
from src.anthale.integrations.langchain import guard_chat_model

# ---------------------------------------------------------------------------
# Build chain
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful weather assistant called WeatherBot. "
    "Always greet the user warmly and provide clear, concise information. "
    "If asked about anything unrelated to weather, politely redirect the conversation."
)


def build_chain():
    """Return an LCEL chain: prompt | guard_chat_model(ChatOpenAI)."""
    model = ChatOpenAI(model="gpt-5-nano", api_key=environ["OPENAI_API_KEY"])
    guarded_model = guard_chat_model(model, policy_id="<your-policy-identifier>", api_key=environ["ANTHALE_API_KEY"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    return prompt | guarded_model


# ---------------------------------------------------------------------------
# Sync chatbot (streaming)
# ---------------------------------------------------------------------------


def main_sync() -> None:
    print("=" * 60)
    print("  LangChain Chain + Anthale (sync stream)  (type 'quit' to exit)")
    print("=" * 60)
    print()

    chain = build_chain()
    history: list[HumanMessage | AIMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        try:
            print("Assistant: ", end="", flush=True)
            ai_response_parts: list[str] = []

            for chunk in chain.stream({"input": user_input, "history": history}):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                print(text, end="", flush=True)
                ai_response_parts.append(text)

            print("\n")
            ai_response = "".join(ai_response_parts)
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=ai_response))

        except AnthalePolicyViolationError as e:
            print(f"\n[BLOCKED by Anthale policy: {e.enforcement_identifier}]\n")

        except Exception as e:
            print(f"\n[Error: {e}]\n")


# ---------------------------------------------------------------------------
# Async chatbot (async stream)
# ---------------------------------------------------------------------------


async def main_async() -> None:
    print("=" * 60)
    print("  LangChain Chain + Anthale (async stream)  (type 'quit' to exit)")
    print("=" * 60)
    print()

    chain = build_chain()
    history: list[HumanMessage | AIMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        try:
            print("Assistant: ", end="", flush=True)
            ai_response_parts: list[str] = []

            async for chunk in chain.astream({"input": user_input, "history": history}):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                print(text, end="", flush=True)
                ai_response_parts.append(text)

            print("\n")
            ai_response = "".join(ai_response_parts)
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=ai_response))

        except AnthalePolicyViolationError as e:
            print(f"\n[BLOCKED by Anthale policy: {e.enforcement_identifier}]\n")

        except Exception as e:
            print(f"\n[Error: {e}]\n")


# ---------------------------------------------------------------------------
# Entry point – runs the async variant by default
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Switch to main_sync() to use synchronous streaming instead.
    asyncio.run(main_async())
