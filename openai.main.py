from __future__ import annotations

from openai import OpenAI

from src.anthale.integrations.openai import guard_openai_client

client = OpenAI(api_key=environ["OPENAI_API_KEY"])
client = guard_openai_client(client, policy_id="<your-policy-identifier>", api_key=environ["ANTHALE_API_KEY"])

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

print("Agent ready. Type 'exit' or 'quit' to stop.\n")

while True:
    try:
        prompt = input("You: ").strip()

    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not prompt:
        continue

    if prompt.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    try:
        messages.append({"role": "user", "content": prompt})
        message = client.chat.completions.create(model="gpt-5-nano", messages=messages)
        message = message.choices[0].message.content.strip()

    except Exception as exc:
        print(f"Error: {exc}\n")
        continue

    print(f"Assistant: {message}\n", end="", flush=True)
    messages.append({"role": "assistant", "content": message})
