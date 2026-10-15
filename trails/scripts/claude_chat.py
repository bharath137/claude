from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

history = []

print("Chat with Claude. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "quit":
        break

    history.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        system="You are a patient Claude tutor for a beginner.",
        messages=history,
    )

    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})

    print(f"\nClaude: {reply}\n")