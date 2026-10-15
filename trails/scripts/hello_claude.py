from dotenv import load_dotenv
import anthropic

load_dotenv()  # loads ANTHROPIC_API_KEY from the .env file

client = anthropic.Anthropic()  # picks up the key automatically

message = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain what a Python virtual environment is in two sentences."}
    ],
)

print('>>>>>  '+ message.content[0].text)
print(message.usage)