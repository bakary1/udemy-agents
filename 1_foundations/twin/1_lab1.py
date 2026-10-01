import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    sys.exit("Failed to load ANTHROPIC_API_KEY - check your .env file")

client = Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a funny short joke about Swedes."}],
)

print(response.content[0].text)
