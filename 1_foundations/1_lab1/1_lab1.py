import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    sys.exit("Failed to load ANTHROPIC_API_KEY - check your .env file")

client = Anthropic(api_key=api_key)

messages = [{"role": "user", "content": "What is the capital of Sweden?"}]

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=messages,
)

print(response.content[0].text)


# Example 2
def get_text_response(message) -> str:
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


question = "Please propose a hard, challenging question to asccess someone's IQ. Respond only with the question itself. Do not provide a name or topic for the actual question."
messages = [{"role": "user", "content": question}]

response = client.messages.create(
    model="claude-sonnet-5", max_tokens=1024, messages=messages
)

question = get_text_response(response)

# Send the question back to the LLM
messages = [{"role": "user", "content": question}]

response = client.messages.create(
    model="claude-sonnet-5", max_tokens=1024, messages=messages
)

answer = get_text_response(response)
print(answer)


# Example
message = f"""
Here is a question:
{question}

And here is a possible answer that might be correct or incorrect:
{answer}

Please evaluate if the answer is correct or incorrect. Don't elaborate on how or why, just give your final judgement.
"""

messages = [{"role": "user", "content": message}]


response = client.messages.create(
    model="claude-sonnet-5", max_tokens=1024, messages=messages
)

llm_judge = get_text_response(response)
print(llm_judge)
