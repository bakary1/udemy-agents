import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from pypdf import PdfReader
from pathlib import Path
from IPython.display import Markdown, display

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Failed to load ANTHROPIC_API_KEY - check your .env file")

client = Anthropic(api_key=api_key)

# Parse document
reader = PdfReader("../twin/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# Read summary
file_path = Path(__file__).resolve().parent.parent / "twin" / "summary.txt"
with open(file_path, "r", encoding="utf-8") as f:
    summary = f.read()

# Create a system prompt
system_prompt = f"""

# Your role

You are a digital twin running on a website, chatting with visitors of the website. You represent the person who's website you are on.
You answer questions related to to their career, background, skills and experience.

Here are the details of the person you are representing:

{summary}

# Context

Here is a summary of the person's LinkedIn profile so that you can answer questions:

{linkedin}

# Rules

Engage with the user. Be professional and engaging, as if talking to a potential client or future employer who came across the website.
Avoid answering questions that are not related to the user's career, background, skills and experience;
steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

IMPORTANT: If you don't know the answer, say so. Never make up an answer.
If the user asks about something not in the context, say that you don't know.
"""

display(Markdown(system_prompt))


# Create helper function
def get_text_response(message: str) -> str:
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


# Create the chat loop
def chat():
    messages = []

    print("Chat with you your assistant (type 'quit' to exit)")

    while True:
        user_input = input("\n")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        print(messages)

        response = client.messages.create(
            model="claude-sonnet-5",
            system=system_prompt,
            max_tokens=1024,
            messages=messages,
        )

        reply = get_text_response(response)
        messages.append({"role": "assistant", "content": reply})
        print(reply)


chat()
