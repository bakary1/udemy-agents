import os
import json
from dotenv import load_dotenv
import re
from anthropic import Anthropic
import gradio as gr
import logging
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


# Create simple chat
def chat(message, history):
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-sonnet-5",
        system=system_prompt,
        max_tokens=1024,
        messages=messages,
    )
    return get_text_response(response)


gr.ChatInterface(chat).launch(inbrowser=True)


#############################
# Create Agent with tools
#############################

EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


# Define function / tool
def save_email(email: str) -> str:
    """Save a user's email to the contact list.

    Call this when the user has provided their email address and wants to be
    contacted.
    """
    email = email.strip()

    if not EMAIL_PATTERN.fullmatch(email):
        return (
            f" Error: {email!r} is not a valid email address. Ask the user to check it"
        )

    file_path = Path(__file__).resolve().parent.parent / "twin" / "email.txt"

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"{email}\n")
    except OSError:
        logging.exception("Failed to save email to %s", file_path)
        return "Error: could not save the email due to a server problem."

    return f"Successfully recorded email: {email}"


# Define the list of tools
tools = [
    {
        "name": "save_email",
        "description": "Save an email to the contact list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The user's email address",
                }
            },
            "required": ["email"],
        },
    }
]

# Map our functions and tools
tool_functions = {"save_email": save_email}


# Define conversation function
def run_conversation(message, history):
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-sonnet-5",
        system=system_prompt,
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                function = tool_functions[block.name]
                result = function(**block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-5",
            system=system_prompt,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

    return get_text_response(response)


# Test with gradio
gr.ChatInterface(run_conversation).launch(inbrowser=True)
