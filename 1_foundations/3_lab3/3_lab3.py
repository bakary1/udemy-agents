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

# Initialize API key from .env file - required to authenticate with Anthropic API
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Failed to load ANTHROPIC_API_KEY - check your .env file")

# Create Anthropic client for making API calls
client = Anthropic(api_key=api_key)

# Extract text from LinkedIn PDF to use as context for the AI
reader = PdfReader("../twin/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text  # Concatenate all pages into one string

# Load the person's summary (biographical info) from a text file
file_path = Path(__file__).resolve().parent.parent / "twin" / "summary.txt"
with open(file_path, "r", encoding="utf-8") as f:
    summary = f.read()

# Create a system prompt that instructs Claude how to behave
# System prompts define the AI's role and behavior constraints
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

# Display the system prompt for reference
display(Markdown(system_prompt))


# Helper function to extract text from Claude's response
# Claude returns response.content as a list of content blocks with different types
def get_text_response(message: str) -> str:
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


# Simple chat function without tools (basic version)
def chat(message, history):
    # Reconstruct conversation history from Gradio's format
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    # Add the user's current message
    messages.append({"role": "user", "content": message})

    # Call Claude API with the full conversation history
    response = client.messages.create(
        model="claude-sonnet-5",
        system=system_prompt,  # Use system prompt to define behavior
        max_tokens=1024,       # Limit response length
        messages=messages,     # Include all conversation history
    )
    return get_text_response(response)


# Launch simple chat interface in browser
gr.ChatInterface(chat).launch(inbrowser=True)


#############################
# Advanced Agent with Tools
#############################
# Tools allow Claude to take actions (like saving emails) instead of just talking

# Regex pattern to validate email format
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


# Define the actual Python function that will be called when Claude uses this tool
def save_email(email: str) -> str:
    """Save a user's email to the contact list.

    Call this when the user has provided their email address and wants to be
    contacted.
    """
    email = email.strip()

    # Validate email format
    if not EMAIL_PATTERN.fullmatch(email):
        return (
            f" Error: {email!r} is not a valid email address. Ask the user to check it"
        )

    # Determine file path for storing emails
    file_path = Path(__file__).resolve().parent.parent / "twin" / "email.txt"

    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Append email to the contact list file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"{email}\n")
    except OSError:
        logging.exception("Failed to save email to %s", file_path)
        return "Error: could not save the email due to a server problem."

    return f"Successfully recorded email: {email}"


# Define the tool schema that tells Claude what tools are available and how to use them
# Claude reads this schema to decide whether to call the tool
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

# Map tool names to their Python implementations
# Used to execute the function when Claude calls the tool
tool_functions = {"save_email": save_email}


# Advanced conversation function that supports tool use
# This implements the "agentic loop": Claude can decide to call tools, we execute them,
# and feed results back to Claude in a loop until it generates a final response
def run_conversation(message, history):
    # Build conversation history
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    # Initial API call with tools available
    response = client.messages.create(
        model="claude-sonnet-5",
        system=system_prompt,
        max_tokens=1024,
        tools=tools,  # Provide available tools to Claude
        messages=messages,
    )

    # Agentic loop: keep going while Claude wants to use tools
    # When Claude decides to call a tool, response.stop_reason == "tool_use"
    while response.stop_reason == "tool_use":
        # Add Claude's response (which includes tool_use blocks) to conversation
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tools Claude requested
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Look up the Python function for this tool
                function = tool_functions[block.name]
                # Call it with the arguments Claude provided
                result = function(**block.input)
                # Package the result to send back to Claude
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,  # Links result to the specific tool call
                        "content": str(result),
                    }
                )

        # Add tool results back to conversation as user message
        messages.append({"role": "user", "content": tool_results})

        # Call Claude again with tool results - it may use more tools or generate final response
        response = client.messages.create(
            model="claude-sonnet-5",
            system=system_prompt,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

    # Loop exits when response.stop_reason != "tool_use" (usually "end_turn")
    return get_text_response(response)


# Launch the advanced chat interface with tool use support
gr.ChatInterface(run_conversation).launch(inbrowser=True)
