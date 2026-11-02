import os
import requests
import asyncio
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, trace, function_tool, SQLiteSession

load_dotenv(override=True)

EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


# Define the actual Python function that will be called when Claude uses this tool


@function_tool
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
    file_path = Path(__file__).resolve().parent.parent / "email.txt"

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


# Create a simple agent
agent = Agent(
    name="Office assistant",
    model="gpt-5.4-mini",
    instructions="You are a helpful assistant that saves customers' email address.",
    tools=[save_email],
)

# Add observability with a trace
with trace("Recording email"):
    result = await Runner.run(
        agent, "Save my emailso that I can be contacted: test45@gmail.com"
    )
print(result.final_output)

# check details of LLM calls
result.to_input_list()


# # stream back results
# result = Runner.run_streamed(agent, input="Tell me 5 jokes about AI agents.")
# async for event in result.stream_events():
#     if event.type == "raw_response_event" and isinstance(
#         event.data, ResponseTextDeltaEvent
#     ):
#         print(event.data.delta, end="", flush=True)
