import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Validate API key
if openai_api_key:
    print(f"OpenAI API Key is valid and begins with {openai_api_key[:8]}")
else:
    print("Invalid API key. Please check the configurations")

# Create an instance
openai = OpenAI()

# For pushover
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

# Validate user
if pushover_user:
    if pushover_user.startswith("u"):
        print("Pushover user is valid")
    else:
        print("Pushover user is found but does not start with u")
else:
    print("Pushover user not found, please check the configurations")

# Validate token
if pushover_token:
    if pushover_token.startswith("a"):
        print("Pushover token is valid")
    else:
        print("Pushover token is found but does not start with a")
else:
    print("Pushover token not found, please check the configurations")


# Create a push
def push(message: str):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)


# Test function
push("Hello my friend!")


# Define record user function
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return "OK"


# Define function to record question
def record_unknown_question(question):
    push(f"Recording question:{question}. I do not know the answer to the question")
    return "OK"


# Describe the user details function (tools)
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": "Any additional info about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

# Describe the unknown question function (tools)
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

# Store the tool descriptions in a list of dicts
tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


# This function can take a list of tool calls, and run them. This is the IF statement!!
def handle_tool_calls_with_manual_if(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)

        # THE BIG IF STATEMENT!!!

        if tool_name == "record_user_details":
            result = record_user_details(**arguments)
        elif tool_name == "record_unknown_question":
            result = record_unknown_question(**arguments)

        results.append(
            {
                "role": "user",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )
    return results


# Using Python built-in globals()

# This gives us a more elegant way that avoids the IF statement.


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else "No tool found"
        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )
    return results


# Extract text from pdf
reader = PdfReader("./twin/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# Read summary from text file
with open("./twin/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

# Create system prompt
system_prompt = f"""

# Your role

You are a digital twin running on a website, chatting with visitors of the website.
You represent the person whos's website you are on.
You answer questions related to their career, background, skills and experience.

Here are the details of the person you are representing:

{summary}

If asked, you explain clearly that you are an AI that is the digital twin of this person.

# Context

Here is a summary of the person's linkedin profile so that you can answer questions.

{linkedin}

# Rules

Engage with the user. Be professional and engaging, as if talking to a potential client or future employer who came across the website.
Avoid answering questions that are not related to the user's career, background, skills and experience;
steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

IMPORTANT: If you don't know the answer to a question, say so. Never make up an answer. If the users asks about something not in the context, say that you don't know.
"""


def chat(message, history):
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(
        model="gpt-5.4-mini", messages=messages, tools=tools
    )
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = openai.chat.completions.create(
            model="gpt-5.4-mini", messages=messages, tools=tools
        )
    return response.choices[0].message.content


gr.ChatInterface(chat).launch(inbrowser=True)
