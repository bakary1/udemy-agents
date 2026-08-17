import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from IPython.display import Markdown, display

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-5.4-mini"

# Validate API key
if openai_api_key:
    print(f"OPENAI API key is valid and begins with {openai_api_key[:8]}")
else:
    print("Failed to fetch API key. Please check the configurations")

# Create an instance of OpenAI
openai = OpenAI()

# Extract text from pdf
reader = PdfReader("./twin/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# Read the summary file
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

display(Markdown(system_prompt))

# Structure messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Hi there, tell me about yourself."},
]

# Make API call
response = openai.chat.completions.create(model=model, messages=messages)
answer = response.choices[0].message.content


def chat(message, history):
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


chat("please summarize who you are", [])

# Test with gradio to build chat interface
gr.ChatInterface(chat).launch(inbrowser=True)


# Create a function that record emails - A tool
def record_email_tool(email):
    print(f"Tool called to record an email: {email}")
    with open("./twin/email.txt", "a", encoding="utf-8") as f:
        f.write(email + "\n")
        return "Email received"


# Step 1 - Write some JSON to describe the the tool
record_email_tool_json = {
    "name": "record_email_tool",
    "description": "Use this tool to record that a user provided their email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"}
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

# Create a list with my tools
tools = [{"type": "function", "function": record_email_tool_json}]


# Step 2 - Create a new chat function
def chat(message, history):
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(
        model="gpt-5.4-mini", messages=messages, tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_call = message.tool_calls[0]
        email = json.loads(tool_call.function.arguments).get("email")
        record_email_tool(email)
        messages.append(message)
        messages.append(
            {"role": "tool", "content": "Email recorded", "tool_call_id": tool_call.id}
        )
        response = openai.chat.completions.create(
            model="gpt-5.4-mini", messages=messages, tools=tools
        )

    return response.choices[0].message.content


# Test with gradio to build chat interface
gr.ChatInterface(chat).launch(inbrowser=True)


# Step 3 - Create a loop instead of just one tool call
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
        messages.append(message)
        for tool_call in message.tool_calls:
            email = json.loads(tool_call.function.arguments).get("email")
            record_email_tool(email)
            messages.append(
                {
                    "role": "tool",
                    "content": "Email recorded",
                    "tool_call_id": tool_call.id,
                }
            )
        response = openai.chat.completions.create(
            model="gpt-5.4-mini", messages=messages, tools=tools
        )

    return response.choices[0].message.content


# Test with gradio to build chat interface - now with llm in a loop
gr.ChatInterface(chat).launch(inbrowser=True)
