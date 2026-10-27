import os

import requests
from agents import Agent, ModelSettings, function_tool
from dotenv import load_dotenv

from messenger import push


load_dotenv(override=True)
MODEL_NAME = os.getenv("MODEL_NAME")

INSTRUCTIONS = """
You are provided with a subject and a message. Use your tool to send a push \
notification, converting the message into a clean, well presented message \
with an appropriate subject line.
"""


@function_tool
def send_message_tool(subject: str, text_body: str) -> str:
    """
    Send out a push notification with a given subject and text body

    Args:
        subject: The subject of the push notification
        text_body: The text and actual message in the push notification
    """
    try:
        push(f"Subject: {subject}\n\n{text_body}")
    except requests.RequestException as e:
        print(f"Push notification failed: {e}")
        return "Failed to send push notification"
    return "Push notification sent successfully"


push_notification_agent = Agent(
    name="Push Notification Agent",
    model=MODEL_NAME,
    instructions=INSTRUCTIONS,
    tools=[send_message_tool],
    model_settings=ModelSettings(tool_choice="required"),
)
