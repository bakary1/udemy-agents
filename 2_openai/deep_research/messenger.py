import os

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


def push(message: str) -> None:
    """
    Send a push notification via Pushover.

    Raises:
        requests.RequestException: If the request fails or Pushover
            returns an error status.
    """
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    response = requests.post(pushover_url, data=payload, timeout=10)
    response.raise_for_status()


def main() -> None:
    try:
        push("Test message from messenger.py")
    except requests.RequestException as e:
        print(f"Push failed: {e}")


if __name__ == "__main__":
    main()
