import os
import asyncio
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, trace, function_tool

load_dotenv(override=True)

agent = Agent(
    name="Databricks tutor",
    instructions="You are a Databricks expert that answers questions clearly and pedagocially.",
    model="gpt-5.4-mini",
)


async def main() -> None:
    result = await Runner.run(
        agent,
        "What is the best way to learn databricks id I want to become proficient in data engineering as well as AI engineering?",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
