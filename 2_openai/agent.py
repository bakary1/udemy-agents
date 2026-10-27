import os
import asyncio
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, trace, function_tool

load_dotenv(override=True)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gpt-5.4-mini",
)


async def main() -> None:
    result = await Runner.run(
        agent,
        "Hi there, my name is?",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
