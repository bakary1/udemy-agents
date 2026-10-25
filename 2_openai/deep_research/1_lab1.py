import os
import requests
import asyncio
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, trace, function_tool, SQLiteSession

load_dotenv(override=True)

# Create a simple agent
agent = Agent(
    name="jokester",
    instructions="You are a very funny joke teller",
    model="gpt-5.4-mini",
)


# Send a query to agent
result = await Runner.run(agent, "Tell a joke about Autonomus AI agents")
print(result.final_output)

# check details of LLM calls
result.to_input_list()

# Add observability with a trace
with trace("Telling a joke"):
    result = await Runner.run(agent, "Tell a joke about Swedes")
print(result.final_output)


# stream back results
result = Runner.run_streamed(agent, input="Tell me 5 jokes about AI agents.")
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(
        event.data, ResponseTextDeltaEvent
    ):
        print(event.data.delta, end="", flush=True)
