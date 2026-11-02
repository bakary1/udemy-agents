import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool, SQLiteSession

load_dotenv(override=True)

agent = Agent(
    name="Helpful assistant",
    model="gpt-5.4-mini",
    instructions="You are a helpful assistant.",
)

response = await Runner.run(agent, "Hi there my name is Bakary.")
print(response.final_output)

response.to_input_list()

# Add memory manually
next_input = response.to_input_list() + [{"role": "user", "content": "What is my name"}]
next_input

response = await Runner.run(agent, next_input)
print(response.final_output)

# Add in-memory, for on-disk memory use SQLitesession("12345", "memory´.db") -> the memory is just a list of dictionaries that is stored in db, tracking the conversation history
session = SQLiteSession("123456")

response = await Runner.run(agent, "Hi there, my name is Bakary", session=session)
print(response.final_output)


response = await Runner.run(agent, "What is my name", session=session)
print(response.final_output)
