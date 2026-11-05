import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-5.4-mini")

# Simple llm call
message = "What is the capital of Sweden?"

reply = llm.invoke(message)

print(reply.content)

# Streaming
for chunk in llm.stream("Tell me a small poem about Stockholm"):
    print(chunk.content, end="", flush=True)

# Messages
messages = [
    SystemMessage(
        "You are an expert tour guide with specialized knowledge about Stockholm"
    ),
    HumanMessage(
        "What are some areas I should visit if I go to Stockholm for the first time?"
    ),
]

print(llm.invoke(messages).content)


# messages as dict - The exact same call using plain dicts instead
messages_as_dict = [
    {
        "role": "system",
        "content": "You are an expert tour guide with specialized knowledge about Stockholm",
    },
    {
        "role": "user",
        "content": "What are some areas I should visit if I go to Stockholm for the first time?",
    },
]

for chunk in llm.stream(messages_as_dict):
    print(chunk.content, end="", flush=True)


#############################################
# tool calling - create tools with decorator
#############################################


@tool
def get_weather(city: str) -> str:
    """Return the current weather in a given city"""
    cities = {"Paris": 25, "London": 31, "Marid": 36}
    return cities.get(city, 0.0)


@tool
def get_share_price(symbol: str) -> float:
    """Return the current share price for a given ticker symbol."""
    fake_prices = {"AAPL": 241.5, "AMZN": 198.0, "GOOG": 168.2}
    return fake_prices.get(symbol.upper(), 0.0)


print("name:", get_share_price.name)
print("description:", get_share_price.description)
print("args:", get_share_price.args)
print("called directly:", get_share_price.invoke({"symbol": "AAPL"}))  # test the tool

## Giving tools to the model - this is a bit of work by hand, but later when we use deep agents it will become easier
llm_with_tools = llm.bind_tools([get_share_price])
response = llm_with_tools.invoke("What is the share price of Amazon?")
print(f"content: {repr(response.content)}")
print(f"tool_calls: {response.tool_calls}")

# Add and test the weather tool
llm_with_tools = llm.bind_tools([get_share_price, get_weather])
response = llm_with_tools.invoke("What is the weather like in Paris?")
print(f"content: {repr(response.content)}")
print(f"tool_calls: {response.tool_calls}")

# Start the conversation and keep the model's tool request in the history
conversation = [HumanMessage("What is the weather like in Paris right now?")]
ai_message = llm_with_tools.invoke(conversation)
conversation.append(ai_message)

for call in ai_message.tool_calls:
    if call["name"] == "get_weather":
        result = get_weather.invoke(call["args"])
        conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

# call model again with all the context
final = llm_with_tools.invoke(conversation)
print(final.content)

#################################
### We must run the loop by hand
#################################

# Start the conversation and keep the model's tool request in the history
conversation = [HumanMessage("What is the share price of Amazon?")]
ai_message = llm_with_tools.invoke(conversation)
conversation.append(ai_message)

# Run each requested tool and add its result as a ToolMessage
for call in ai_message.tool_calls:
    if call["name"] == "get_share_price":
        result = get_share_price.invoke(call["args"])
        conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

# Invoke again, now that the model can see the tool result
final = llm_with_tools.invoke(conversation)
print(final.content)


# Structured output - we can require the model to respond with a Pydantic subclass
class Company(BaseModel):
    name: str = Field(description="The company name")
    ticker: str = Field(description="The stock ticker symbol")
    founded_year: int = Field(description="The year the company was founded")


structured_llm = llm.with_structured_output(Company)
company = structured_llm.invoke("Tell me about Amazon the technology company")
print(company)
print(f"Just the ticker: {company.ticker}")
