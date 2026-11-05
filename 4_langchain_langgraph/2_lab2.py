import os
import random
import requests
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from IPython.display import display, Image, Markdown
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)


# The State is the object that flows through the graph. We describe it with a TypeDict.
# Each field updates on its own, and by default a new value simpliy replaces the old one.

# Sometimes we want updates to accumulate instead. A reducer is a small function that
# says how to combine the old value with the new one. We attach it with Annotated.
# LangGraph ships the reducer we want for chat, add_messages, which appends new
# messages to the running list rather than overwriting it.

####################################
# Example - Five steps without LLM
####################################


# Step 1 - Create / define State object


# This is how we descibe a State to LangGraph: We tell it the state has
# one field called messages that is a list, and we let it know that add_messages is the reducer
class State(TypedDict):
    messages: Annotated[list, add_messages]


# The process is always the same in LangGraph: define the state, start a builder, add nodes, add edges, then compile!
# Here is a simple example:

nouns = ["Cabbages", "Unicorns", "Toasters", "Penguins", "Bananas"]
adjectives = ["Smelly", "outrageous", "pedantic", "sparkly"]


def silly_node(state: State) -> dict:
    sentence = f"{random.choice(nouns)} are {random.choice(adjectives)}"
    return {"messages": [{"role": "assistant", "content": sentence}]}


# Create the graph
builder = StateGraph(State)
builder.add_node("silly", silly_node)
builder.add_edge(START, "silly")
builder.add_edge("silly", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# Now we run the graph
result = graph.invoke({"messages": [{"role": "user", "content": "say something"}]})
print(result["messages"][-1].content)


########################
# Example 2
########################


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-5.4-mini", max_tokens=1024)


def chatbot(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph = (
    StateGraph(MessagesState)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Tell me a short poem about Stockholm"}]}
)
print(result["messages"][-1].content)


#######################################
# Example 3 - Adding tools
#######################################


# Define state
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-5.4-mini")

# A built-in tool that comes with langchain
search = GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper())


# Define push tool
@tool
def send_push_notification(text: str) -> str:
    """Send a short push notification to the user's phone."""
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )
    return "Notification sent"


tools = [search, send_push_notification]
llm_with_tools = llm.bind_tools(tools)


# Define the nodes
def chatbot_node(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph with nodes and edges

graph = (
    StateGraph(MessagesState)
    .add_node("chatbot", chatbot_node)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "chatbot")
    .add_conditional_edges("chatbot", tools_condition)
    .add_edge("tools", "chatbot")
    .compile()
)

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Use your search tool to tell me the ingredients in a banofee pie and send me a push notification.",
            }
        ]
    }
)
print(result["messages"][-1].content)

#############################
# Example 4
#############################


# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    spanish: str


# Define nodes


def chatbot_node(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def translator_node(state: State) -> dict:
    last = state["messages"][-1].content
    prompt = f"Translate this into spanish, reply with the translation only: \n\n{last}"
    return {"spanish": llm.invoke(prompt).content}


# Build graph
graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot_node)
    .add_node("tools", ToolNode(tools))
    .add_node("translator", translator_node)
    .add_edge(START, "chatbot")
    .add_conditional_edges(
        "chatbot", tools_condition, {"tools": "tools", END: "translator"}
    )
    .add_edge("tools", "chatbot")
    .add_edge("translator", END)
    .compile()
)

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "In one sentence, what is special about a banana?",
            }
        ]
    }
)
print(result["messages"][-1].content)
print(result["spanish"])
