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


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-5.4-mini", max_tokens=2048)


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
    {
        "messages": [
            {"role": "user", "content": "What is a directed graph, in one sentence?"}
        ]
    }
)
print(result["messages"][-1].content)
############################################
# Notes explaining the code
############################################


# The State is the shared memory that flows through the graph.
# Every node reads from it and returns updates to it.
class MessagesState(TypedDict):
    # add_messages is a "reducer": when a node returns messages, they get
    # APPENDED to this list instead of replacing it.
    messages: Annotated[list, add_messages]


# The LLM client. Nodes call it; it is not part of the graph itself.
llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=1024)


# A node is just a function: it receives the current state and returns
# a dict with only the keys it wants to update. LangGraph merges that
# dict into the state using each key's reducer.
def chatbot(state: MessagesState) -> dict:
    # Send the whole conversation so far to the model.
    # The AI reply is wrapped in a list because add_messages expects a list.
    return {"messages": [llm.invoke(state["messages"])]}


# Build the graph. Each method returns the builder, so the calls chain.
graph = (
    # StateGraph(MessagesState) starts a graph that uses this State schema
    StateGraph(MessagesState)
    # Register "chatbot" as a node that runs chatbot()
    .add_node("chatbot", chatbot)
    # Edges define control flow: START -> chatbot -> END
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    # compile() checks the structure and returns a runnable graph
    # (tools like LangGraph Studio load the graph by this variable name)
    .compile()
)
