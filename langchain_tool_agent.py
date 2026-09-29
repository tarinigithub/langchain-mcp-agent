# -*- coding: utf-8 -*-
# Langchain simple Agents comoponets - Tool Agent
    #  1. Models  
    #  2. Tools   - RAG Store, Python functions , APIs,data bases, files store (MCP) and etc.
    #  3. Prompt
    #  4. Agents Egnine or Executor
    #  5. memory
"""
Latest LangChain Agent Implementation using create_agent (LangGraph-based)
Reference: https://docs.langchain.com/oss/python/langchain/agents
"""

# --- Step 0: import required libraries and API keys ---

import asyncio
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import SystemMessage, HumanMessage


# Load Environment Variables
load_dotenv()
if not os.getenv('GROQ_API_KEY'):
    raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file (see .env_example).")

# Folder of the weather-stock MCP server (provides getWeather and getStockPrice).
# Defaults to a weather-stock-mcp folder next to this project; override with MCP_SERVER_DIR.
# The server reads its own API keys from the .env file in that folder.
MCP_SERVER_DIR = os.getenv('MCP_SERVER_DIR', str(Path(__file__).resolve().parents[1] / 'weather-stock-mcp'))

# --- Step 1: Initialize LLM ---
llm = init_chat_model(model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b'), model_provider="groq")


# --- Step 2: Define Tools ---

# getWeather and getStockPrice come from the MCP server (loaded in Step 4).
# Each tool call starts the server over stdio via `uv run weather-stock-mcp`.
mcp_client = MultiServerMCPClient({
    "weather-stock": {
        "transport": "stdio",
        "command": "uv",
        "args": ["--directory", MCP_SERVER_DIR, "run", "weather-stock-mcp"],
    }
})

@tool
def get_currency_exchange_rates(currency: str) -> dict:
    """Gets the currency exchange rates for a given currency.
    
    Args:
        currency: The currency code (e.g., 'USD').
    """
    print(f"Debug: get_currency_exchange_rates called for {currency}")
    url = f'https://v6.exchangerate-api.com/v6/6f9f5f76947ce2150d20b85c/latest/{currency}/'
    try:
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        return {"error": str(e)}

# Initialize YouTube Tool
youtube = YouTubeSearchTool()

@tool
def search_youtube(query: str) -> str:
    """Search for videos on YouTube.
    
    Args:
        query: The search query for YouTube.
    """
    return youtube.run(query)


# Local tools; the MCP tools are added when the agent is built
local_tools = [get_currency_exchange_rates, search_youtube]

# STEP 3 .create agent prompt without memory and agent_scratchpad
agent_prompt="""
You are a helpful assistant. You can have a conversation and use tools.

User Preferences: {preferences}
"""


## PART III: Create Agent with Memory with custom state
class CustomAgentState(AgentState):
    user_id: str = "001"
    preferences: dict = {"theme": "light"}

# STEP 4. Build the agent with local tools + MCP tools (getWeather, getStockPrice).
# Loading MCP tools is async, so LangGraph Studio/API uses this async graph factory
# (see langgraph.json). The platform handles persistence, so no checkpointer here.
mcp_tools = None

async def make_graph():
    global mcp_tools
    if mcp_tools is None:  # LangGraph calls the factory per request; load the tool list once
        mcp_tools = await mcp_client.get_tools()
    return create_agent(
        model=llm,
        tools=local_tools + mcp_tools,
        system_prompt=agent_prompt,
        state_schema=CustomAgentState,
    )


# MCP tools are async-only, so the agent must be called with ainvoke/astream.
# Prints each tool call and result so you can see the MCP tools being used.
async def main(question):
    agent_graph = await make_graph()
    print(f"MCP tools loaded from server: {[t.name for t in mcp_tools]}")
    result = await agent_graph.ainvoke({
        "messages": [HumanMessage(content=question)],
        "preferences": {"theme": "light"},
    })
    mcp_tool_names = {t.name for t in mcp_tools}
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            source = "MCP" if call["name"] in mcp_tool_names else "local"
            print(f"-> [{source}] {call['name']}({call['args']})")
        if message.type == "tool":
            print(f"<- {message.name} returned: {message.text}")
    print(f"\nAnswer:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What's the weather in Tokyo and the stock price of IBM?"
    asyncio.run(main(question))
