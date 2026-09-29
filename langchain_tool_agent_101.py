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

from calendar import c
import os
import yfinance as yf
import requests
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import SystemMessage, HumanMessage


# Load Environment Variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
weatherAPIKey = str(os.getenv('weatherAPIKey'))

# --- Step 1: Initialize LLM ---
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")


# --- Step 2: Define Tools ---

@tool
def get_stock_price(ticker: str) -> dict:
    """Gets the stock price for a given ticker symbol.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL' for Apple).
    """
    print(f"Debug: get_stock_price called for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return {"price": str(hist['Close'].iloc[-1])}
        else:
            return {"error": "No data available"}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_temperature(city: str) -> dict:
    """Gets the current temperature for a given city.
    
    Args:
        city: The name of the city (e.g., 'San Francisco').
    """
    print(f"Debug: get_temperature called for {city}")
    # Using OpenWeatherMap API
    weatherAPIUrl = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weatherAPIKey}&units=metric"
    try:
        response = requests.get(weatherAPIUrl)
        data = response.json()
        if "main" in data:
            return {
                "temperature": f"{data['main']['temp']}°C",
                "condition": data['weather'][0]['description']
            }
        else:
            return {"error": data.get("message", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}

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

# List of tools
tools = [get_stock_price, get_temperature, get_currency_exchange_rates, search_youtube]

# STEP 3 .create agent prompt without memory and agent_scratchpad
agent_prompt="""
You are a helpful assistant. You can have a conversation and use tools.
"""

# Step 4: create_agent  without memory and agent_scratchpad
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=agent_prompt
)

# STEP 5: Invoke the agent with a simple query
print("\n PART 1: Invoke the agent with a simple query")

result = agent.invoke(
    {"messages": [("user", "what is the current stock price of Apple?")]}
)
print(result["messages"][-1].content)

result = agent.invoke(
    {"messages": [("user", "hello, my name is Ram")]}
)
print(result["messages"][-1].content)

result = agent.invoke(
    {"messages": [("user", "what is my name")]}
)
print(result["messages"][-1].content)


# PART II: Create Agent with Memory
print("\n PART II: Create Agent with Memory")

# Step 4: create_agent  with memory
agent_with_memory = create_agent(
    model=llm,
    tools=tools,
    system_prompt=agent_prompt,
    checkpointer=MemorySaver(),
)

# STEP 5: Invoke the agent with a simple query
# When using a checkpointer, we MUST provide a thread_id to identify the session
config = {"configurable": {"thread_id": "session_1"}}

result = agent_with_memory.invoke(
    {"messages": [("user", "hello, my name is Ram")]},
    config=config
)
print(result["messages"][-1].content)

result = agent_with_memory.invoke(
    {"messages": [("user", "what is my name")]},
    config=config
)
print(result["messages"][-1].content)

## PART III: Create Agent with Memory with custom state
print("\n PART III: Create Agent with Memory with custom state")
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent_with_custom_state = create_agent(
    model=llm,
    tools=tools,
    system_prompt=agent_prompt,
    state_schema=CustomAgentState,
    checkpointer=MemorySaver(),
)

# Custom state can be passed in invoke
result = agent_with_custom_state.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})

print(result["messages"][-1].content)


# To access custom state in subsequent turns, we can inject it into the context
# or use a state_modifier if using create_react_agent.
# With create_agent, we can manually check state and inform the agent.
state = agent_with_custom_state.get_state({"configurable": {"thread_id": "1"}})
current_preferences = state.values.get("preferences", {})

result = agent_with_custom_state.invoke(
    {
        "messages": [
            SystemMessage(content=f"User Preferences: {current_preferences}"),
            HumanMessage(content="what is the current temperature in San Francisco?")
        ],
    },
    {"configurable": {"thread_id": "1"}})

print(result["messages"][-1].content)


