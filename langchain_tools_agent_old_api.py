# -*- coding: utf-8 -*-
# Langchain simple Agents comoponets - Tool Agent
    #  1. Models  
    #  2. Tools   - RAG Store, Python functions , APIs,data bases, files store (MCP) and etc.
    #  3. Prompt
    #  4. Agents Egnine or Executor
    #  5. memory

# STEP 0 - Import ENVIRONMENT VAIRABLES and Standard libraries
import os
import select
import yfinance as yf
import requests

from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
weatherAPIKey = str(os.getenv('weatherAPIKey'))

# STEP 1 - Import Langchain Libraries for Agent constuction 
from langchain.chat_models import init_chat_model

from langchain.agents import (
    AgentExecutor,
    create_react_agent, Tool
)
# Import Pydantic for data validation and settings management
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage 

from langchain import hub # for prompts template

from langchain.memory import ChatMessageHistory  #  for message history

from langchain_core.runnables.history import RunnableWithMessageHistory   #  for agent executor with memory

from langchain_community.tools import YouTubeSearchTool  #  youtube search tool


# STEP 2 - Choose the LLM - Langchain goole-gena
from langchain.chat_models import init_chat_model
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# STEP 2.1 - Test the lLM with simple prompt
prompt = "What is the current stock price of Apple and Tesla"
response=llm.invoke(prompt)
print(response.content)

# STEP 3 - Setting Custom tools Functions
youtube = YouTubeSearchTool(
   description="A tool to search YouTube videos. Use this tool if you think the user’s asked concept can be best explained by watching a video."
)
# Define the input schema for the get_stock_price tool
class StockPriceInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., 'AAPL' for Apple).")
    
# Function to Get Stock Price
def get_stock_price(ticker: str) -> dict:
    """Gets the stock price for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).

    Returns:
        dict: A dictionary containing the stock price or an error message.
    """
    print("Entered the method / function get_stock_price");
    print(ticker)
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if not hist.empty:
        return {"price": str(hist['Close'].iloc[-1])}
    else:
        return {"error": "No data available"}

# Define the input schema for the get_temperature tool
class TemperatureInput(BaseModel):
    city: str = Field(description="The name of the city (e.g., 'San Francisco').")

# Function to get temperature
def get_temperature(city: str) -> dict:
    """Gets the current temperature for a given city.

    Args:
        city (str): The name of the city (e.g., 'San Francisco').

    Returns:
        dict: A dictionary containing the temperature data or an error message.
    """
    print("Entered the method / function get_temperature");
    weatherAPIUrl = "http://api.weatherapi.com/v1/current.json?key=" + weatherAPIKey + "&q=" + city;
    print(weatherAPIUrl)
    response = requests.get(weatherAPIUrl)
    data = response.json()
    print(data)
    return data

# Define the input schema for the get_currency_exchange_rates tool
class CurrencyExchangeInput(BaseModel):
    currency: str = Field(description="The currency code (e.g., 'USD').")

# Function to get currency exchange rates
def get_currency_exchange_rates(currency: str) -> dict:
    """Gets the currency exchange rates for a given currency.

    Args:
        currency (str): The currency code (e.g., 'USD').

    Returns:
        dict: A dictionary containing the exchange rate data.
    """
    print("Entered the method / function get_currency_exchange_rates");
    # Where USD is the base currency you want to use
    url = 'https://v6.exchangerate-api.com/v6/6f9f5f76947ce2150d20b85c/latest/' + currency + "/"

    # Making our request
    response = requests.get(url)
    data = response.json()
    return data

# STEP 3.1 - Create Tools
tools=[
    Tool(
        name="get_stock_price",
        func=get_stock_price,
        description="useful for when you need to get the stock price of a company",
        args_schema=StockPriceInput
    ),
    Tool(
        name="get_temperature",
        func=get_temperature,
        description="useful for when you need to get the temperature of a city",
        args_schema=TemperatureInput
    ),
    Tool(
        name="get_currency_exchange_rates",
        func=get_currency_exchange_rates,
        description="useful for when you need to get the currency exchange rates for target currencly like INR, USD, EUR etc.",
        args_schema=CurrencyExchangeInput
    ),
    youtube,
    Tool(
        name="search",
        func=youtube.run,
        description="useful for when you need to answer questions about current events"
    )
]
# STEP 3.1 
llm_with_tools=llm.bind(tools=tools)

# query = "What the current exchange rate for Indian Ruppees INR"
query = "What is the current stock price of Apple and Tesla"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)

print(ai_msg.tool_calls)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = next((t for t in tools if t.name == tool_call['name']), None)
    if selected_tool:
        tool_response = selected_tool.invoke(tool_call['args'])
        messages.append(ToolMessage(tool_response, tool_call_id=tool_call['id']))
# print(messages)
ai_msg = llm_with_tools.invoke(messages)
messages.append(AIMessage(ai_msg.content))
print(messages)

print("\n\nFinal AI repose:",ai_msg.content)

# STEP 4 - create A prompt
prompt = hub.pull("hwchase17/react")

# STEP 5 - Create Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# STEP 6  - Run the Agent
# Create an agent executor from the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)
# Run the agent with a test query
response = agent_executor.invoke({"input": "How the stock market works?"})
print("response:", response)

# Print  the response from the agent
print("------------memory teset-----------------")
response = agent_executor.invoke({"input": "My name is Ram"})
print("response:", response)
response = agent_executor.invoke({"input": "what is my name?"})
print("response:", response)
print("------------memory teset-----------------")

### ---------------###

# STEP 7: Agent with memory
prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

memory = ChatMessageHistory(session_id="test-session")
# Wrap the agent with memory
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory, 
    input_messages_key="input",
    history_messages_key="chat_history",
)
config={"configurable": {"session_id": "test-session"}}
# Run the agent with a test query
response = agent_with_chat_history.invoke({"input": "Hi , My name is Venkat?"}, config=config)
# Print  the response from the agent
print("response:", response)
response = agent_with_chat_history.invoke({"input": "What is my name?"}, config=config)
# Print  the response from the agent
print("response:", response)