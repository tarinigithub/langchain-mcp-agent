# Langchain Agents

A LangChain agent (Groq LLM) that answers questions about weather, stock prices, currency exchange rates and YouTube videos. Weather and stock prices come from an **MCP server** ([tarinigithub/weather-stock-mcp](https://github.com/tarinigithub/weather-stock-mcp)); the other tools are local Python functions.

## Features

- **MCP Tools** (from the `weather-stock-mcp` server, loaded with `langchain-mcp-adapters`):
    - `getWeather`: Current weather for a city (WeatherAPI.com).
    - `getStockPrice`: Latest stock quote for a ticker symbol (Alpha Vantage).
- **Local Tools:**
    - `get_currency_exchange_rates`: Gets currency exchange rates using the ExchangeRate-API.
    - `search_youtube`: Searches for YouTube videos.
- **LLM Integration:** Uses Groq (`openai/gpt-oss-120b` by default).
- **LangGraph UI Integration:** Supports running the agent in a local web-based chat interface.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager
- The MCP server project, cloned next to this one, with its own `.env` containing `WEATHER_API_KEY` and `ALPHAVANTAGE_API_KEY` (see its README). If it lives elsewhere, set `MCP_SERVER_DIR` in `.env` to its path.

You do **not** need to start the MCP server yourself. It uses the stdio transport, so the agent launches it (`uv --directory <MCP_SERVER_DIR> run weather-stock-mcp`) whenever it needs the MCP tools.

## Setup

1. **Clone both repositories side by side:**
   ```bash
   git clone https://github.com/tarinigithub/langchain-mcp-agent.git
   git clone https://github.com/tarinigithub/weather-stock-mcp.git
   cd weather-stock-mcp && uv sync && cp .env.example .env && cd ..   # then add the weather/stock keys to weather-stock-mcp/.env
   cd langchain-mcp-agent
   ```

2. **Sync dependencies:**
   ```bash
   uv sync
   ```

3. **Set up your environment variables:**
   - Create a `.env` file in the root of the project (copy from `.env_example`).
   - Add your Groq key (weather and stock keys belong in the MCP server's `.env`, not here):
     ```
     GROQ_API_KEY="YOUR_GROQ_API_KEY"
     ```
   - Optional: `GROQ_MODEL`, `MCP_SERVER_DIR`, and the `LANGSMITH_*` settings for tracing.

## Usage

### Running via Command Line

Ask a question from the terminal:

```bash
uv run langchain_tool_agent.py "What's the weather in Paris and how is AAPL trading?"
```

With no question, it asks a default one ("What's the weather in Tokyo and the stock price of IBM?"):

```bash
uv run langchain_tool_agent.py
```

### Confirming the MCP tools are used

The command-line run prints the tools loaded from the MCP server, each tool call (tagged `[MCP]` or `[local]`), and what each tool returned, before the final answer. Example output:

```
MCP tools loaded from server: ['getWeather', 'getStockPrice']
-> [MCP] getWeather({'city': 'Paris', 'units': 'metric'})
<- getWeather returned: {
  "location": "Paris, France",
  "description": "Overcast",
  "temperature": 18.5,
  ...
}
-> [MCP] getStockPrice({'symbol': 'AAPL'})
<- getStockPrice returned: {
  "symbol": "AAPL",
  "price": 338.4,
  ...
}

Answer:
**Weather in Paris (metric)** ...
```

This shows that:

- The tool list comes from the running MCP server; this project doesn't define `getWeather` or `getStockPrice` itself.
- The model calls those MCP tools by name, marked `[MCP]`.
- The results have the fields the MCP server returns (`feels_like`, `humidity_percent`, `latest_trading_day`, ...).
- This project's `.env` has no weather or stock API keys, so that data can only come through the MCP server.

### Running with LangGraph UI

This project is configured to run with LangGraph Studio/UI for a rich interactive experience.

1. **Start the LangGraph Development Server:**

   ```bash
   uv run langgraph dev
   ```

   This command uses the configuration in `langgraph.json` to load the agent from the `make_graph` function in `langchain_tool_agent.py`.

2. **Access the UI:**

   Once the server starts, it will output a URL. Typically, you can access the UI at:

   [https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)

   (Note: The port might vary, check the terminal output. If it's 2025, change the port in the URL).

3. **Interact with the Agent:**
   - Use the chat interface to send messages to your agent.
   - Inspect the graph visualization.
   - View tool calls and outputs in real-time. Calls to `getWeather` and `getStockPrice` are the MCP tools.

## Project Structure

- `langchain_tool_agent.py`: Main agent implementation (Groq LLM, MCP client, local tools).
- `langgraph.json`: Configuration file for LangGraph CLI.
- `.env`: Environment variables (API keys).
