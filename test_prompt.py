
from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

# Mock
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
tools = []

class State(AgentState):
    preferences: dict

try:
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="Preferences: {preferences}",
        state_schema=State
    )
    print("Graph created successfully")
    # Try invoking
    # We need to mock the LLM response or just see if it crashes before LLM
    # But init_chat_model connects to real API if key present.
    # We just want to see if prompt formatting works.
    res = graph.invoke({"messages": [HumanMessage(content="hi")], "preferences": {"theme": "dark"}})
    print("Invoke success")
except Exception as e:
    print(f"Error: {e}")
