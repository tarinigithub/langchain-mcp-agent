
import streamlit as st
import asyncio
from langgraph_sdk import get_client

# Page configuration
st.set_page_config(page_title="Agent Chat Interface", page_icon="🤖")

# Initialize session state for chat history and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("Agent API URL", value="http://127.0.0.1:2024")
    
    if st.button("New Chat Session"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.success("Started a new session!")

    st.markdown("---")
    # Theme selection
    theme_preference = st.toggle("Dark Theme", value=False)
    
    st.markdown("### How to use")
    st.markdown("1. Ensure your LangGraph server is running (`uv run langgraph dev`).")
    st.markdown("2. Check the API URL matches your running server.")
    st.markdown("3. Start chatting!")

# Main chat interface
st.title("🤖 LangGraph Agent Chat")

# Initialize client
client = get_client(url=api_url)

async def create_thread():
    """Create a new thread if one doesn't exist."""
    if not st.session_state.thread_id:
        try:
            thread = await client.threads.create()
            st.session_state.thread_id = thread["thread_id"]
        except Exception as e:
            st.error(f"Failed to connect to agent: {e}")

async def process_message(user_input):
    """Send message to agent and get response."""
    if not st.session_state.thread_id:
        await create_thread()
        if not st.session_state.thread_id:
            return

    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response from the agent
            input_message = {"role": "user", "content": user_input}
            
            # Prepare input with required state fields
            input_state = {
                "messages": [input_message],
                "user_id": "streamlit_user",
                "preferences": {
                    "theme": "dark" if theme_preference else "light"
                }
            }
            
            # Use streaming run
            async for chunk in client.runs.stream(
                thread_id=st.session_state.thread_id,
                assistant_id="agent",
                input=input_state,
                stream_mode="values"
            ):
                if chunk.event == "values" and "messages" in chunk.data:
                    messages = chunk.data["messages"]
                    if messages:
                        last_message = messages[-1]
                        if last_message["type"] == "ai":
                             # In LangGraph, the AI response is the final content
                            full_response = last_message["content"]
                            message_placeholder.markdown(full_response)
            
            # Add assistant response to state if we got one
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                 # Fallback if streaming behavior is different than expected
                 st.warning("No response received. Check server logs.")

        except Exception as e:
            st.error(f"Error communicating with agent: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is your question?"):
    asyncio.run(process_message(prompt))
