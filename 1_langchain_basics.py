"""
Tutorial 1-A: LangChain Basics

Covers core LangChain building blocks through self-contained examples:
  - Basic agent creation with a tool
  - Dynamic model selection middleware
  - Tool specifications (ToolRuntime, state access)
  - Static vs dynamic system prompts
  - Message types (System, Human, AI, Tool)
  - Tool definition and agent invocation

Required API keys: GROQ_API_KEY (Google Gemini)
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import tool as core_tool
from langchain_groq import ChatGroq


# ============================================================
# Section: Environment & Shared Config
# ============================================================

load_dotenv()
# ChatGroq auto-detects GROQ_API_KEY from env,
# but we load explicitly here for visibility.
groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# Section: Shared Tool
# ============================================================

@core_tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

TOOLS = [get_weather]


# ============================================================
# Section: Basic Agent
# ============================================================

def run_basic_agent() -> None:
    """Create a minimal agent with one tool and invoke it.

    Demonstrates the simplest LangChain agent setup:
    model + tool list -> create_agent -> invoke.
    """
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=groq_api_key,
    )

    agent = create_agent(model, TOOLS)

    # Uncomment to test: create agent directly from model string
    # agent = create_agent("openai/gpt-oss-120b", tools=TOOLS)

    # Uncomment to test: model-only invocation (no agent)
    # from langchain.chat_models import init_chat_model
    # model = init_chat_model("openai/gpt-oss-120b")
    # model.invoke("Why do parrots talk?")

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf."}]}
    )
    print(response["messages"][-1].content)


# ============================================================
# Section: Dynamic Model Selection
# ============================================================

def run_dynamic_model_agent() -> None:
    """Swap between a fast and advanced model at runtime based on conversation length.

    Uses `wrap_model_call` middleware to intercept every LLM call
    and choose which backend model to forward to.
    """
    basic_model = ChatGroq(
        model="openai/gpt-oss-120b", groq_api_key=groq_api_key
    )
    advanced_model = ChatGroq(
        model="openai/gpt-oss-120b", groq_api_key=groq_api_key
    )

    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        """Choose model based on conversation complexity."""
        message_count = len(request.state["messages"])
        if message_count > 10:
            request.model = advanced_model
        else:
            request.model = basic_model
        return handler(request)

    agent = create_agent(
        model=basic_model, tools=TOOLS, middleware=[dynamic_model_selection]
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what is the weather in sf. I think it might be rainy but I'm not sure, try to think you",
                }
            ]
        }
    )
    print(response["messages"][-1].content)


# ============================================================
# Section: System Prompts (Static & Dynamic)
# ============================================================

# Uncomment for static system prompt:
# agent = create_agent(model=model, tools=TOOLS, system_prompt="You are a helpful assistant, Be concise")


def test_system_prompt() -> None:
    """Dynamic system prompt that adapts based on a user_role context variable.

    Uses `dynamic_prompt` middleware and `context_schema` to inject
    a role-aware system message before every model call.
    """

    class Context(TypedDict):
        user_role: str

    @dynamic_prompt
    def user_role_prompt(request: ModelRequest) -> str:
        """Generate system prompt based on user role."""
        user_role = request.runtime.context.get("user_role", "user")
        base_prompt = "You are a helpful assistant."

        if user_role == "expert":
            return f"{base_prompt} Provide detailed technical responses."
        elif user_role == "beginner":
            return f"{base_prompt} Explain concepts simply and avoid jargon."
        return base_prompt

    agent = create_agent(
        model="openai/gpt-oss-120b",
        middleware=[user_role_prompt],
        context_schema=Context,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Explain machine learning"}]},
        context={"user_role": "expert"},
    )


# ============================================================
# Section: Custom Tool with Agent
# ============================================================


def test_tools() -> dict:
    """Define a custom search tool, wire it into an agent, and invoke.

    Returns:
        The full agent response dict.
    """

    @tool
    def search_database(query: str, limit: int = 10) -> str:
        """Search the internet for records matching the query.

        Args:
            query: Search terms to look for
            limit: Maximum number of results to return
        """
        return f"Found {limit} results for '{query}'"

    model = ChatGroq(
        model="openai/gpt-oss-120b", groq_api_key=groq_api_key
    )
    agent = create_agent(model, [search_database])
    human_msg = HumanMessage(
        "Find the most popular wireless headphones right now "
        "and check if they're in stock of Egypt and where they are"
    )
    response = agent.invoke({"messages": human_msg})
    return response


# ============================================================
# Section: Message Types (System, Human, AI, Tool)
# ============================================================

def test_messages_type() -> None:
    """Demonstrate the four core message types and how they compose a conversation.

    Shows how to build a multi-turn conversation with tool calls
    and tool results, then pass it to a model.
    """
    system_msg = SystemMessage(
        "You are a senior Python developer with expertise in web frameworks. "
        "Always provide code examples and explain your reasoning. "
        "Be concise but thorough in your explanations."
    )

    messages = [system_msg, HumanMessage("How do I create a REST API?")]

    # Uncomment to test: HumanMessage with optional fields
    # human_msg = HumanMessage(
    #     content="Hello!",
    #     name="alice",      # Optional: identify different users
    #     id="msg_123",      # Optional: unique identifier for tracing
    # )

    # Uncomment to test: multimodal message with image
    # human_message = HumanMessage(content_blocks=[
    #     {"type": "text", "text": "Hello, how are you?"},
    #     {"type": "image", "url": "https://example.com/image.jpg"},
    # ])

    # After a model makes a tool call
    ai_message = AIMessage(
        content=[],
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "San Francisco"},
                "id": "call_123",
            }
        ],
    )

    # Execute tool and create result message (tool_call_id must match)
    weather_result = "Sunny, 72\u00b0F"
    tool_message = ToolMessage(content=weather_result, tool_call_id="call_123")

    # Full conversation with tool-call round-trip
    messages = [
        HumanMessage("What's the weather in San Francisco?"),
        ai_message,
        tool_message,
    ]

    # Uncomment to invoke (requires a model instance in scope):
    # model = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
    # response = model.invoke(messages)

# ============================================================
# Section: Tool Specifications (ToolRuntime & State Access)
# ============================================================


@tool
def count_conversation_msgs(runtime: ToolRuntime) -> str:
    """Count the conversation message types so far.

    The ToolRuntime parameter gives the tool access to the full
    agent state, including the conversation history.
    """
    messages = runtime.state["messages"]
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")
    return (
        f"Conversation has {human_msgs} user messages, "
        f"{ai_msgs} AI responses, and {tool_msgs} tool results"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Uncomment any function below to test individually:
    # run_dynamic_model_agent()
    # test_system_prompt()
    # test_messages_type()
    # test_tools()

    run_basic_agent()

