
"""
Tutorial 6: Deep Agents, Subagents & Role-Based Tool Registry

Three approaches:
1. Single Deep Research Agent.
2. Main Agent + Research Subagent.
3. JSON-defined Subagents + Runtime Tool Registry.

Required environment variables:
- GROQ_API_KEY
- TAVILY_API_KEY
"""

import json
import os
import re
import time

from pathlib import Path
from typing import Literal, Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.tools import tool
from tavily import TavilyClient


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

MODEL_ID = "groq:openai/gpt-oss-120b"

BASE_DIR = Path(__file__).resolve().parent

ROLES_JSON_PATH = BASE_DIR / "test_agent_roles.json"

MAX_RETRIES = 5

BASE_RETRY_DELAY = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

if not TAVILY_API_KEY:
    raise EnvironmentError(
        "TAVILY_API_KEY is missing. Add it to your .env file."
    )

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)

TOOL_REGISTRY = {}


# ============================================================
# 2. QUICK INTERNET SEARCH
# ============================================================

@tool
def quick_internet_search(
    query: str,
    topic: Literal["news", "general", "finance"] = "general",
) -> str:
    """
    Search the internet for basic facts using Tavily.

    Args:
        query: Search query.
        topic: Search category.
    """

    try:
        response = tavily_client.search(
            query=query,
            max_results=2,
            topic=topic,
            search_depth="basic",
            include_raw_content=False,
        )

        results = response.get("results", [])

        if not results:
            return "No search results found."

        output = []

        for index, item in enumerate(results, start=1):

            output.append(
                f"Result {index}\n"
                f"Title: {item.get('title', 'N/A')}\n"
                f"Summary: {item.get('content', 'N/A')}\n"
                f"URL: {item.get('url', 'N/A')}"
            )

        return "\n\n".join(output)

    except Exception as error:
        return f"Quick internet search failed: {error}"


# ============================================================
# 3. DEEP INTERNET SEARCH
# ============================================================

@tool
def deep_search(
    query: str,
    topic: Literal["news", "general", "finance"] = "general",
) -> str:
    """
    Perform detailed internet research using Tavily.

    Args:
        query: Research topic.
        topic: Search category.
    """

    try:
        response = tavily_client.search(
            query=query,
            max_results=3,
            topic=topic,
            search_depth="advanced",
            include_raw_content=False,
        )

        results = response.get("results", [])

        if not results:
            return f"No detailed search results found for: {query}"

        findings = [
            f"Detailed research results for: {query}"
        ]

        for index, item in enumerate(results, start=1):

            findings.append(
                f"Source {index}\n"
                f"Title: {item.get('title', 'Untitled')}\n"
                f"Summary: {item.get('content', 'N/A')}\n"
                f"URL: {item.get('url', 'N/A')}"
            )

        return "\n\n".join(findings)

    except Exception as error:
        return f"Deep internet search failed: {error}"


# ============================================================
# 4. FINAL ANSWER EXTRACTION
# ============================================================

def extract_final_answer(result: Any) -> str:
    """
    Extract the latest non-empty AI response from agent state.
    """

    if isinstance(result, str):
        return result

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    for message in reversed(messages):

        if getattr(message, "type", None) != "ai":
            continue

        content = getattr(message, "content", "")

        if isinstance(content, str):

            if content.strip():
                return content

        elif isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))

                elif isinstance(item, str):
                    parts.append(item)

            answer = "\n".join(parts)

            if answer.strip():
                return answer

        elif content:
            return str(content)

    return "No final assistant response was returned."


# ============================================================
# 5. RATE-LIMIT RETRY HANDLER
# ============================================================

def run_agent(
    agent: Any,
    question: str,
    max_attempts: int = MAX_RETRIES,
) -> str:
    """
    Execute an agent with retry handling for Groq HTTP 429.

    Uses exponential backoff and Groq's suggested retry time
    when that information is available.
    """

    for attempt in range(1, max_attempts + 1):

        try:

            print(
                f"\nAgent execution: "
                f"attempt {attempt}/{max_attempts}"
            )

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                }
            )

            return extract_final_answer(result)

        except Exception as error:

            error_text = str(error)

            is_rate_limit = (
                isinstance(error, RateLimitError)
                or "429" in error_text
                or "rate_limit" in error_text.lower()
                or "rate limit" in error_text.lower()
            )

            if not is_rate_limit:
                raise

            if attempt == max_attempts:
                print(
                    "\nGroq rate limit persisted after "
                    "all retry attempts."
                )
                raise

            match = re.search(
                r"try again in\s+([0-9.]+)\s*s",
                error_text,
                re.IGNORECASE,
            )

            suggested_wait = (
                float(match.group(1))
                if match
                else 0
            )

            backoff_wait = (
                BASE_RETRY_DELAY * (2 ** (attempt - 1))
            )

            wait_time = max(
                suggested_wait + 2,
                backoff_wait,
            )

            print(
                "\nGroq rate limit reached."
                f"\nWaiting {wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Agent execution failed after all retry attempts."
    )


# ============================================================
# 6. APPROACH 1: SINGLE MAIN AGENT
# ============================================================

def run_main_agent():
    """
    Approach 1:
    One Deep Agent performs research using its own tools.
    """

    system_prompt = """
You are an internet research assistant.

Your available tool is quick_internet_search.

Research the user's question using the search tool.

Return:
1. A short explanation.
2. Important findings.
3. Practical examples when relevant.
4. Source URLs.

Rules:
- Do not invent facts or citations.
- Clearly distinguish verified facts from explanations.
- Do not attempt filesystem operations.
- Use only registered tools.
"""

    return create_deep_agent(
        model=MODEL_ID,

        tools=[
            quick_internet_search,
        ],

        system_prompt=system_prompt,

        name="single-research-agent",
    )


# ============================================================
# 7. APPROACH 2: MAIN AGENT + SUBAGENT
# ============================================================

def run_subagent_approach():
    """
    Approach 2:
    Main agent delegates specialized research to a subagent.
    """

    research_subagent = {
        "name": "research-subagent",

        "description": (
            "Researches internet topics and returns "
            "important findings with source URLs."
        ),

        "system_prompt": """
You are a specialized internet research subagent.

Use quick_internet_search to investigate the assigned topic.

Return:
- Important findings.
- Short explanations.
- Practical examples.
- Source URLs.

Do not invent facts or sources.
Do not attempt filesystem operations.
Use only your registered tools.
""",

        "tools": [
            quick_internet_search,
        ],

        "model": MODEL_ID,
    }

    return create_deep_agent(
        model=MODEL_ID,

        tools=[
            quick_internet_search,
        ],

        subagents=[
            research_subagent,
        ],

        system_prompt="""
You are the main research coordinator.

You have access to a specialized research-subagent.

Delegate research questions to research-subagent
when appropriate.

Review the returned findings and produce a concise
final answer with source URLs.

Do not invent facts or sources.
Use only available tools.
""",

        name="main-research-agent",
    )


# ============================================================
# 8. APPROACH 3: BUILD RUNTIME TOOL REGISTRY
# ============================================================

def build_tool_registry() -> dict:
    """
    Register Python tools using stable JSON-compatible names.

    JSON tool names do not have to match Python function names.
    They must match the keys in this registry.
    """

    TOOL_REGISTRY.clear()

    TOOL_REGISTRY.update(
        {
            "quick_search": quick_internet_search,
            "deep_analyze": deep_search,
        }
    )

    return TOOL_REGISTRY


# ============================================================
# 9. LOAD SUBAGENTS FROM JSON
# ============================================================

def load_subagents_from_json(
    roles_path: Path = ROLES_JSON_PATH,
    registry: dict | None = None,
) -> list[dict]:
    """
    Load JSON subagent definitions and resolve their tools.

    Uses utf-8-sig to support both normal UTF-8 and
    UTF-8 BOM-encoded JSON files.
    """

    if registry is None:
        registry = TOOL_REGISTRY

    if not roles_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {roles_path}"
        )

    # FIX 1: utf-8-sig handles the BOM.
    with roles_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        configs = json.load(file)

    if not isinstance(configs, list):
        raise ValueError(
            "JSON configuration must contain a list."
        )

    if not configs:
        raise ValueError(
            "No subagent configurations were found."
        )

    subagents = []

    registered_names = set()

    required_fields = {
        "name",
        "description",
        "system_prompt",
        "tools",
        "model",
    }

    for index, config in enumerate(configs, start=1):

        if not isinstance(config, dict):
            raise ValueError(
                f"Subagent {index} must be a JSON object."
            )

        missing_fields = required_fields - set(config.keys())

        if missing_fields:
            raise ValueError(
                f"Subagent {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        name = config["name"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Subagent {index} has an invalid name."
            )

        if name in registered_names:
            raise ValueError(
                f"Duplicate subagent name: {name}"
            )

        registered_names.add(name)

        if not isinstance(config["tools"], list):
            raise ValueError(
                f"Tools for {name} must be a JSON list."
            )

        resolved_tools = []

        for tool_name in config["tools"]:

            if tool_name not in registry:
                raise ValueError(
                    f"Tool '{tool_name}' for subagent "
                    f"'{name}' is not registered.\n"
                    f"Available tools: {list(registry.keys())}"
                )

            resolved_tools.append(
                registry[tool_name]
            )

        subagent = {
            "name": name,
            "description": config["description"],
            "system_prompt": config["system_prompt"],
            "tools": resolved_tools,
            "model": config["model"],
        }

        subagents.append(subagent)

        print(
            f"\nRegistered subagent: {name}"
            f"\nJSON tools: {config['tools']}"
            f"\nResolved tools: "
            f"{[tool.name for tool in resolved_tools]}"
            f"\nModel: {config['model']}"
        )

    return subagents


# ============================================================
# 10. APPROACH 3: RUN JSON REGISTRY AGENT
# ============================================================

def run_registry_agent(
    subagents: list[dict],
) -> str:
    """
    Create a master agent using JSON-defined subagents.
    """

    master_agent = create_deep_agent(
        model=MODEL_ID,

        tools=[
            quick_internet_search,
            deep_search,
        ],

        subagents=subagents,

        system_prompt="""
You are the master research coordinator.

You have access to specialized subagents configured
through an external JSON role registry.

For the user's question:

1. Delegate a short overview to quick-researcher.
2. Delegate detailed analysis to deep-researcher.
3. Compare their findings.
4. Combine the results into one structured answer.
5. Include relevant source URLs.

Use subagents when appropriate rather than doing
all research yourself.

Do not invent facts or citations.
""",

        name="master-research-agent",
    )

    question = (
        "What is LangGraph? Explain its main concepts, "
        "architecture, benefits, and practical applications."
    )

    return run_agent(
        master_agent,
        question,
    )


# ============================================================
# 11. RUN INDIVIDUAL APPROACH SAFELY
# ============================================================

def execute_approach(
    name: str,
    function,
) -> bool:
    """
    Execute one approach independently.

    A failure does not prevent the next approach.
    """

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    try:

        answer = function()

        print(f"\n{name} RESULT:\n")
        print(answer)

        print(f"\n{name}: SUCCESS")

        return True

    except Exception as error:

        print(f"\n{name}: FAILED")

        print(
            f"{type(error).__name__}: {error}"
        )

        return False


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("TUTORIAL 6: DEEP AGENTS, SUBAGENTS AND TOOL REGISTRY")
    print("=" * 65)

    results = {}

    # --------------------------------------------------------
    # APPROACH 1
    # --------------------------------------------------------

    def approach_1():

        agent = run_main_agent()

        return run_agent(
            agent,
            "What is LangGraph? Explain briefly with "
            "two practical use cases.",
        )

    results["Approach 1"] = execute_approach(
        "APPROACH 1: SINGLE MAIN AGENT",
        approach_1,
    )

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    print("\nWaiting before Approach 2...")
    time.sleep(15)

    # --------------------------------------------------------
    # APPROACH 2
    # --------------------------------------------------------

    def approach_2():

        agent = run_subagent_approach()

        return run_agent(
            agent,
            "Research the main benefits of LangGraph "
            "for building AI agents.",
        )

    results["Approach 2"] = execute_approach(
        "APPROACH 2: SUBAGENT DELEGATION",
        approach_2,
    )

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    print("\nWaiting before Approach 3...")
    time.sleep(15)

    # --------------------------------------------------------
    # APPROACH 3
    # --------------------------------------------------------

    def approach_3():

        registry = build_tool_registry()

        print("\nREGISTERED TOOLS:")
        print(list(registry.keys()))

        subagents = load_subagents_from_json(
            roles_path=ROLES_JSON_PATH,
            registry=registry,
        )

        print("\nLOADED SUBAGENTS:")
        print([
            subagent["name"]
            for subagent in subagents
        ])

        return run_registry_agent(subagents)

    results["Approach 3"] = execute_approach(
        "APPROACH 3: JSON ROLE REGISTRY",
        approach_3,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TUTORIAL 6 EXECUTION SUMMARY")
    print("=" * 65)

    for approach, success in results.items():

        status = "PASSED" if success else "FAILED"

        print(f"{approach}: {status}")

    passed = sum(results.values())

    print(
        f"\nTotal: {passed}/{len(results)} approaches passed."
    )

    print("\n" + "=" * 65)
    print("TUTORIAL 6 EXECUTION FINISHED")
    print("=" * 65)


if __name__ == "__main__":
    main()
