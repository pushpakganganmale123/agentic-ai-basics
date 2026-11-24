
"""
Tutorial 1-B: Model Connection Testing

Smoke-test LLM providers through LangChain's unified
init_chat_model interface.

Required API keys:
    GROQ_API_KEY
    GOOGLE_API_KEY (optional)
    ANTHROPIC_API_KEY (optional)
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()


def test_connection(name: str, model_id: str, provider: str) -> bool:
    """
    Test connection to an LLM provider.

    Args:
        name: Human-readable model name.
        model_id: Provider-specific model identifier.
        provider: LangChain provider name.

    Returns:
        True if successful, otherwise False.
    """

    print(f"\nTesting {name}...", end=" ")

    try:
        llm = init_chat_model(
            model=model_id,
            model_provider=provider,
            temperature=0
        )

        response = llm.invoke(
            "Say 'Hello' and nothing else."
        )

        print(f"PASS: {response.content}")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


if __name__ == "__main__":

    print("=" * 55)
    print("       LLM PROVIDER CONNECTION TEST")
    print("=" * 55)

    results = []

    # =========================================
    # GROQ MODELS
    # =========================================

    if os.getenv("GROQ_API_KEY"):

        results.append(
            test_connection(
                "Qwen 3.8 (Groq)",
                "qwen/qwen3.8-27b",
                "groq"
            )
        )

        results.append(
            test_connection(
                "GPT OSS 120B (Groq)",
                "openai/gpt-oss-120b",
                "groq"
            )
        )

    else:
        print("\nSkipping Groq: GROQ_API_KEY not configured.")

    # =========================================
    # GOOGLE GEMINI
    # =========================================

    if os.getenv("GOOGLE_API_KEY"):

        results.append(
            test_connection(
                "Gemini 3.5 Flash Lite",
                "gemini-3.5-flash-lite",
                "google_genai"
            )
        )

    else:
        print("\nSkipping Gemini: GOOGLE_API_KEY not configured.")

    # =========================================
    # ANTHROPIC CLAUDE (OPTIONAL)
    # =========================================

    if os.getenv("ANTHROPIC_API_KEY"):

        results.append(
            test_connection(
                "Claude Opus",
                "claude-opus-4-6",
                "anthropic"
            )
        )

    else:
        print("\nSkipping Claude: ANTHROPIC_API_KEY not configured.")

    # =========================================
    # FINAL SUMMARY
    # =========================================

    passed = sum(results)
    total = len(results)
    failed = total - passed

    print("\n" + "=" * 55)
    print("             FINAL TEST SUMMARY")
    print("=" * 55)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if total > 0 and failed == 0:
        print("\nAll configured model tests passed!")

    elif failed > 0:
        print("\nSome model connections need attention.")

    print("=" * 55)
