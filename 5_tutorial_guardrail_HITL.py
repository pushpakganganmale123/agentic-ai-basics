
"""
Tutorial 5: Guardrails & Human-in-the-Loop (HITL)

Demonstrates:
1. PII protection using LangChain middleware.
2. Blocking sensitive API keys.
3. Human approval before sensitive tool execution.
4. Approval and rejection using LangGraph interrupts.

Required API keys:
    GOOGLE_API_KEY
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    PIIMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

MODEL = "google_genai:gemini-3.5-flash-lite"


# ============================================================
# SECTION 1: PII GUARDRAILS
# ============================================================

def run_pii_guardrail():
    """Test email redaction, card masking and API key blocking."""

    print("\n========== PII GUARDRAIL TEST ==========")

    agent = create_agent(
        model=MODEL,
        tools=[],
        system_prompt=(
            "You are a privacy-conscious assistant. "
            "Summarize the information provided by the user. "
            "Never invent or reveal sensitive personal information."
        ),
        middleware=[
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
            ),
            PIIMiddleware(
                "credit_card",
                strategy="mask",
                apply_to_input=True,
            ),
            PIIMiddleware(
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",
                strategy="block",
                apply_to_input=True,
            ),
        ],
    )

    # Test 1: Email redaction and card masking
    print("\n--- Test 1: Email and Credit Card ---")

    try:
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "My email is john.doe@example.com "
                        "and my card is 5105-1051-0510-5100. "
                        "Tell me what information you received."
                    ),
                }
            ]
        })

        print("Agent response:")
        print(result["messages"][-1].content)

    except Exception as error:
        print("PII test failed:", error)

    # Test 2: API key blocking
    print("\n--- Test 2: API Key Blocking ---")

    try:
        agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "My API key is "
                        "sk-12345678901234567890123456789012"
                    ),
                }
            ]
        })

        print("WARNING: API key was not blocked.")

    except Exception as error:
        print("API key successfully blocked.")
        print("Exception type:", type(error).__name__)


# ============================================================
# SECTION 2: DUMMY EMAIL TOOL
# ============================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a simulated email to a recipient.

    This is a dummy tool. It does not send real emails.
    """

    print("\n[EMAIL TOOL EXECUTED]")
    print("Recipient:", to)
    print("Subject:", subject)
    print("Body:", body)

    return f"Simulated email successfully sent to {to}"


# ============================================================
# SECTION 3: HUMAN-IN-THE-LOOP AGENT
# ============================================================

def create_hitl_agent():
    """Create an agent that requires approval before sending email."""

    return create_agent(
        model=MODEL,
        tools=[send_email],
        system_prompt=(
            "You are an assistant that can send emails using the "
            "send_email tool. When the user requests an email, "
            "you must call send_email with the supplied details. "
            "Never claim an email was sent without using the tool."
        ),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": True,
                }
            ),
        ],
        checkpointer=InMemorySaver(),
    )


# ============================================================
# SECTION 4: APPROVAL AND REJECTION TESTS
# ============================================================

def run_hitl_test(decision: str):
    """Test human approval or rejection of a sensitive tool call."""

    if decision not in ("approve", "reject"):
        raise ValueError("Decision must be approve or reject.")

    print(f"\n========== HITL TEST: {decision.upper()} ==========")

    agent = create_hitl_agent()

    # Each test receives an independent conversation thread.
    config = {
        "configurable": {
            "thread_id": f"tutorial5_{decision}"
        }
    }

    # Ask Gemini to invoke the email tool.
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Send an email to team@example.com. "
                        "Subject: Project Update. "
                        "Body: The project is progressing successfully."
                    ),
                }
            ]
        },
        config=config,
    )

    # Check whether execution was interrupted.
    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        print("ERROR: Expected human approval interrupt was not found.")
        print(result)
        return

    print("\nHuman approval required.")
    print("Pending operation:", interrupts)

    # Simulate a human decision.
    print(f"\nHuman decision: {decision.upper()}")

    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": decision
                    }
                ]
            }
        ),
        config=config,
    )

    print("\nExecution result:")
    print(result)


# ============================================================
# SECTION 5: MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("TUTORIAL 5: GUARDRAILS & HUMAN-IN-THE-LOOP")
    print("=" * 55)

    # PII protection
    run_pii_guardrail()

    # Human approval
    run_hitl_test("approve")

    # Human rejection
    run_hitl_test("reject")

    print("\n" + "=" * 55)
    print("TUTORIAL 5 EXECUTION COMPLETED")
    print("=" * 55)
