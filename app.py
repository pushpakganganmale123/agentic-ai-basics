
import os
import gradio as gr
from fastapi import FastAPI

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

load_dotenv()

# ==================================================
# CONFIGURATION
# ==================================================

GROQ_MODEL = "openai/gpt-oss-120b"

TUTORIALS = {
    "1. LangChain Basics": "Agents, tools, prompts, messages and model selection.",
    "2. Semantic Search": "Embeddings, vector stores and similarity search.",
    "3. RAG Agent": "RAG chains, RAG agents and retrieval patterns.",
    "4. LangGraph RAG": "Graph states, nodes, edges and self-correcting RAG.",
    "5. Guardrails & HITL": "PII protection, validation and human approval.",
    "6. Deep Agents & Subagents": "Agent delegation, subagents and tool registries."
}


# ==================================================
# LANGGRAPH STATE
# ==================================================

class TutorState(TypedDict):
    topic: str
    question: str
    answer: str


# ==================================================
# AI RESPONSE GENERATION
# ==================================================

def generate_answer(state: TutorState):

    prompt = f"""
You are an expert Python, LangChain, LangGraph and Agentic AI tutor.

Topic: {state['topic']}
Description: {TUTORIALS[state['topic']]}

Student question:
{state['question']}

Instructions:
- Explain concepts for beginners.
- Use technically accurate terminology.
- Provide practical Python code.
- Explain code step by step.
- Mention common mistakes.
- Format using Markdown.
"""

    errors = []

    # ==================================================
    # 1. GROQ PRIMARY
    # ==================================================

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:

        try:
            print("[1/2] Trying Groq...", flush=True)

            client = Groq(api_key=groq_key)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Agentic AI tutor."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_completion_tokens=6000
            )

            result = response.choices[0].message.content

            if not result:
                raise ValueError("Groq returned an empty response.")

            print("[SUCCESS] Groq response generated.", flush=True)

            return {
                "answer": "**Provider: Groq**\n\n---\n\n" + result
            }

        except Exception as error:
            print(f"[GROQ ERROR] {error}", flush=True)
            errors.append(
                f"Groq: {type(error).__name__}: {error}"
            )

    else:
        errors.append("GROQ_API_KEY is missing.")

    # ==================================================
    # 2. GEMINI FALLBACK
    # ==================================================

    google_key = os.getenv("GOOGLE_API_KEY")

    if google_key:

        try:
            print("[2/2] Trying Gemini fallback...", flush=True)

            from google import genai

            client = genai.Client(api_key=google_key)

            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )

            result = response.output_text

            if not result:
                raise ValueError("Gemini returned an empty response.")

            print("[SUCCESS] Gemini response generated.", flush=True)

            return {
                "answer": "**Provider: Google Gemini**\n\n---\n\n" + result
            }

        except Exception as error:
            print(f"[GEMINI ERROR] {error}", flush=True)

            errors.append(
                f"Gemini: {type(error).__name__}: {error}"
            )

    else:
        errors.append("GOOGLE_API_KEY is missing.")

    return {
        "answer": (
            "### Both AI providers failed\n\n"
            + "\n\n".join(f"- {e}" for e in errors)
            + "\n\nCheck API keys, quotas and model availability."
        )
    }


# ==================================================
# LANGGRAPH WORKFLOW
# ==================================================

workflow = StateGraph(TutorState)

workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_answer")
workflow.add_edge("generate_answer", END)

tutor_graph = workflow.compile()


# ==================================================
# TUTOR FUNCTION
# ==================================================

def run_tutor(topic, question):

    if not question or not question.strip():
        return "Please enter a question."

    if topic not in TUTORIALS:
        return "Please select a valid tutorial."

    try:

        result = tutor_graph.invoke({
            "topic": topic,
            "question": question.strip(),
            "answer": ""
        })

        return result["answer"]

    except Exception as error:

        print(f"[APP ERROR] {error}", flush=True)

        return (
            "### Application error\n\n"
            f"`{type(error).__name__}: {error}`"
        )


# ==================================================
# GRADIO USER INTERFACE
# ==================================================

with gr.Blocks(
    title="Agentic AI Basics",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Agentic AI Basics

        ### Interactive LangChain, LangGraph and Agentic AI Tutor

        Ask questions about six practical tutorials.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            topic = gr.Dropdown(
                choices=list(TUTORIALS.keys()),
                value="1. LangChain Basics",
                label="Select tutorial"
            )

            question = gr.Textbox(
                label="Your question",
                placeholder="Explain LangGraph state with an example...",
                lines=5
            )

            submit = gr.Button(
                "Ask AI Tutor",
                variant="primary"
            )

            gr.Markdown(
                """
                **Powered by**

                - Groq
                - Google Gemini
                - LangGraph
                - Gradio
                """
            )

        with gr.Column(scale=2):

            answer = gr.Markdown(
                "Select a tutorial and ask your question."
            )

    submit.click(
        fn=run_tutor,
        inputs=[topic, question],
        outputs=answer
    )

    gr.Markdown(
        "---\nAgentic AI Basics | Educational Project"
    )


# ==================================================
# VERCEL PYTHON ENTRYPOINT
# ==================================================

# Vercel requires a top-level ASGI application.
app = FastAPI()

app = gr.mount_gradio_app(
    app,
    demo.queue(),
    path="/"
)


# ==================================================
# LOCAL DEVELOPMENT
# ==================================================

if __name__ == "__main__":

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860"))
    )
