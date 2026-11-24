# 🤖 agentic-ai-basics

**Learn LangChain, LangGraph & Agentic AI from scratch — one script at a time.**

Beginner-friendly Python tutorials covering agents, tools, RAG, vector stores, guardrails, and multi-agent orchestration using LangChain, LangGraph, and Google Gemini.

> **Who is this for?** Anyone starting out with agentic AI who learns best by running real code.

---

## 📂 Tutorial Map

Run them in order — each builds on concepts from the previous one.

| # | File | What You'll Learn |
|---|------|-------------------|
| 1A | `1_langchain_basics.py` | Agents, tools, messages, prompts, dynamic model selection |
| 1B | `1_models_test.py` | Smoke-test LLM providers (Gemini, Claude, Groq) |
| 2 | `2_tutorial_semantic_search_engine.py` | Embeddings, vector stores, similarity search |
| 3 | `3_tutorial_rag_agent.py` | RAG chain vs RAG agent (middleware vs tool-based) |
| 4 | `4_tutorial_rag_langgraph.py` | Full LangGraph RAG pipeline with self-correction |
| 5 | `5_tutorial_guardrail_HITL.py` | PII guardrails & human-in-the-loop approval |
| 6 | `6_deep_agents_tut_and_roles.py` | Multi-agent systems, subagents, JSON role configs |

### 🛠️ Utilities

| File | Purpose |
|------|---------|
| `test_agent_roles.json` | Sample subagent role definitions (used by Tutorial 6) |

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/agentic-ai-basics.git
cd agentic-ai-basics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API keys
cp .env.example .env
# Edit .env with your keys (see below)

# 4. Run any tutorial
python 1_langchain_basics.py
```

---

## 🔑 API Keys

Create a `.env` file in the project root with the keys you need:

```env
GOOGLE_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
HF_TOKEN=your-token-here            
```

> 💡 **Not all keys are needed!** Each script's docstring lists exactly which keys it requires. For some, **no API keys** at all — they run fully local.
>
> 📌 **Gemini note:** `GOOGLE_API_KEY` is automatically recognized by `ChatGoogleGenerativeAI` and LangChain's `init_chat_model` as the Gemini API key if written in the `.env` correctly as `GOOGLE_API_KEY` — same for OpenAI and Anthropoic

---

## 🗺️ Learning Path

```
Start here
  │
  ▼
┌─────────────────────────┐
│  1. LangChain Basics    │  ← agents, tools, prompts, messages
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  2. Semantic Search     │  ← embeddings, vector stores, retrieval
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  3. RAG Agent           │  ← chain vs agent retrieval patterns
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  4. LangGraph RAG       │  ← state graphs, self-correcting RAG
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  5. Guardrails & HITL   │  ← safety: PII filtering, human approval
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  6. Multi-Agent Systems  │  ← subagents, tool registries, roles
└─────────────────────────┘
```

---

## 📝 How to Use Each Script

Every script follows the same pattern:

1. **Read the docstring** at the top — it explains what the script covers and which API keys are needed.
2. **Run it** — `python <filename>.py`
3. **Experiment** — look for `# Uncomment to test:` lines to try alternative approaches.
4. **Each script is self-contained** — no need to run them in sequence (though the concepts build on each other).

---

## 🤝 Contributing

Found a bug or want to add a tutorial? PRs welcome! Keep the same style:
- Module docstring with description + required API keys
- Functions with docstrings (not narrating comments)
- `if __name__ == "__main__":` guard
- Section headers with `# ====` separators
