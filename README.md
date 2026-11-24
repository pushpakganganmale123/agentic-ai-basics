# 🤖 Agentic AI Basics

**Learn LangChain, LangGraph & Agentic AI from scratch — one script at a time.**

A beginner-friendly, hands-on Python learning repository covering LLMs, AI agents, tools, semantic search, Retrieval-Augmented Generation (RAG), LangGraph workflows, guardrails, Human-in-the-Loop (HITL), and multi-agent orchestration.

Built using **Python, LangChain, LangGraph, and Google Gemini**, this project provides practical tutorials with runnable code to understand how modern agentic AI applications work.

🌐 **Live Demo:** https://agentic-ai-basics123.vercel.app/  
💻 **GitHub Repository:** https://github.com/PushpakG123/agentic-ai-basics

---

## 🚀 Project Overview

Agentic AI Basics is a practical learning resource designed to help developers understand and implement agentic AI concepts through small, self-contained Python tutorials.

Instead of focusing only on theory, each tutorial demonstrates concepts through actual code, implementation examples, and practical workflows.

**What you'll explore:**
- Building LLM-powered applications with LangChain
- Creating AI agents and integrating external tools
- Understanding embeddings and semantic search
- Implementing RAG pipelines and intelligent retrieval
- Building stateful workflows using LangGraph
- Implementing AI safety guardrails and human approval
- Designing multi-agent systems with specialized subagents

## 📂 Tutorial Map

Follow the tutorials in order to progressively understand agentic AI architecture.

| # | Tutorial | What You'll Learn |
|---|---|---|
| 1A | `1_langchain_basics.py` | Agents, tools, messages, prompts, dynamic model selection |
| 1B | `1_models_test.py` | Testing LLM providers: Gemini, Claude, Groq |
| 2 | `2_tutorial_semantic_search_engine.py` | Embeddings, vector stores, similarity search |
| 3 | `3_tutorial_rag_agent.py` | RAG chains vs. RAG agents, middleware and tool-based retrieval |
| 4 | `4_tutorial_rag_langgraph.py` | LangGraph RAG pipeline, state management, self-correction |
| 5 | `5_tutorial_guardrail_HITL.py` | PII detection, guardrails, human-in-the-loop approval |
| 6 | `6_deep_agents_tut_and_roles.py` | Multi-agent systems, subagents, JSON role configurations |

### 🛠️ Utilities

| File | Purpose |
|---|---|
| `test_agent_roles.json` | Sample subagent role definitions for Tutorial 6 |

## 🧠 Key Concepts Covered

### 1. LangChain Fundamentals
Understand how LLM applications are built using prompts, messages, tools, and agents.

### 2. Semantic Search
Learn how text is converted into embeddings and how vector stores retrieve semantically similar content.

### 3. Retrieval-Augmented Generation (RAG)
Explore how an AI application retrieves relevant information before generating responses.

### 4. LangGraph Workflows
Build stateful AI workflows using nodes, edges, conditional routing, and self-correcting retrieval.

### 5. Guardrails & Human-in-the-Loop
Implement safety mechanisms such as PII detection, approval workflows, and controlled AI execution.

### 6. Multi-Agent Architecture
Understand how specialized agents and subagents collaborate, delegate tasks, and execute complex workflows.

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip
- API keys for the providers you want to use

### 1. Clone the repository

```bash
git clone https://github.com/PushpakG123/agentic-ai-basics.git
cd agentic-ai-basics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-google-api-key
GROQ_API_KEY=your-groq-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
TAVILY_API_KEY=your-tavily-api-key
HF_TOKEN=your-huggingface-token
```

Add only the API keys required by the tutorial you want to run.

### 4. Run a tutorial

```bash
python 1_langchain_basics.py
```

You can execute each tutorial independently.

## 🔑 API Key Configuration

| Provider | Environment Variable | Purpose |
|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | Gemini-powered LLM applications |
| Groq | `GROQ_API_KEY` | Groq-hosted language models |
| Anthropic | `ANTHROPIC_API_KEY` | Claude models |
| Tavily | `TAVILY_API_KEY` | Web search integrations |
| Hugging Face | `HF_TOKEN` | Hugging Face model access |

**Note:** Not every tutorial requires all API keys. Check the module docstring at the beginning of each Python file for its specific requirements.

Never commit your `.env` file or expose API keys in source code.

## 🗺️ Learning Path

```text
1. LangChain Basics
       |
       v
2. Semantic Search Engine
       |
       v
3. RAG Agent
       |
       v
4. LangGraph RAG
       |
       v
5. Guardrails & Human-in-the-Loop
       |
       v
6. Deep Agents & Multi-Agent Systems
```

Each stage introduces concepts that help you understand more advanced agentic AI architectures.

## 📝 How to Use the Tutorials

1. Open a tutorial and read its module docstring.
2. Check the required API keys and dependencies.
3. Run the Python script.
4. Examine the implementation and execution flow.
5. Experiment with the provided examples.
6. Uncomment optional examples to explore alternative implementations.

Each tutorial is self-contained, so you can run it independently.

## 🛠️ Technology Stack

- **Language:** Python
- **LLM Framework:** LangChain
- **Agent Workflow Framework:** LangGraph
- **LLM Providers:** Google Gemini, Groq, Anthropic
- **Embeddings & Retrieval:** Vector stores and semantic search
- **Search Integration:** Tavily
- **Configuration:** Environment variables and JSON role configurations

## 🎯 Project Goals

- Develop practical understanding of modern LLM frameworks.
- Learn how autonomous AI agents interact with tools.
- Understand the architecture behind RAG applications.
- Build stateful and self-correcting AI workflows.
- Explore AI safety through guardrails and human approval.
- Understand multi-agent collaboration and task delegation.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

If you find a bug or want to add a tutorial:

1. Fork the repository.
2. Create a new feature branch.
3. Follow the existing tutorial structure.
4. Include clear documentation and runnable examples.
5. Submit a pull request.

Keep tutorials self-contained, beginner-friendly, and focused on practical implementation.

## 👨‍💻 Author

**Pushpak Ganganmale**

- GitHub: https://github.com/PushpakG123
- Project Repository: https://github.com/PushpakG123/agentic-ai-basics
- Live Demo: https://agentic-ai-basics123.vercel.app/

---

⭐ If you find this project useful, consider giving the repository a star!

**Learn by building. Experiment with agents. Understand Agentic AI.**
