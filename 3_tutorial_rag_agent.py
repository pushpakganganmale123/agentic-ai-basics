"""
Tutorial 3: RAG Chain vs RAG Agent

Two retrieval-augmented generation patterns using the same vector store:

1. **RAG Chain** ("Search First, Then Ask Model"):
   Uses `dynamic_prompt` middleware to inject retrieved context into the
   system message *before* the model sees the query. No tools given.

2. **RAG Agent** ("Give the Model a Retrieval Tool"):
   The model decides *when* and *how often* to call the retriever tool,
   enabling multi-hop reasoning.

Required API keys: GOOGLE_API_KEY (Gemini -- for the LLM; embeddings are local HuggingFace)

Reference: https://docs.langchain.com/oss/python/langchain/rag
"""

import bs4
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_ID = "google_genai:gemini-3.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BLOG_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# ============================================================
# Section: Initialisation (Model, Embeddings, Vector Store)
# ============================================================

def init_components():
    """Create the LLM, embedding model, and an empty in-memory vector store.

    Returns:
        (model, embeddings, vector_store) tuple.
    """
    model = init_chat_model(MODEL_ID)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = InMemoryVectorStore(embeddings)
    return model, embeddings, vector_store


# ============================================================
# Section: Document Loading & Splitting
# ============================================================

def load_and_index_blog(vector_store: InMemoryVectorStore) -> None:
    """Fetch a blog post, split it into chunks, and index into the vector store.

    Uses BeautifulSoup SoupStrainer to extract only post-title,
    post-header, and post-content elements from the HTML.
    """
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(BLOG_URL,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print(f"Total characters: {len(docs[0].page_content)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    vector_store.add_documents(documents=all_splits)


# ============================================================
# Section: RAG Chain (middleware-based, no tools)
# ============================================================

def run_rag_chain(model, vector_store: InMemoryVectorStore) -> None:
    """'Search First, Then Ask Model' -- context is injected via dynamic_prompt middleware.

    No tools are given to the model; the middleware silently prepends
    retrieved context to the system message before every model call.
    """

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject retrieved context into the system message."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

    agent = create_agent(model, tools=[], middleware=[prompt_with_context])

    query = "What is task decomposition?"
    # agent.stream outputs all intermediate states, not just the final answer
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


# ============================================================
# Section: RAG Agent (tool-based, model decides when to retrieve)
# ============================================================

def run_rag_agent(model, vector_store: InMemoryVectorStore) -> None:
    """The model is given a retriever tool and decides when/how often to call it.

    Enables multi-hop reasoning: the model can retrieve, read, then
    retrieve again with a refined query.
    """

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information related to a query from the indexed blog post."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    agent = create_agent(model, [retrieve_context], system_prompt=prompt)

    query = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    # stream_mode="values" outputs the entire state (list of messages) at every step
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model, embeddings, vector_store = init_components()
    load_and_index_blog(vector_store)

    print("\n===== RAG Chain (middleware) =====\n")
    run_rag_chain(model, vector_store)

    print("\n===== RAG Agent (tool-based) =====\n")
    run_rag_agent(model, vector_store)
