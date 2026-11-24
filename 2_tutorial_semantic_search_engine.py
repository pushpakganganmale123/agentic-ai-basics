"""
Tutorial 2: Semantic Search Engine

Builds a semantic search pipeline from scratch:
  - Load documents (in-memory & PDF)
  - Split into chunks
  - Embed with HuggingFace
  - Store in a vector store
  - Run similarity search & batch retrieval

Required API keys: None (uses local HuggingFace embeddings)

Reference: https://docs.langchain.com/oss/python/langchain/knowledge-base
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Uncomment to use Google embeddings instead of HuggingFace:
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PDF_PATH = "nke-10k-2023.pdf"


# ============================================================
# Section: Document Loading
# ============================================================


def create_sample_documents() -> list[Document]:
    """Create a small in-memory document set for quick testing."""
    return [
        Document(page_content="dogs are stupid", metadata={"source": "dogs-doc"}),
        Document(page_content="cats are villans", metadata={"source": "doc-cats"}),
    ]


def load_pdf(file_path: str = PDF_PATH) -> list[Document]:
    """Load a PDF into LangChain Documents (one document per page).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects, one per PDF page.
    """
    loader = PyPDFLoader(file_path)  # each page is loaded in one document
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from '{file_path}'")
    print(f"First 100 chars: {docs[0].page_content[:100]}")
    return docs


# ============================================================
# Section: Text Splitting
# ============================================================


def split_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    add_start_index stores the character offset, that represents where each chunk begins
    in the original document -- useful for tracing back to source.

    Args:
        docs: Full documents to split.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits


# ============================================================
# Section: Embeddings & Vector Store
# ============================================================


def build_vector_store(
    splits: list[Document],
    model_name: str = EMBEDDING_MODEL,
    max_docs: int | None = None,
) -> tuple[InMemoryVectorStore, HuggingFaceEmbeddings]:
    """Embed document chunks and load them into an in-memory vector store.

    Args:
        splits: Document chunks to embed and store.
        model_name: HuggingFace embedding model identifier.
        max_docs: If set, only index the first N chunks (useful for quick tests).

    Returns:
        (vector_store, embeddings) tuple.
    """
    # Uncomment to use Google embeddings:
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Quick sanity check on the first chunk
    sample_vec = embeddings.embed_query(splits[0].page_content[:300])
    print(f"Embedding dim sample (first 10): {sample_vec[:10]}")

    vector_store = InMemoryVectorStore(embeddings)
    docs_to_index = splits[:max_docs] if max_docs else splits
    ids = vector_store.add_documents(documents=docs_to_index)
    print(f"Indexed {len(ids)} documents")

    return vector_store, embeddings


# ============================================================
# Section: Similarity Search
# ============================================================


def run_similarity_search(
    vector_store: InMemoryVectorStore,
    embeddings: HuggingFaceEmbeddings,
    query: str,
) -> None:
    """Run similarity search two ways and verify they match.

    1. Text-based search (vector_store embeds the query internally).
    2. Pre-computed vector search (manual embed, then search by vector).

    Args:
        vector_store: Populated vector store.
        embeddings: Embedding model for manual query embedding.
        query: Natural-language question to search for.
    """
    results = vector_store.similarity_search(query)
    print(f"\n[Text search] Top result:\n{results[0]}")

    # Should return the same result -- vector_store auto-embeds internally
    query_vector = embeddings.embed_query(query)
    results_emb = vector_store.similarity_search_by_vector(query_vector)
    print(f"\n[Vector search] Top result:\n{results_emb[0]}")


# ============================================================
# Section: Retriever (invoke / batch interface)
# ============================================================


def run_batch_retrieval(
    vector_store: InMemoryVectorStore,
    queries: list[str],
    k: int = 1,
) -> list:
    """Wrap the vector store as a Retriever and batch-query it.

    The retriever exposes `.invoke()` and `.batch()` for pipeline integration.

    Args:
        vector_store: Populated vector store.
        queries: List of questions to retrieve answers for.
        k: Number of documents to return per query.

    Returns:
        List of retrieval results, one per query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    results = retriever.batch(queries)
    for q, r in zip(queries, results):
        print(f"\nQuery: {q}\nResult: {r}\n")
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    docs = load_pdf()
    all_splits = split_documents(docs)
    vector_store, embeddings = build_vector_store(all_splits, max_docs=20)

    run_similarity_search(
        vector_store,
        embeddings,
        query="How many distribution centers does Nike have in the US?",
    )

    run_batch_retrieval(
        vector_store,
        queries=[
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )
