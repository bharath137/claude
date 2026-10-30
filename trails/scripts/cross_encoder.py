from pathlib import Path
from uuid import uuid4
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# 1. Load your documents
def load_documents(file_path: str):
    # Extract text one page at a time
    loader = PyPDFLoader(
        str(file_path),
        mode="page"
    )
    pages = loader.load()
    documents = []

    for page in pages[2:]:
        page_number = page.metadata["page"] + 1
        cleaned_text = clean_page(page.page_content, page_number)

        if cleaned_text:
            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": file_path.name,
                        "page_number": page_number,
                        "company": "37signals"
                    }
                )
            )

    return documents


def clean_page(text, page_number):
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if re.fullmatch(
            r"37SIGNALS\s*/\s*EMPLOYEE HANDBOOK",
            line
        ):
            continue

        if line.startswith("Uploaded archive edition |"):
            continue

        if line == str(page_number):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_vector_store(documents):
    # Split pages into smaller searchable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"chunk_{index:04d}"

    # Convert one text chunk into an embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build the searchable vector index
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        collection_name=f"handbook_{uuid4().hex[:8]}"
    )

    return vector_store


def similarity_retrieve(vector_store, question: str, top_k=20):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    retrieved_docs = retriever.invoke(question)

    return retrieved_docs


def tokenize(text: str) -> list[str]:
    # Lowercase and strip punctuation so "Leave," matches "leave"
    return re.findall(r"\w+", text.lower())


def bm25_retrieve(vector_store, question: str, top_k=20):
    # Pull the stored chunks back out of Chroma, keeping their metadata
    data = vector_store.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(data["documents"], data["metadatas"])
    ]

    # Build a BM25 index over the chunk text (keyword-based, no embeddings)
    retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize,
        k=top_k
    )
    retrieved_docs = retriever.invoke(question)

    return retrieved_docs


def hybrid_retrieve(
    vector_store,
    question: str,
    top_k=20,
    candidate_k=20,
    rrf_k=60,
    similarity_weight=1.0,
    bm25_weight=1.0,
):
    # 1. Similarity candidates (semantic matches)
    similarity_docs = vector_store.similarity_search(question, k=candidate_k)

    # 2. BM25 candidates (keyword matches), built from the same stored chunks
    data = vector_store.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(data["documents"], data["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize,
        k=candidate_k
    )
    bm25_docs = bm25.invoke(question)

    # 3. Reciprocal Rank Fusion: score = weight / (rrf_k + rank)
    scores = {}
    docs_by_id = {}

    for weight, docs in [
        (similarity_weight, similarity_docs),
        (bm25_weight, bm25_docs),
    ]:
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.metadata["chunk_id"]
            docs_by_id[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)

    # 4. Keep the best top_k fused results
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    retrieved_docs = [docs_by_id[chunk_id] for chunk_id in ranked_ids]

    return retrieved_docs

_cross_encoder = None
def rerank_retrieve(question: str, docs, top_k=3,
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

    # Score every (question, chunk) pair together
    pairs = [(question, doc.page_content) for doc in docs]
    scores = _cross_encoder.score(pairs)

    # Sort by cross-encoder score, highest first
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

    reranked_docs = []
    for doc, score in ranked:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    return reranked_docs


def print_results(docs, label: str, preview_chars: int = 200):
    print(f"\n{'=' * 60}")
    print(f"{label.upper()} RESULTS")
    print("=" * 60)

    for rank, doc in enumerate(docs, start=1):
        preview = doc.page_content[:preview_chars].replace("\n", " ")
        print(f"\n#{rank}")
        print("Chunk:   ", doc.metadata["chunk_id"])
        print("PDF page:", doc.metadata["page_number"])
        if "rerank_score" in doc.metadata:
            print("Rerank score:", round(doc.metadata["rerank_score"], 3))
        print("Preview: ", preview + ("..." if len(doc.page_content) > preview_chars else ""))


if __name__ == "__main__":
    # 1. Load documents
    file_path = Path(__file__).resolve().parent.parent / "data" / "37signals_Complete_Employee_Handbook.pdf"
    documents = load_documents(file_path)

    # 2. Build vector store
    vector_store = build_vector_store(documents)

    # 3. Query
    question = (
        "What paid parental leave is available to primary and "
        "secondary caregivers, and when must it be taken?"
    )

    # 4. Similarity retriever
    similarity_docs = similarity_retrieve(vector_store, question)
    print_results(similarity_docs, "similarity")

    # 5. BM-25 retriever
    bm25_docs = bm25_retrieve(vector_store, question)
    print_results(bm25_docs, "bm25")

    # 6. Hybrid retriever
    hybrid_docs = hybrid_retrieve(vector_store, question)
    print_results(hybrid_docs, "hybrid")

    # 7. Cross-encoder rerank, keep top 3
    reranked_docs = rerank_retrieve(question, hybrid_docs, top_k=3)
    print_results(reranked_docs, "reranked")
