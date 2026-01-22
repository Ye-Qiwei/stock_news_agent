import logging
import os
import shutil
from typing import Dict, Iterable, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from stock_agents.config import RAG_DIR
from stock_agents.embedding_factory import get_embeddings

logger = logging.getLogger(__name__)


def ensure_rag_dir() -> str:
    os.makedirs(RAG_DIR, exist_ok=True)
    return RAG_DIR


def get_vectorstore() -> Chroma:
    ensure_rag_dir()
    embeddings = get_embeddings()
    return Chroma(persist_directory=RAG_DIR, embedding_function=embeddings)


def add_documents(docs: Iterable[Document]) -> None:
    docs_list = list(docs)
    if not docs_list:
        return
    vectorstore = get_vectorstore()
    try:
        vectorstore.add_documents(docs_list)
    except Exception as exc:
        logger.warning("rag write failed, retrying after reset: %s", exc)
        try:
            clear_rag()
            vectorstore = get_vectorstore()
            vectorstore.add_documents(docs_list)
        except Exception as retry_exc:
            logger.warning("rag write failed after reset: %s", retry_exc)


def query_documents(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    vectorstore = get_vectorstore()
    try:
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        logger.warning("rag query failed, fallback to empty: %s", exc)
        return []


def query_documents_exact(filters: Dict[str, str]) -> List[Document]:
    vectorstore = get_vectorstore()
    try:
        results = vectorstore._collection.get(where=filters, include=["documents", "metadatas"])
        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        hits = []
        for text, meta in zip(documents, metadatas):
            hits.append(Document(page_content=text or "", metadata=meta or {}))
        return hits
    except Exception as exc:
        logger.warning("rag exact query failed, fallback to empty: %s", exc)
        return []


def get_rag_size_bytes() -> int:
    if not os.path.exists(RAG_DIR):
        return 0
    total = 0
    for root, _, files in os.walk(RAG_DIR):
        for name in files:
            path = os.path.join(root, name)
            total += os.path.getsize(path)
    return total


def clear_rag() -> None:
    if os.path.exists(RAG_DIR):
        shutil.rmtree(RAG_DIR)
    ensure_rag_dir()
