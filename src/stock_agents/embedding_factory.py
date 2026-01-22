import os
from typing import Any

from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> Any:
    provider = os.getenv("STOCK_EMBED_PROVIDER", "openai").lower()
    model = os.getenv("STOCK_EMBED_MODEL", "text-embedding-3-small")
    if provider == "jina":
        api_key = os.getenv("STOCK_EMBED_API_KEY") or os.getenv("JINA_API_KEY")
        return JinaEmbeddings(model=model, jina_api_key=api_key)
    return OpenAIEmbeddings(model=model)
