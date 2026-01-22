import os

DEFAULT_CHAT_MODEL = os.getenv("STOCK_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_SUMMARY_MODEL = os.getenv("STOCK_SUMMARY_MODEL", DEFAULT_CHAT_MODEL)
DEFAULT_EMBEDDING_MODEL = os.getenv("STOCK_EMBED_MODEL", "text-embedding-3-small")

DATA_DIR = os.getenv("STOCK_DATA_DIR", "/home/gungnir/work/stock_news_agents/data")
RAG_DIR = os.path.join(DATA_DIR, "rag_store")

NEWS_RELEVANCE_THRESHOLD = float(os.getenv("STOCK_NEWS_RAG_THRESHOLD", "0.62"))
