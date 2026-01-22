from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from stock_agents.llm_factory import get_chat_model


_FALLBACK = {
    "aapl": "Consumer Electronics",
    "msft": "Software",
    "nvda": "Semiconductors",
    "tsla": "Automobiles",
    "amzn": "E-Commerce",
    "goog": "Internet Services",
    "meta": "Social Media",
    "7203": "Automobiles",
    "6758": "Consumer Electronics",
    "9432": "Telecom",
}


def infer_industry(ticker: str) -> str:
    normalized = ticker.strip().lower()
    if normalized in _FALLBACK:
        return _FALLBACK[normalized]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You map a ticker to a short industry label. Reply with 1-3 English words only.",
            ),
            (
                "human",
                "Ticker: {ticker}\nReturn only the industry label.",
            ),
        ]
    )
    llm = get_chat_model(temperature=0)
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"ticker": ticker}).strip()
    except Exception:
        return "General Industry"
