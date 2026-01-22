from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from stock_agents.llm_factory import get_chat_model


_US_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
_JP_TICKER_PATTERN = re.compile(r"^\d{4}$")


def looks_like_ticker(value: str, market: str) -> bool:
    symbol = value.strip()
    if not symbol:
        return False
    market_key = market.strip().upper()
    if market_key == "US":
        return bool(_US_TICKER_PATTERN.match(symbol.upper()))
    if market_key == "JP":
        return bool(_JP_TICKER_PATTERN.match(symbol))
    return False


def infer_ticker(company_name: str, market: str) -> str:
    name = company_name.strip()
    if not name:
        return ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You map a company name to its stock ticker. "
                "If a company is listed in both US and JP, return the ticker for the given Market. "
                "For JP, return the numeric ticker only. For US, return the ticker only. "
                "Return ONLY the ticker symbol. No punctuation, no exchange suffix.",
            ),
            (
                "human",
                "Company: {company}\nMarket: {market}\nReturn only the ticker.",
            ),
        ]
    )
    llm = get_chat_model(temperature=0)
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"company": name, "market": market}).strip().upper()
    except Exception:
        return ""


def infer_company_name(ticker: str, market: str) -> str:
    symbol = ticker.strip().upper()
    if not symbol:
        return ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You map a stock ticker to its company name. "
                "Return ONLY the company name. No extra text.",
            ),
            (
                "human",
                "Ticker: {ticker}\nMarket: {market}\nReturn only the company name.",
            ),
        ]
    )
    llm = get_chat_model(temperature=0)
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"ticker": symbol, "market": market}).strip()
    except Exception:
        return ""


def normalize_ticker_for_market(ticker: str, market: str) -> str:
    symbol = ticker.strip()
    if not symbol:
        return ""
    company = infer_company_name(symbol, market)
    if not company:
        return symbol
    mapped = infer_ticker(company, market)
    return mapped or symbol
