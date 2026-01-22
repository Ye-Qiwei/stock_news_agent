from __future__ import annotations

from typing import Dict, Mapping

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from stock_agents.llm_factory import get_chat_model


_FALLBACK: Dict[str, str] = {
    "apple": "AAPL",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "meta": "META",
    "toyota": "TM",
    "sony": "SONY",
    "ntt": "NTT",
}

_FALLBACK_BY_MARKET: Dict[str, Mapping[str, str]] = {
    "toyota": {"US": "TM", "JP": "7203"},
    "sony": {"US": "SONY", "JP": "6758"},
    "ntt": {"US": "NTT", "JP": "9432"},
    "nintendo": {"US": "NTDOY", "JP": "7974"},
    "honda": {"US": "HMC", "JP": "7267"},
    "softbank": {"US": "SFTBY", "JP": "9984"},
}


_FALLBACK_COMPANY: Dict[str, str] = {
    "aapl": "Apple",
    "googl": "Alphabet",
    "goog": "Alphabet",
    "msft": "Microsoft",
    "nvda": "NVIDIA",
    "tsla": "Tesla",
    "amzn": "Amazon",
    "meta": "Meta",
    "7203": "Toyota",
    "6758": "Sony",
    "9432": "NTT",
    "tm": "Toyota",
    "sony": "Sony",
    "ntt": "NTT",
    "ntdoy": "Nintendo",
    "7974": "Nintendo",
    "hmc": "Honda",
    "7267": "Honda",
    "sftby": "SoftBank",
    "9984": "SoftBank",
}


def infer_ticker(company_name: str, market: str) -> str:
    name = company_name.strip()
    if not name:
        return ""
    key = name.lower()
    market_key = market.strip().upper()
    if key in _FALLBACK_BY_MARKET:
        return _FALLBACK_BY_MARKET[key].get(market_key, _FALLBACK_BY_MARKET[key].get("US", ""))
    if key in _FALLBACK:
        return _FALLBACK[key]

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
    fallback = _FALLBACK_COMPANY.get(symbol.lower())
    if fallback:
        return fallback

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
    symbol = ticker.strip().upper()
    if not symbol:
        return ""
    company = _FALLBACK_COMPANY.get(symbol.lower())
    if not company:
        return symbol
    mapped = infer_ticker(company, market)
    return mapped or symbol
