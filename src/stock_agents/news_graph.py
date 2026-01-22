from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
import re
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from stock_agents.llm_factory import get_chat_model, get_summary_model
from stock_agents.news_service import NewsService, NewsSearchDirection


class NewsSummary(BaseModel):
    title: str = Field(description="Original news title")
    summary: List[str] = Field(description="Three sentence summary in Chinese")
    sentiment: str = Field(description="positive or negative")
    score: int = Field(description="1 for positive, -1 for negative")
    link: str = Field(description="News url")
    language: str = Field(description="zh/ja/en")
    source_type: str = Field(description="media/blog")


@dataclass
class NewsRequest:
    ticker: str
    week_start: date
    direction: NewsSearchDirection
    industry: str


class GraphState(BaseModel):
    ticker: str
    week_start: date
    industry: str
    company_name: str = ""
    company_news: List[Dict[str, Any]] = Field(default_factory=list)
    industry_news: List[Dict[str, Any]] = Field(default_factory=list)


def build_summary_chain() -> Any:
    parser = JsonOutputParser(pydantic_object=NewsSummary)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize financial news. Reply ONLY with JSON that matches the schema.",
            ),
            (
                "human",
                "Title: {title}\nSnippet: {snippet}\nLink: {link}\n"
                "Language: {language}\nSource: {source_type}\n"
                "Task: Provide a three-sentence Chinese summary, then classify as positive or negative for the company.\n"
                "Return JSON with fields: title, summary (list of 3 sentences), sentiment, score (1 or -1), link, language, source_type.",
            ),
        ]
    )
    llm = get_summary_model(temperature=0)
    return prompt | llm | parser


def _dump_summary(result: Any) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {"title": "", "summary": [], "sentiment": "neutral", "score": 0, "link": "", "language": "", "source_type": ""}


def _fallback_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    snippet = item.get("snippet", "") or ""
    parts = [p.strip() for p in re.split(r"[。！？.!?]", snippet) if p.strip()]
    summary = (parts + ["", "", ""])[:3]
    return {
        "title": item.get("title", ""),
        "summary": summary,
        "sentiment": "neutral",
        "score": 0,
        "link": item.get("link", ""),
        "language": item.get("language", ""),
        "source_type": item.get("source_type", ""),
    }


def build_graph(news_service: NewsService) -> Any:
    chain = build_summary_chain()
    max_workers = int(os.getenv("STOCK_SUMMARY_CONCURRENCY", "6"))

    def fetch_company(state: GraphState) -> Dict[str, Any]:
        items = news_service.search_news(
            ticker=state.ticker,
            week_start=state.week_start,
            direction=NewsSearchDirection.COMPANY,
            industry=state.industry,
            company_name=state.company_name,
        )
        payloads = []
        for item in items:
            payloads.append(
                {
                    "item": item,
                    "input": {
                        "title": item["title"],
                        "snippet": item["snippet"],
                        "link": item["link"],
                        "language": item["language"],
                        "source_type": item["source_type"],
                    },
                }
            )
        summaries: List[Any] = [None] * len(payloads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(chain.invoke, payload["input"]): idx
                for idx, payload in enumerate(payloads)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                item = payloads[idx]["item"]
                try:
                    summaries[idx] = future.result()
                except Exception:
                    summaries[idx] = _fallback_summary(item)
        return {"company_news": [_dump_summary(entry) for entry in summaries if entry is not None]}

    def fetch_industry(state: GraphState) -> Dict[str, Any]:
        items = news_service.search_news(
            ticker=state.ticker,
            week_start=state.week_start,
            direction=NewsSearchDirection.INDUSTRY,
            industry=state.industry,
            company_name=state.company_name,
        )
        payloads = []
        for item in items:
            payloads.append(
                {
                    "item": item,
                    "input": {
                        "title": item["title"],
                        "snippet": item["snippet"],
                        "link": item["link"],
                        "language": item["language"],
                        "source_type": item["source_type"],
                    },
                }
            )
        summaries: List[Any] = [None] * len(payloads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(chain.invoke, payload["input"]): idx
                for idx, payload in enumerate(payloads)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                item = payloads[idx]["item"]
                try:
                    summaries[idx] = future.result()
                except Exception:
                    summaries[idx] = _fallback_summary(item)
        return {"industry_news": [_dump_summary(entry) for entry in summaries if entry is not None]}

    graph = StateGraph(GraphState)
    graph.add_node("company_news", fetch_company)
    graph.add_node("industry_news", fetch_industry)
    graph.set_entry_point("company_news")
    graph.add_edge("company_news", "industry_news")
    graph.add_edge("industry_news", END)
    return graph.compile()
