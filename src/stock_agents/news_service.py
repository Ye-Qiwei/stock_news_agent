from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Dict, List

from langchain_core.documents import Document

from stock_agents.mcp_client import MCPToolCall, call_tool_sync, unwrap_mcp_content
from stock_agents.rag_store import add_documents, query_documents, query_documents_exact
from stock_agents.llm_factory import get_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from stock_agents.config import NEWS_RELEVANCE_THRESHOLD
import logging


class NewsSearchDirection(str, Enum):
    COMPANY = "company"
    INDUSTRY = "industry"


@dataclass
class NewsItem:
    title: str
    link: str
    snippet: str
    language: str
    source_type: str


class NewsService:
    def __init__(self, news_server_script: str) -> None:
        self.news_call = MCPToolCall(server_script=news_server_script, tool_name="search_news")
        self._logger = logging.getLogger(__name__)

    def search_news(
        self,
        ticker: str,
        week_start: date,
        direction: NewsSearchDirection,
        industry: str,
        company_name: str = "",
        limit: int = 10,
    ) -> List[Dict[str, str]]:
        query = self._build_query(ticker, direction, industry, company_name)
        week_end = week_start + timedelta(days=6)
        rag_hits = self._query_rag_exact(query, direction, week_start)
        if rag_hits:
            return rag_hits[:limit]
        rag_hits = self._query_rag(query, direction, limit, week_start, week_end)
        if rag_hits and len(rag_hits) >= limit:
            return rag_hits

        normalized = self._fetch_mcp(query, week_start, week_end, limit)
        retry_items: List[Dict[str, str]] = []
        combined = []
        seen = set()
        for item in (rag_hits or []) + normalized:
            key = item.get("link") or item.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            combined.append(item)
            if len(combined) >= limit:
                break

        if len(combined) < limit:
            alt_query = self._rewrite_query(query, direction, company_name, industry)
            if alt_query and alt_query != query:
                retry_items = self._fetch_mcp(alt_query, week_start, week_end, limit)
                for item in retry_items:
                    key = item.get("link") or item.get("title")
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    combined.append(item)
                    if len(combined) >= limit:
                        break
        self._store_rag(query, direction, normalized + retry_items, week_start, week_end)
        return combined[:limit]

    def _fetch_mcp(self, query: str, week_start: date, week_end: date, limit: int) -> List[Dict[str, str]]:
        payload = {
            "query": query,
            "limit": limit,
            "languages": ["zh", "ja", "en"],
            "start_date": week_start.isoformat(),
            "end_date": week_end.isoformat(),
        }
        result = call_tool_sync(self.news_call, payload)
        items = unwrap_mcp_content(result)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            self._logger.warning("news MCP response type=%s value=%s", type(items), items)
            return []
        self._logger.info("news MCP items=%d sample=%s", len(items), items[:1])
        return [self._normalize(item) for item in items]

    def _rewrite_query(
        self,
        query: str,
        direction: NewsSearchDirection,
        company_name: str,
        industry: str,
    ) -> str:
        llm = get_chat_model(temperature=0)
        parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the query to fetch more relevant financial news. "
                    "Return ONLY the new query string.",
                ),
                (
                    "human",
                    "Original query: {query}\n"
                    "Direction: {direction}\n"
                    "Company name: {company}\n"
                    "Industry: {industry}\n"
                    "Return a concise query.",
                ),
            ]
        )
        chain = prompt | llm | parser
        try:
            return chain.invoke(
                {
                    "query": query,
                    "direction": direction.value,
                    "company": company_name,
                    "industry": industry,
                }
            ).strip()
        except Exception:
            return ""

    def _build_query(self, ticker: str, direction: NewsSearchDirection, industry: str, company_name: str) -> str:
        if direction == NewsSearchDirection.INDUSTRY:
            return industry
        company_name = company_name.strip()
        if company_name:
            return f"{ticker} {company_name}".strip()
        return ticker

    def _normalize(self, item: Dict[str, str]) -> Dict[str, str]:
        return {
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "language": item.get("language", ""),
            "source_type": item.get("source_type", ""),
        }

    def _query_rag(
        self,
        query: str,
        direction: NewsSearchDirection,
        limit: int,
        week_start: date,
        week_end: date,
    ) -> List[Dict[str, str]]:
        docs = query_documents(query, k=limit)
        if not docs:
            return []
        top_score = docs[0][1]
        if top_score < NEWS_RELEVANCE_THRESHOLD:
            return []
        hits = []
        for doc, score in docs:
            if score < NEWS_RELEVANCE_THRESHOLD:
                continue
            published_raw = doc.metadata.get("published", "")
            pub_date = None
            if published_raw:
                try:
                    pub_date = parsedate_to_datetime(published_raw).date()
                except Exception:
                    pub_date = None
            if pub_date and not (week_start <= pub_date <= week_end):
                continue
            if published_raw and not pub_date:
                continue
            data = doc.metadata.copy()
            data["title"] = doc.metadata.get("title", "")
            data["link"] = doc.metadata.get("link", "")
            data["snippet"] = doc.page_content
            hits.append(data)
        return hits[:limit]

    def _store_rag(
        self,
        query: str,
        direction: NewsSearchDirection,
        items: List[Dict[str, str]],
        week_start: date,
        week_end: date,
    ) -> None:
        docs = []
        for item in items:
            docs.append(
                Document(
                    page_content=item.get("snippet", ""),
                    metadata={
                        "query": query,
                        "direction": direction.value,
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "language": item.get("language", ""),
                        "source_type": item.get("source_type", ""),
                        "published": item.get("published", ""),
                        "week_start": week_start.isoformat(),
                        "week_end": week_end.isoformat(),
                    },
                )
            )
        add_documents(docs)
    def _query_rag_exact(
        self,
        query: str,
        direction: NewsSearchDirection,
        week_start: date,
    ) -> List[Dict[str, str]]:
        filters = {
            "query": query,
            "direction": direction.value,
            "week_start": week_start.isoformat(),
        }
        docs = query_documents_exact(filters)
        if not docs:
            return []
        hits = []
        for doc in docs:
            data = doc.metadata.copy()
            data["title"] = doc.metadata.get("title", "")
            data["link"] = doc.metadata.get("link", "")
            data["snippet"] = doc.page_content
            hits.append(data)
        return hits
