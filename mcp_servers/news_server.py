from __future__ import annotations

from typing import Dict, List, Optional
import datetime
import html
import json
import logging
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("news_server")
logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value


def _llm_expand_queries(query: str, languages: List[str], max_per_lang: int) -> Dict[str, List[str]]:
    _load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    model = os.getenv("STOCK_CHAT_MODEL", "gpt-4o-mini")
    system = (
        "You generate concise search queries for financial news. "
        "Return ONLY strict JSON: {\"zh\":[],\"ja\":[],\"en\":[]} with up to N items each."
    )
    user = (
        f"Base query: {query}\n"
        f"Languages: {', '.join(languages)}\n"
        f"N={max_per_lang}\n"
        "Each query should be a short phrase suitable for Google News search.\n"
        "Vary intents (price move, earnings, product, regulation, lawsuit, supply chain) without listing intents.\n"
        "Do not include explanations."
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        result: Dict[str, List[str]] = {}
        for lang in languages:
            values = data.get(lang, [])
            if isinstance(values, list):
                cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
                result[lang] = cleaned[:max_per_lang]
        logger.info("llm query variants=%s", result)
        return result
    except Exception as exc:
        logger.warning("llm expand failed: %s", exc)
        return {}

def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def _search_rss(
    query: str,
    lang: str,
    max_results: int,
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> List[Dict[str, str]]:
    locale = {
        "zh": ("zh-CN", "CN", "CN:zh-Hans"),
        "ja": ("ja", "JP", "JP:ja"),
        "en": ("en-US", "US", "US:en"),
    }.get(lang, ("en-US", "US", "US:en"))
    hl, gl, ceid = locale
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    items = []
    for entry in root.findall(".//item"):
        title = entry.findtext("title", default="") or ""
        link = entry.findtext("link", default="") or ""
        desc = entry.findtext("description", default="") or ""
        pub_raw = entry.findtext("pubDate", default="") or ""
        pub_date = None
        if pub_raw:
            try:
                pub_date = parsedate_to_datetime(pub_raw).date()
            except Exception:
                pub_date = None
        if start_date and end_date:
            if not pub_date or not (start_date <= pub_date <= end_date):
                continue
        snippet = html.unescape(_strip_tags(desc))
        items.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet,
                "source_type": "media",
                "language": lang,
                "published": pub_raw,
            }
        )
        if len(items) >= max_results:
            break
    logger.info("rss items=%d lang=%s query=%s", len(items), lang, query)
    return items


@mcp.tool()
def search_news(
    query: str,
    limit: int,
    languages: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict[str, str]]:
    per_lang = max(1, (limit + max(1, len(languages)) - 1) // max(1, len(languages)))
    collected: List[Dict[str, str]] = []
    seen = set()
    buckets: Dict[str, List[Dict[str, str]]] = {}
    expansions = _llm_expand_queries(query, languages, max_per_lang=8)
    start_dt = datetime.date.fromisoformat(start_date) if start_date else None
    end_dt = datetime.date.fromisoformat(end_date) if end_date else None

    def _add(items: List[Dict[str, str]], bucket: List[Dict[str, str]]) -> None:
        for item in items:
            key = item.get("link") or item.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            bucket.append(item)

    for lang in languages:
        bucket: List[Dict[str, str]] = []
        variants = [query] + (expansions.get(lang) or [])
        logger.info("rss variants=%s lang=%s", variants, lang)
        for q in variants:
            try:
                _add(_search_rss(q, lang, max(limit, per_lang * 3), start_dt, end_dt), bucket)
            except Exception:
                continue
            if len(bucket) >= per_lang:
                break
        buckets[lang] = bucket
        collected.extend(bucket[:per_lang])

    if len(collected) < limit:
        for lang in languages:
            bucket = buckets.get(lang, [])
            if len(bucket) < limit:
                variants = [query] + (expansions.get(lang) or [])
                for q in variants:
                    try:
                        _add(_search_rss(q, lang, max(limit, per_lang * 3), start_dt, end_dt), bucket)
                    except Exception:
                        continue
                    if len(bucket) >= limit:
                        break
            for item in bucket[per_lang:]:
                collected.append(item)
                if len(collected) >= limit:
                    break
            if len(collected) >= limit:
                break

    return collected[:limit]


if __name__ == "__main__":
    mcp.run()
