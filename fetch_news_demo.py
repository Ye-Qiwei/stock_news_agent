# save as fetch_news_demo.py
import re
import html
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")

def fetch_rss(query: str, lang: str, max_results: int = 5):
    locale = {
        "zh": ("zh-CN", "CN", "CN:zh-Hans"),
        "ja": ("ja", "JP", "JP:ja"),
        "en": ("en-US", "US", "US:en"),
    }[lang]
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
        snippet = html.unescape(strip_tags(desc))
        items.append({"title": title, "link": link, "snippet": snippet})
        if len(items) >= max_results:
            break
    return items

if __name__ == "__main__":
    base = "AAPL"
    queries = {
        "zh": [base, f"{base} 财报", f"{base} 股价"],
        "ja": [base, f"{base} 決算", f"{base} 株価"],
        "en": [base, f"{base} earnings", f"{base} stock"],
    }

    all_items = []
    seen = set()
    for lang, qs in queries.items():
        for q in qs:
            for item in fetch_rss(q, lang, max_results=3):
                key = item["link"] or item["title"]
                if key in seen:
                    continue
                seen.add(key)
                all_items.append((lang, item))

    for lang, item in all_items:
        print(f"[{lang}] {item['title']}")
        print(item["link"])
        print(item["snippet"][:120])
        print("-" * 60)