from __future__ import annotations

from io import StringIO
from typing import Any, Dict, List

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("price_server")


def _build_stooq_symbol(ticker: str, market: str) -> str:
    raw = ticker.strip().lower()
    if "." in raw:
        if raw.endswith(".t"):
            return f"{raw[:-2]}.jp"
        return raw
    suffix = ".us" if market.upper() == "US" else ".jp"
    return f"{raw}{suffix}"


@mcp.tool()
def fetch_price(ticker: str, market: str) -> Dict[str, Any]:
    symbol = _build_stooq_symbol(ticker, market)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        return {"rows": [], "source": url}

    first_line = text.splitlines()[0].lower()
    if first_line.startswith("date,"):
        frame = pd.read_csv(StringIO(text))
    else:
        frame = pd.read_csv(
            StringIO(text),
            header=None,
            names=["Date", "Open", "High", "Low", "Close", "Volume"],
        )

    frame.rename(columns=str.lower, inplace=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")

    rows = frame[["date", "close"]].copy()
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
    rows["close"] = rows["close"].map(lambda value: f"{value:.6f}")

    return {"rows": rows.to_dict(orient="records"), "source": url}


if __name__ == "__main__":
    mcp.run()
