import logging
from typing import Dict, List, Tuple

import pandas as pd

from stock_agents.mcp_client import MCPToolCall, call_tool_sync, unwrap_mcp_content


logger = logging.getLogger(__name__)


class PriceService:
    def __init__(self, price_server_script: str) -> None:
        self.price_call = MCPToolCall(server_script=price_server_script, tool_name="fetch_price")

    def fetch_price(self, ticker: str, market: str) -> Tuple[pd.DataFrame, str]:
        payload = {"ticker": ticker, "market": market}
        logger.info("fetch_price payload=%s", payload)
        result = call_tool_sync(self.price_call, payload)
        logger.info("fetch_price raw_result_type=%s", type(result))
        data = unwrap_mcp_content(result)
        logger.info("fetch_price decoded_type=%s", type(data))
        if not isinstance(data, dict):
            logger.warning("fetch_price unexpected response: %s", data)
            return pd.DataFrame(), ""
        rows = data.get("rows", [])
        source = data.get("source", "")
        logger.info("fetch_price rows=%d source=%s", len(rows), source)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame, source
        frame["date"] = pd.to_datetime(frame["date"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["close"]).sort_values("date")
        return frame, source
