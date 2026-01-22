import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class MCPToolCall:
    server_script: str
    tool_name: str


async def call_tool(call: MCPToolCall, arguments: Dict[str, Any]) -> Any:
    server_params = StdioServerParameters(command="python", args=[call.server_script])
    logger.info("mcp call_tool server=%s tool=%s args=%s", call.server_script, call.tool_name, arguments)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(call.tool_name, arguments)
            logger.info("mcp call_tool result_type=%s", type(result))
            return result.content


def call_tool_sync(call: MCPToolCall, arguments: Dict[str, Any]) -> Any:
    return asyncio.run(call_tool(call, arguments))


def unwrap_mcp_content(content: Any) -> Any:
    raw = content
    if isinstance(content, list):
        if content and all(hasattr(item, "text") for item in content):
            parsed_items = []
            for item in content:
                text = item.text
                if isinstance(text, str):
                    try:
                        parsed_items.append(json.loads(text))
                    except json.JSONDecodeError:
                        logger.warning("mcp unwrap JSON decode failed: %s", text[:200])
                        parsed_items.append(text)
                else:
                    parsed_items.append(text)
            if len(parsed_items) == 1:
                return parsed_items[0]
            return parsed_items
        raw = content
    elif hasattr(content, "text"):
        raw = content.text

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("mcp unwrap JSON decode failed: %s", raw[:200])
            return raw
    return raw
