from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None


def _json_env(var_name: str, *, default: Any) -> Any:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return default

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"{var_name} must be valid JSON: {e.msg}"
        raise ValueError(msg) from e


def get_mcp_server_configs() -> dict[str, dict[str, Any]]:
    """Build Notion MCP server config for langchain-mcp-adapters."""
    notion_mcp_url = os.environ.get("NOTION_MCP_URL", "").strip()
    if not notion_mcp_url:
        return {}

    notion_headers = _json_env("NOTION_MCP_HEADERS_JSON", default={})
    if not isinstance(notion_headers, dict):
        msg = "NOTION_MCP_HEADERS_JSON must be a JSON object."
        raise ValueError(msg)

    return {
        "notion": {
            "transport": os.environ.get("NOTION_MCP_TRANSPORT", "http").strip().lower() or "http",
            "url": notion_mcp_url,
            "headers": {str(key): str(value) for key, value in notion_headers.items()},
        }
    }


async def load_mcp_tools() -> list[Any]:
    """Load LangChain-compatible tools from configured MCP servers."""
    server_configs = get_mcp_server_configs()
    if not server_configs:
        return []

    try:
        if MultiServerMCPClient is None:
            logger.warning("langchain-mcp-adapters is not installed; skipping MCP tool loading")
            return []
        client = MultiServerMCPClient(server_configs)
        return await client.get_tools()
    except Exception:
        logger.exception("Failed to load MCP tools")
        return []
