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


def _build_http_server_config(prefix: str) -> dict[str, Any] | None:
    url = os.environ.get(f"{prefix}_URL", "").strip()
    if not url:
        return None

    headers = _json_env(f"{prefix}_HEADERS_JSON", default={})
    if not isinstance(headers, dict):
        msg = f"{prefix}_HEADERS_JSON must be a JSON object."
        raise ValueError(msg)

    return {
        "transport": os.environ.get(f"{prefix}_TRANSPORT", "http").strip().lower() or "http",
        "url": url,
        "headers": {str(key): str(value) for key, value in headers.items()},
    }


def get_mcp_server_configs() -> dict[str, dict[str, Any]]:
    """Build MCP server config for langchain-mcp-adapters."""
    server_configs: dict[str, dict[str, Any]] = {}

    notion_config = _build_http_server_config("NOTION_MCP")
    if notion_config:
        server_configs["notion"] = notion_config

    sentry_config = _build_http_server_config("SENTRY_MCP")
    if sentry_config:
        server_configs["sentry"] = sentry_config

    return server_configs


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
