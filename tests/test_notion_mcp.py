from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from agent.utils.mcp import get_mcp_server_configs, load_mcp_tools

mcp_module = importlib.import_module("agent.utils.mcp")


def test_get_mcp_server_configs_supports_notion_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTION_MCP_URL", "http://localhost:3000/mcp")
    monkeypatch.setenv(
        "NOTION_MCP_HEADERS_JSON",
        '{"Authorization":"Bearer notion-mcp-secret"}',
    )

    config = get_mcp_server_configs()

    assert config == {
        "notion": {
            "transport": "http",
            "url": "http://localhost:3000/mcp",
            "headers": {"Authorization": "Bearer notion-mcp-secret"},
        }
    }


def test_load_mcp_tools_uses_langchain_mcp_adapter_for_notion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTION_MCP_URL", "http://notion-mcp.internal:3000/mcp")

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, server_configs: dict[str, Any]) -> None:
            captured["server_configs"] = server_configs

        async def get_tools(self) -> list[str]:
            return ["mcp-notion-search"]

    monkeypatch.setattr(mcp_module, "MultiServerMCPClient", FakeClient, raising=False)

    result = asyncio.run(load_mcp_tools())

    assert captured["server_configs"] == {
        "notion": {
            "transport": "http",
            "url": "http://notion-mcp.internal:3000/mcp",
            "headers": {},
        }
    }
    assert result == ["mcp-notion-search"]
