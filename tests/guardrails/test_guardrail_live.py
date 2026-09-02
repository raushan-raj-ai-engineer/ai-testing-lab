from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import (
    streamable_http_client,
)

MCP_URL = os.getenv(
    "PLAYWRIGHT_MCP_URL",
    "http://localhost:8931/mcp",
)


RUN_GUARDRAIL_LIVE = (
    os.getenv(
        "RUN_GUARDRAIL_LIVE",
        "0",
    )
    == "1"
)


pytestmark = [
    pytest.mark.guardrail,
    pytest.mark.guardrail_live,
]


@asynccontextmanager
async def open_mcp_session(
    url: str = MCP_URL,
) -> AsyncIterator[ClientSession]:
    """
    Open and initialize an MCP Streamable HTTP session.

    Compatible with MCP SDK variants where
    streamable_http_client() returns:

        (
            read_stream,
            write_stream,
            get_session_id,
        )

    We only need the first two streams for ClientSession.
    """

    async with streamable_http_client(
        url,
    ) as streams:
        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            yield session


@pytest.mark.skipif(
    not RUN_GUARDRAIL_LIVE,
    reason=("Set RUN_GUARDRAIL_LIVE=1 to run live guardrail tests"),
)
@pytest.mark.anyio
async def test_playwright_mcp_is_reachable() -> None:
    """
    Confirm the Playwright MCP server can be reached
    and exposes browser tools.
    """

    async with open_mcp_session() as session:
        result = await session.list_tools()

        tool_names = [tool.name for tool in result.tools]

        assert tool_names, "MCP server returned no tools"

        assert any(
            name.startswith(
                "browser_",
            )
            for name in tool_names
        ), "Expected Playwright browser tools were not available"


@pytest.mark.skipif(
    not RUN_GUARDRAIL_LIVE,
    reason=("Set RUN_GUARDRAIL_LIVE=1 to run live guardrail tests"),
)
@pytest.mark.anyio
async def test_browser_snapshot_tool_available() -> None:
    """
    Verify that browser_snapshot is exposed
    by Playwright MCP.
    """

    async with open_mcp_session() as session:
        result = await session.list_tools()

        tool_names = {tool.name for tool in result.tools}

        assert "browser_snapshot" in tool_names, (
            "browser_snapshot tool is not available"
        )


@pytest.mark.skipif(
    not RUN_GUARDRAIL_LIVE,
    reason=("Set RUN_GUARDRAIL_LIVE=1 to run live guardrail tests"),
)
@pytest.mark.anyio
async def test_browser_snapshot_can_execute() -> None:
    """
    Execute a real browser_snapshot MCP tool call.
    """

    async with open_mcp_session() as session:
        result = await session.call_tool(
            "browser_snapshot",
            {},
        )

        assert result is not None

        assert not getattr(
            result,
            "isError",
            False,
        ), "browser_snapshot returned an MCP error"

        assert result.content, "browser_snapshot returned no content"


@pytest.mark.skipif(
    not RUN_GUARDRAIL_LIVE,
    reason=("Set RUN_GUARDRAIL_LIVE=1 to run live guardrail tests"),
)
@pytest.mark.anyio
async def test_mcp_does_not_expose_unknown_admin_tool() -> None:
    """
    Least-privilege security check.

    Playwright MCP should not expose arbitrary
    privileged/admin tools.
    """

    async with open_mcp_session() as session:
        result = await session.list_tools()

        tool_names = {tool.name for tool in result.tools}

        blocked_tool_names = {
            "delete_user",
            "admin_shell",
            "execute_system_command",
            "read_secrets",
        }

        exposed_blocked_tools = tool_names & blocked_tool_names

        assert not exposed_blocked_tools, (
            f"Unexpected privileged tools exposed: {sorted(exposed_blocked_tools)}"
        )
