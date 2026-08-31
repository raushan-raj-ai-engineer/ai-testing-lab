import re

import pytest
from mcp import Client
from mcp.types import TextContent


@pytest.mark.anyio
async def test_playwright_mcp_tools_are_available():

    async with Client("http://localhost:8931/mcp") as client:
        result = await client.list_tools()

        tool_names = {tool.name for tool in result.tools}

        print(
            "\nPlaywright MCP tools:",
            sorted(tool_names),
        )

        assert "browser_navigate" in tool_names
        assert "browser_snapshot" in tool_names


@pytest.mark.anyio
async def test_playwright_mcp_can_navigate():

    async with Client("http://localhost:8931/mcp") as client:
        result = await client.call_tool(
            "browser_navigate",
            {
                "url": "https://demo.playwright.dev/todomvc",
            },
        )

        print(
            "\nNavigate result:",
            result,
        )

        assert result.is_error is False


@pytest.mark.anyio
async def test_playwright_mcp_snapshot():

    async with Client("http://localhost:8931/mcp") as client:
        await client.call_tool(
            "browser_navigate",
            {
                "url": "https://demo.playwright.dev/todomvc",
            },
        )

        result = await client.call_tool(
            "browser_snapshot",
            {},
        )

        print(
            "\nSnapshot result:",
            result,
        )

        assert result.is_error is False


def get_text_content(result) -> str:
    parts: list[str] = []

    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)

    return "\n".join(parts)


@pytest.mark.anyio
async def test_playwright_mcp_can_add_todo():

    async with Client("http://localhost:8931/mcp") as client:
        await client.call_tool(
            "browser_navigate",
            {
                "url": "https://demo.playwright.dev/todomvc",
            },
        )

        snapshot = await client.call_tool(
            "browser_snapshot",
            {},
        )

        snapshot_text = get_text_content(snapshot)

        print(
            "\nSnapshot:\n",
            snapshot_text,
        )

        match = re.search(
            r'textbox "What needs to be done\?".*?\[ref=(e\d+)\]',
            snapshot_text,
        )

        assert match is not None

        textbox_ref = match.group(1)

        print(
            "\nTextbox ref:",
            textbox_ref,
        )

        result = await client.call_tool(
            "browser_type",
            {
                "target": textbox_ref,
                "text": "Learn Playwright MCP",
                "submit": True,
            },
        )

        assert result.is_error is False

        verification = await client.call_tool(
            "browser_snapshot",
            {},
        )

        verification_text = get_text_content(verification)

        print(
            "\nVerification snapshot:\n",
            verification_text,
        )

        assert "Learn Playwright MCP" in verification_text
