import pytest
from mcp import Client

from src.mcp_server.qa_server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():

    async with Client(
        mcp,
        raise_exceptions=True,
    ) as connected_client:
        yield connected_client


@pytest.mark.anyio
async def test_calculate_pass_rate(
    client: Client,
):

    result = await client.call_tool(
        "calculate_pass_rate",
        {
            "passed": 95,
            "total": 100,
        },
    )

    assert result.is_error is False

    assert result.structured_content == {"result": 95.0}


@pytest.mark.anyio
async def test_release_allowed(
    client: Client,
):

    result = await client.call_tool(
        "release_decision",
        {
            "pass_rate": 98,
            "critical_failures": 0,
        },
    )

    assert result.is_error is False

    assert result.structured_content == {"result": "RELEASE"}


@pytest.mark.anyio
async def test_release_blocked_for_low_pass_rate(
    client: Client,
):

    result = await client.call_tool(
        "release_decision",
        {
            "pass_rate": 90,
            "critical_failures": 0,
        },
    )

    assert result.is_error is False

    assert result.structured_content == {"result": "BLOCK"}


@pytest.mark.anyio
async def test_release_blocked_for_critical_failure(
    client: Client,
):

    result = await client.call_tool(
        "release_decision",
        {
            "pass_rate": 100,
            "critical_failures": 1,
        },
    )

    assert result.is_error is False

    assert result.structured_content == {"result": "BLOCK"}


@pytest.mark.anyio
async def test_tools_are_exposed(
    client: Client,
):

    result = await client.list_tools()

    tool_names = {tool.name for tool in result.tools}

    assert "calculate_pass_rate" in tool_names
    assert "release_decision" in tool_names


@pytest.mark.anyio
async def test_release_decision_schema(
    client: Client,
):

    result = await client.list_tools()

    tool = next(tool for tool in result.tools if tool.name == "release_decision")

    schema = tool.input_schema

    assert schema["properties"]["pass_rate"]["type"] == "number"

    assert schema["properties"]["critical_failures"]["type"] == "integer"

    assert "pass_rate" in schema["required"]

    assert "critical_failures" in schema["required"]


@pytest.mark.anyio
async def test_invalid_pass_rate_is_rejected(
    client: Client,
):

    result = await client.call_tool(
        "release_decision",
        {
            "pass_rate": 150,
            "critical_failures": 0,
        },
    )

    assert result.is_error is True
