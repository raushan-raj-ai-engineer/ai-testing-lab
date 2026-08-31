import pytest
from mcp import Client

from src.agent.tool_agent import (
    choose_tool,
    get_ollama_tools,
    normalize_arguments,
)
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
async def test_agent_selects_pass_rate_tool(
    client: Client,
):

    tools = await get_ollama_tools(client)

    decision = choose_tool(
        ("Calculate the test pass rate when 95 tests passed out of 100."),
        tools,
    )

    print(
        "\nAgent decision:",
        decision,
    )

    assert decision.tool_name == "calculate_pass_rate"

    assert decision.arguments["passed"] == 95

    assert decision.arguments["total"] == 100


@pytest.mark.anyio
async def test_agent_selects_release_decision_tool(
    client: Client,
):

    tools = await get_ollama_tools(client)

    decision = choose_tool(
        (
            "Can we release the application "
            "if the test pass rate is 98 percent "
            "and there are zero critical failures?"
        ),
        tools,
    )

    print(
        "\nAgent decision:",
        decision,
    )

    assert decision.tool_name == "release_decision"

    assert decision.arguments["pass_rate"] == 98

    assert decision.arguments["critical_failures"] == 0


@pytest.mark.anyio
async def test_agent_does_not_call_tool_for_explanation(
    client: Client,
):

    tools = await get_ollama_tools(client)

    decision = choose_tool(
        ("Explain in simple words what test pass rate means."),
        tools,
    )

    print(
        "\nNo-tool decision:",
        decision,
    )

    assert decision.tool_name is None

    assert decision.arguments == {}


@pytest.mark.anyio
async def test_agent_selects_pass_rate_tool_for_paraphrase(
    client: Client,
):

    tools = await get_ollama_tools(client)

    decision = choose_tool(
        ("Out of 80 automated tests, 72 are green. What's the pass percentage?"),
        tools,
    )

    print(
        "\nParaphrase decision:",
        decision,
    )

    assert decision.tool_name == "calculate_pass_rate"

    assert decision.arguments["passed"] == 72

    assert decision.arguments["total"] == 80


@pytest.mark.anyio
async def test_agent_selects_release_tool_for_paraphrase(
    client: Client,
):

    tools = await get_ollama_tools(client)

    decision = choose_tool(
        (
            "Our suite is at 97.5 percent "
            "and there are no critical failures. "
            "Should deployment go ahead?"
        ),
        tools,
    )

    print(
        "\nRelease paraphrase:",
        decision,
    )

    assert decision.tool_name == "release_decision"

    assert decision.arguments["pass_rate"] == 97.5

    assert decision.arguments["critical_failures"] == 0


@pytest.mark.anyio
async def test_missing_required_argument_is_rejected(
    client: Client,
):

    tools = await get_ollama_tools(client)

    with pytest.raises(
        ValueError,
        match="Missing required arguments",
    ):
        normalize_arguments(
            tool_name="calculate_pass_rate",
            arguments={
                "passed": "95",
            },
            tools=tools,
        )


@pytest.mark.anyio
async def test_out_of_range_argument_is_rejected(
    client: Client,
):

    tools = await get_ollama_tools(client)

    with pytest.raises(
        ValueError,
        match="<= 100",
    ):
        normalize_arguments(
            tool_name="release_decision",
            arguments={
                "pass_rate": "150",
                "critical_failures": "0",
            },
            tools=tools,
        )


@pytest.mark.anyio
async def test_unknown_argument_is_rejected(
    client: Client,
):

    tools = await get_ollama_tools(client)

    with pytest.raises(
        ValueError,
        match="Unknown argument",
    ):
        normalize_arguments(
            tool_name="release_decision",
            arguments={
                "pass_rate": "98",
                "critical_failures": "0",
                "environment": "production",
            },
            tools=tools,
        )
