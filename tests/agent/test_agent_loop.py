import pytest
from mcp import Client

from src.agent.tool_agent import (
    run_agent,
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
async def test_agent_calculates_pass_rate_end_to_end(
    client: Client,
):

    result = await run_agent(
        ("We executed 100 automated tests and 95 passed. What is the pass rate?"),
        client,
    )

    print("\n===== FULL AGENT RESULT =====")

    print(
        "Tool:",
        result.tool_name,
    )

    print(
        "Arguments:",
        result.arguments,
    )

    print(
        "Tool result:",
        result.tool_result,
    )

    print(
        "Final answer:",
        result.final_answer,
    )

    # ---------------------------------------------
    # Agent decision
    # ---------------------------------------------

    assert result.tool_name == "calculate_pass_rate"

    # ---------------------------------------------
    # Agent arguments
    # ---------------------------------------------

    assert result.arguments == {
        "passed": 95,
        "total": 100,
    }

    # ---------------------------------------------
    # MCP execution
    # ---------------------------------------------

    assert result.tool_result == {"result": 95.0}

    # ---------------------------------------------
    # Final answer
    # ---------------------------------------------

    assert "95" in result.final_answer


@pytest.mark.anyio
async def test_agent_makes_release_decision_end_to_end(
    client: Client,
):

    result = await run_agent(
        (
            "The test pass rate is 98 percent "
            "and there are zero critical failures. "
            "Can we release?"
        ),
        client,
    )

    print("\n===== RELEASE AGENT RESULT =====")

    print(
        "Tool:",
        result.tool_name,
    )

    print(
        "Arguments:",
        result.arguments,
    )

    print(
        "Tool result:",
        result.tool_result,
    )

    print(
        "Final answer:",
        result.final_answer,
    )

    assert result.tool_name == "release_decision"

    assert result.arguments == {
        "pass_rate": 98.0,
        "critical_failures": 0,
    }

    assert result.tool_result == {"result": "RELEASE"}

    assert "release" in result.final_answer.lower()


@pytest.mark.anyio
async def test_agent_blocks_release_when_critical_failure_exists(
    client: Client,
):

    result = await run_agent(
        (
            "Our test pass rate is 100 percent, "
            "but we have 1 critical failure. "
            "Can we deploy?"
        ),
        client,
    )

    assert result.tool_name == "release_decision"

    assert result.arguments == {
        "pass_rate": 100.0,
        "critical_failures": 1,
    }

    assert result.tool_result == {"result": "BLOCK"}

    final_answer = result.final_answer.lower()

    assert (
        "block" in final_answer
        or "cannot" in final_answer
        or "should not" in final_answer
    )


@pytest.mark.anyio
async def test_agent_answers_explanation_without_tool(
    client: Client,
):

    result = await run_agent(
        ("Explain what test pass rate means in simple words."),
        client,
    )

    print("\n===== NO TOOL RESULT =====")

    print(result.final_answer)

    assert result.tool_name is None

    assert result.arguments == {}

    assert result.tool_result is None

    assert len(result.final_answer.strip()) > 0
