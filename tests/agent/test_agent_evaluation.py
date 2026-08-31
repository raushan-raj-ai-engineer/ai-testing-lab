import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallParams,
)
from mcp import Client

from src.agent.tool_agent import run_agent
from src.evaluation_model import evaluation_model
from src.mcp_server.qa_server import mcp

# ============================================================
# PYTEST / ANYIO CONFIGURATION
# ============================================================


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


# ============================================================
# TEST 1
#
# REAL AGENT
#
# User asks:
# 95 passed out of 100
#
# Expected:
# calculate_pass_rate
#
# Arguments:
# passed = 95
# total = 100
#
# Output:
# 95.0
# ============================================================


@pytest.mark.anyio
async def test_agent_tool_correctness_pass_rate(
    client: Client,
):

    user_request = (
        "We executed 100 automated tests and 95 passed. What is the pass rate?"
    )

    # --------------------------------------------------------
    # Run real agent
    # --------------------------------------------------------

    result = await run_agent(
        user_request,
        client,
    )

    # --------------------------------------------------------
    # Type narrowing
    #
    # AgentRunResult.tool_name is:
    #
    # str | None
    #
    # But this test expects a tool call.
    # --------------------------------------------------------

    tool_name = result.tool_name

    assert tool_name is not None

    # --------------------------------------------------------
    # DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result.final_answer,
        tools_called=[
            ToolCall(
                name=tool_name,
                input_parameters=result.arguments,
                output=result.tool_result,
            )
        ],
        expected_tools=[
            ToolCall(
                name="calculate_pass_rate",
                input_parameters={
                    "passed": 95,
                    "total": 100,
                },
                output={
                    "result": 95.0,
                },
            )
        ],
    )

    # --------------------------------------------------------
    # Tool Correctness Metric
    #
    # Important:
    #
    # model=evaluation_model
    #
    # Otherwise DeepEval may try default OpenAI model
    # and ask for OPENAI_API_KEY.
    # --------------------------------------------------------

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        strict_mode=True,
        should_exact_match=True,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
            ToolCallParams.OUTPUT,
        ],
        model=evaluation_model,
        async_mode=False,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metric.measure(test_case)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== PASS RATE TOOL CORRECTNESS =====")

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

    print(
        "Metric score:",
        metric.score,
    )

    print(
        "Metric reason:",
        metric.reason,
    )

    # --------------------------------------------------------
    # Assertion
    # --------------------------------------------------------

    assert metric.score is not None

    assert metric.score == 1.0


# ============================================================
# TEST 2
#
# REAL AGENT
#
# User asks:
#
# pass rate = 98%
# critical failures = 0
#
# Expected:
# release_decision
#
# Expected result:
# RELEASE
# ============================================================


@pytest.mark.anyio
async def test_agent_tool_correctness_release(
    client: Client,
):

    user_request = (
        "The pass rate is 98 percent "
        "and there are zero critical failures. "
        "Can we release?"
    )

    # --------------------------------------------------------
    # Run real agent
    # --------------------------------------------------------

    result = await run_agent(
        user_request,
        client,
    )

    # --------------------------------------------------------
    # Type narrowing
    # --------------------------------------------------------

    tool_name = result.tool_name

    assert tool_name is not None

    # --------------------------------------------------------
    # DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result.final_answer,
        tools_called=[
            ToolCall(
                name=tool_name,
                input_parameters=result.arguments,
                output=result.tool_result,
            )
        ],
        expected_tools=[
            ToolCall(
                name="release_decision",
                input_parameters={
                    "pass_rate": 98.0,
                    "critical_failures": 0,
                },
                output={
                    "result": "RELEASE",
                },
            )
        ],
    )

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        strict_mode=True,
        should_exact_match=True,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
            ToolCallParams.OUTPUT,
        ],
        model=evaluation_model,
        async_mode=False,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metric.measure(test_case)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== RELEASE TOOL CORRECTNESS =====")

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

    print(
        "Metric score:",
        metric.score,
    )

    print(
        "Metric reason:",
        metric.reason,
    )

    # --------------------------------------------------------
    # Assertion
    # --------------------------------------------------------

    assert metric.score is not None

    assert metric.score == 1.0


# ============================================================
# TEST 3
#
# EVALUATOR CALIBRATION
#
# Correct tool name:
# calculate_pass_rate
#
# But WRONG arguments:
#
# expected passed = 95
# actual passed   = 90
#
# Expected:
# metric should detect failure
# ============================================================


def test_tool_metric_rejects_wrong_arguments():

    test_case = LLMTestCase(
        input=("95 tests passed out of 100. Calculate the pass rate."),
        actual_output=("The pass rate is 90 percent."),
        # ----------------------------------------------------
        # Wrong actual tool call
        # ----------------------------------------------------
        tools_called=[
            ToolCall(
                name="calculate_pass_rate",
                input_parameters={
                    "passed": 90,
                    "total": 100,
                },
                output={
                    "result": 90.0,
                },
            )
        ],
        # ----------------------------------------------------
        # Correct expected tool call
        # ----------------------------------------------------
        expected_tools=[
            ToolCall(
                name="calculate_pass_rate",
                input_parameters={
                    "passed": 95,
                    "total": 100,
                },
                output={
                    "result": 95.0,
                },
            )
        ],
    )

    # --------------------------------------------------------
    # No pass/fail threshold required here.
    #
    # We are calibrating the evaluator.
    # --------------------------------------------------------

    metric = ToolCorrectnessMetric(
        threshold=None,
        should_exact_match=True,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
            ToolCallParams.OUTPUT,
        ],
        model=evaluation_model,
        async_mode=False,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metric.measure(test_case)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== WRONG ARGUMENT CALIBRATION =====")

    print(
        "Score:",
        metric.score,
    )

    print(
        "Reason:",
        metric.reason,
    )

    # --------------------------------------------------------
    # Expected:
    #
    # Correct tool name alone should NOT produce perfect score.
    # --------------------------------------------------------

    assert metric.score is not None

    assert metric.score < 1.0


# ============================================================
# TEST 4
#
# EVALUATOR CALIBRATION
#
# User needs:
# calculate_pass_rate
#
# But actual tool:
# release_decision
#
# Expected:
# metric should reject wrong tool
# ============================================================


def test_tool_metric_rejects_wrong_tool():

    test_case = LLMTestCase(
        input=("95 tests passed out of 100. Calculate the pass rate."),
        actual_output=("Release allowed."),
        # ----------------------------------------------------
        # Wrong actual tool
        # ----------------------------------------------------
        tools_called=[
            ToolCall(
                name="release_decision",
                input_parameters={
                    "pass_rate": 95.0,
                    "critical_failures": 0,
                },
            )
        ],
        # ----------------------------------------------------
        # Correct expected tool
        # ----------------------------------------------------
        expected_tools=[
            ToolCall(
                name="calculate_pass_rate",
                input_parameters={
                    "passed": 95,
                    "total": 100,
                },
            )
        ],
    )

    metric = ToolCorrectnessMetric(
        threshold=None,
        should_exact_match=True,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
        ],
        model=evaluation_model,
        async_mode=False,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metric.measure(test_case)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== WRONG TOOL CALIBRATION =====")

    print(
        "Score:",
        metric.score,
    )

    print(
        "Reason:",
        metric.reason,
    )

    # --------------------------------------------------------
    # Assertion
    # --------------------------------------------------------

    assert metric.score is not None

    assert metric.score < 1.0
