import asyncio
from uuid import uuid4

import pytest
from deepeval.metrics import (
    MCPTaskCompletionMetric,
    MCPUseMetric,
    ToolCorrectnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MCPServer,
    MCPToolCall,
    ToolCall,
    ToolCallParams,
    Turn,
)
from mcp import Client
from mcp.types import CallToolResult

from src.browser_agent.browser_agent import (
    run_browser_agent,
)

# ---------------------------------------------------------
# Local DeepEval evaluation model
#
# Important:
# DeepEval otherwise tries to initialize OpenAI.
# ---------------------------------------------------------

evaluation_model = OllamaModel(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0,
)


def build_correct_tools():

    return [
        ToolCall(
            name="browser_type",
            input_parameters={
                "target": "e8",
                "text": "Buy milk",
                "submit": True,
            },
        ),
        ToolCall(
            name="browser_click",
            input_parameters={
                "target": "e21",
                "button": "left",
            },
        ),
    ]


def test_tool_trajectory_is_correct():

    actual_tools = build_correct_tools()
    expected_tools = build_correct_tools()

    test_case = LLMTestCase(
        input="Add Buy milk and mark it complete",
        actual_output="Buy milk added and completed.",
        tools_called=actual_tools,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        model=evaluation_model,
        include_reason=False,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
        ],
        should_exact_match=True,
    )

    metric.measure(test_case)

    print(
        "\nTool correctness score:",
        metric.score,
    )

    assert metric.score is not None
    assert metric.score == 1.0


def test_wrong_tool_order_fails():

    expected_tools = build_correct_tools()

    actual_tools = [
        expected_tools[1],
        expected_tools[0],
    ]

    test_case = LLMTestCase(
        input="Add Buy milk and mark it complete",
        actual_output="Task attempted.",
        tools_called=actual_tools,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        model=evaluation_model,
        include_reason=False,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
        ],
        should_exact_match=True,
    )

    metric.measure(test_case)

    print(
        "\nWrong order score:",
        metric.score,
    )

    assert metric.score is not None
    assert metric.score < 1.0


def test_wrong_tool_argument_fails():

    expected_tools = build_correct_tools()

    actual_tools = [
        ToolCall(
            name="browser_type",
            input_parameters={
                "target": "e8",
                "text": "Buy milk",
                "submit": False,
            },
        ),
        ToolCall(
            name="browser_click",
            input_parameters={
                "target": "e21",
                "button": "left",
            },
        ),
    ]

    test_case = LLMTestCase(
        input="Add Buy milk and mark it complete",
        actual_output="Task attempted.",
        tools_called=actual_tools,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        model=evaluation_model,
        include_reason=False,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
        ],
        should_exact_match=True,
    )

    metric.measure(test_case)

    print(
        "\nWrong argument score:",
        metric.score,
    )

    assert metric.score is not None
    assert metric.score < 1.0


@pytest.mark.anyio
async def test_real_browser_agent_trajectory_with_deepeval():

    async with Client("http://localhost:8931/mcp") as client:
        # =========================================
        # 1. Navigate
        # =========================================

        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        # =========================================
        # 2. Unique todo
        # =========================================

        todo_name = f"DeepEval-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        # =========================================
        # 3. REAL autonomous agent
        # =========================================

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        # =========================================
        # 4. Deterministic business proof
        # =========================================

        assert result.completed is True

        assert todo_name in result.final_snapshot

        assert "[checked]" in result.final_snapshot

        assert len(result.steps) == 2

        # =========================================
        # 5. NAME-ONLY actual trajectory
        #
        # IMPORTANT:
        #
        # This test evaluates TOOL TRAJECTORY,
        # not dynamic browser arguments.
        # =========================================

        actual_tools = [
            ToolCall(
                name=step.tool_name,
                input_parameters=None,
            )
            for step in result.steps
        ]

        # =========================================
        # 6. Expected trajectory
        # =========================================

        expected_tools = [
            ToolCall(
                name="browser_type",
                input_parameters=None,
            ),
            ToolCall(
                name="browser_click",
                input_parameters=None,
            ),
        ]

        test_case = LLMTestCase(
            input=goal,
            actual_output=(f"{todo_name} was added and marked complete."),
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )

        # =========================================
        # 7. Tool trajectory metric
        #
        # NO INPUT_PARAMETERS comparison here.
        # =========================================

        metric = ToolCorrectnessMetric(
            threshold=1.0,
            # Prevent DeepEval 4.2 from
            # defaulting to OpenAI.
            model=evaluation_model,
            include_reason=False,
            # Exact:
            #
            # browser_type
            # browser_click
            #
            # in this exact trajectory
            should_exact_match=True,
        )

        metric.measure(test_case)

        # =========================================
        # 8. Debug output
        # =========================================

        print("\nReal agent steps:")

        for step in result.steps:
            print(
                step.step_number,
                step.tool_name,
                step.arguments,
            )

        print("\nDeepEval actual tools:")

        for tool in actual_tools:
            print(tool)

        print("\nDeepEval expected tools:")

        for tool in expected_tools:
            print(tool)

        print(
            "\nDeepEval trajectory score:",
            metric.score,
        )

        # =========================================
        # 9. Quality gate
        # =========================================

        assert metric.score is not None

        assert metric.score == 1.0


@pytest.mark.anyio
async def test_real_browser_agent_argument_correctness():

    # -----------------------------------------------------
    # 1. Connect to real Playwright MCP server
    # -----------------------------------------------------

    async with Client("http://localhost:8931/mcp") as client:
        # -------------------------------------------------
        # 2. Open TodoMVC
        # -------------------------------------------------

        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        # -------------------------------------------------
        # 3. Unique todo name
        #
        # This prevents collision between test executions.
        # -------------------------------------------------

        todo_name = f"ArgumentEval-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        print(
            "\nGoal:",
            goal,
        )

        # -------------------------------------------------
        # 4. Execute REAL autonomous browser agent
        # -------------------------------------------------

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        # -------------------------------------------------
        # 5. Basic task assertions
        # -------------------------------------------------

        assert result.completed is True

        assert len(result.steps) == 2

        assert todo_name in result.final_snapshot

        assert "[checked]" in result.final_snapshot

        # -------------------------------------------------
        # 6. Convert actual agent trajectory
        #    into DeepEval ToolCalls
        #
        # Important:
        #
        # We deliberately DON'T compare dynamic Playwright
        # refs like:
        #
        # target=e8
        # target=e21
        #
        # because those refs change at runtime.
        #
        # We also ignore optional browser arguments such as:
        #
        # doubleClick
        # modifiers
        #
        # because they are not important for the semantic
        # correctness of this task.
        # -------------------------------------------------

        actual_tools: list[ToolCall] = []

        for step in result.steps:
            # ---------------------------------------------
            # browser_type
            # ---------------------------------------------

            if step.tool_name == "browser_type":
                actual_tools.append(
                    ToolCall(
                        name="browser_type",
                        input_parameters={
                            "text": (step.arguments.get("text")),
                            "submit": (step.arguments.get("submit")),
                        },
                    )
                )

            # ---------------------------------------------
            # browser_click
            # ---------------------------------------------

            elif step.tool_name == "browser_click":
                actual_tools.append(
                    ToolCall(
                        name="browser_click",
                        input_parameters={
                            "button": (
                                step.arguments.get(
                                    "button",
                                    "left",
                                )
                            ),
                        },
                    )
                )

            # ---------------------------------------------
            # Unexpected tool
            #
            # Keep it in actual_tools so DeepEval can
            # correctly fail the trajectory.
            # ---------------------------------------------

            else:
                actual_tools.append(
                    ToolCall(
                        name=step.tool_name,
                        input_parameters={},
                    )
                )

        # -------------------------------------------------
        # 7. Define expected semantic tool arguments
        #
        # No dynamic target refs here.
        # -------------------------------------------------

        expected_tools = [
            ToolCall(
                name="browser_type",
                input_parameters={
                    "text": todo_name,
                    "submit": True,
                },
            ),
            ToolCall(
                name="browser_click",
                input_parameters={
                    "button": "left",
                },
            ),
        ]

        # -------------------------------------------------
        # 8. Print actual trajectory
        # -------------------------------------------------

        print("\nActual browser-agent steps:")

        for step in result.steps:
            print(
                f"Step {step.step_number}:",
                step.tool_name,
                step.arguments,
            )

        # -------------------------------------------------
        # 9. Print semantic tools
        # -------------------------------------------------

        print("\nActual semantic ToolCalls:")

        for tool in actual_tools:
            print(tool)

        print("\nExpected semantic ToolCalls:")

        for tool in expected_tools:
            print(tool)

        # -------------------------------------------------
        # 10. DeepEval test case
        # -------------------------------------------------

        test_case = LLMTestCase(
            input=goal,
            actual_output=(f"{todo_name} was added and marked complete."),
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )

        # -------------------------------------------------
        # 11. Deterministic argument correctness
        #
        # model=evaluation_model is IMPORTANT.
        #
        # Without this, your DeepEval 4.2.0 environment
        # attempts to initialize OpenAI and gives:
        #
        # OPENAI_API_KEY is not configured
        #
        # include_reason=False means we don't need
        # LLM-generated reasoning for this test.
        #
        # INPUT_PARAMETERS means tool arguments are also
        # compared.
        # -------------------------------------------------

        metric = ToolCorrectnessMetric(
            threshold=1.0,
            model=evaluation_model,
            include_reason=False,
            evaluation_params=[
                ToolCallParams.INPUT_PARAMETERS,
            ],
            should_exact_match=True,
        )

        # -------------------------------------------------
        # 12. Evaluate
        # -------------------------------------------------

        metric.measure(test_case)

        print(
            "\nArgument correctness score:",
            metric.score,
        )

        # -------------------------------------------------
        # 13. Final quality gate
        # -------------------------------------------------

        assert metric.score is not None

        assert metric.score == 1.0


@pytest.mark.anyio
async def test_real_browser_agent_creates_valid_mcp_trace():

    async with Client("http://localhost:8931/mcp") as client:
        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        todo_name = f"MCPTrace-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        assert result.completed is True
        assert len(result.steps) == 2

        # -----------------------------------------
        # Convert REAL agent executions
        # into DeepEval MCPToolCall objects
        # -----------------------------------------

        mcp_tools_called = []

        for step in result.steps:
            assert step.execution_result is not None

            assert isinstance(
                step.execution_result,
                CallToolResult,
            )

            mcp_tools_called.append(
                MCPToolCall(
                    name=step.tool_name,
                    args=step.arguments,
                    result=step.execution_result,
                )
            )

        print("\nMCP calls captured:")

        for call in mcp_tools_called:
            print(
                call.name,
                call.args,
            )

        # -----------------------------------------
        # Deterministic assertions
        # -----------------------------------------

        assert len(mcp_tools_called) == 2

        assert mcp_tools_called[0].name == "browser_type"

        assert mcp_tools_called[1].name == "browser_click"

        assert mcp_tools_called[0].args["text"] == todo_name

        assert mcp_tools_called[0].args["submit"] is True

        assert mcp_tools_called[1].args["button"] == "left"


@pytest.mark.anyio
async def test_real_browser_agent_mcp_use():

    async with Client("http://localhost:8931/mcp") as client:
        # -------------------------------------------------
        # 1. Open TodoMVC
        # -------------------------------------------------

        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        # -------------------------------------------------
        # 2. Get REAL MCP tool definitions
        # -------------------------------------------------

        tools_result = await client.list_tools()

        allowed_tool_names = {
            "browser_type",
            "browser_click",
        }

        available_tools = [
            tool for tool in tools_result.tools if tool.name in allowed_tool_names
        ]

        assert len(available_tools) == 2

        # -------------------------------------------------
        # 3. Define DeepEval MCPServer
        # -------------------------------------------------

        playwright_mcp_server = MCPServer(
            server_name="Playwright Browser Agent",
            available_tools=available_tools,
        )

        # -------------------------------------------------
        # 4. Create unique autonomous goal
        # -------------------------------------------------

        todo_name = f"MCPUse-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        print(
            "\nMCP evaluation goal:",
            goal,
        )

        # -------------------------------------------------
        # 5. Execute REAL browser agent
        # -------------------------------------------------

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        assert result.completed is True
        assert len(result.steps) == 2

        # -------------------------------------------------
        # 6. Convert actual executions into MCPToolCall
        # -------------------------------------------------

        mcp_tools_called: list[MCPToolCall] = []

        for step in result.steps:
            assert step.execution_result is not None

            mcp_tools_called.append(
                MCPToolCall(
                    name=step.tool_name,
                    args=step.arguments,
                    result=step.execution_result,
                )
            )

        # -------------------------------------------------
        # 7. DeepEval MCP test case
        # -------------------------------------------------

        test_case = LLMTestCase(
            input=goal,
            actual_output=(f"{todo_name} was added and marked complete."),
            mcp_servers=[
                playwright_mcp_server,
            ],
            mcp_tools_called=(mcp_tools_called),
        )

        # -------------------------------------------------
        # 8. MCP Use metric
        #
        # IMPORTANT:
        # model=evaluation_model prevents OpenAI default.
        #
        # include_reason=False keeps local evaluation
        # smaller/faster.
        # -------------------------------------------------

        metric = MCPUseMetric(
            threshold=0.7,
            model=evaluation_model,
            include_reason=False,
            verbose_mode=False,
            async_mode=False,
        )

        # -------------------------------------------------
        # 9. Run LLM judge OUTSIDE main async event loop
        #
        # Earlier we learned that long synchronous Ollama
        # work inside MCP async session can disturb MCP.
        # -------------------------------------------------

        await asyncio.to_thread(
            metric.measure,
            test_case,
        )

        # -------------------------------------------------
        # 10. Print captured MCP trajectory
        # -------------------------------------------------

        print("\nAvailable MCP tools:")

        for tool in available_tools:
            print(
                "-",
                tool.name,
            )

        print("\nMCP tools actually called:")

        for call in mcp_tools_called:
            print(
                call.name,
                call.args,
            )

        print(
            "\nMCP Use score:",
            metric.score,
        )

        # -------------------------------------------------
        # 11. Quality gate
        # -------------------------------------------------

        assert metric.score is not None

        assert metric.score >= 0.7


@pytest.mark.anyio
async def test_real_browser_agent_mcp_task_completion():

    async with Client("http://localhost:8931/mcp") as client:
        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        todo_name = f"MCPComplete-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        # -----------------------------------------
        # REAL quality gate:
        # deterministic browser evidence
        # -----------------------------------------

        assert result.completed is True
        assert todo_name in result.final_snapshot
        assert "[checked]" in result.final_snapshot

        assert len(result.steps) == 2

        assert result.steps[0].tool_name == "browser_type"

        assert result.steps[1].tool_name == "browser_click"

        # -----------------------------------------
        # Diagnostic MCPTaskCompletionMetric
        #
        # Local llama3.2 is NOT used as a hard gate.
        # -----------------------------------------

        tools_result = await client.list_tools()

        allowed_tool_names = {
            "browser_type",
            "browser_click",
        }

        available_tools = [
            tool for tool in tools_result.tools if tool.name in allowed_tool_names
        ]

        playwright_server = MCPServer(
            server_name="Playwright Browser Agent",
            available_tools=available_tools,
        )

        mcp_tools_called = []

        for step in result.steps:
            assert step.execution_result is not None

            mcp_tools_called.append(
                MCPToolCall(
                    name=step.tool_name,
                    args=step.arguments,
                    result=step.execution_result,
                )
            )

        conversation = ConversationalTestCase(
            turns=[
                Turn(
                    role="user",
                    content=goal,
                ),
                Turn(
                    role="assistant",
                    content=(
                        f"Task completed. The todo '{todo_name}' exists and is checked."
                    ),
                    mcp_tools_called=mcp_tools_called,
                ),
            ],
            mcp_servers=[
                playwright_server,
            ],
        )

        metric = MCPTaskCompletionMetric(
            threshold=None,
            model=evaluation_model,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        )

        await asyncio.to_thread(
            metric.measure,
            conversation,
        )

        print(
            "\nLocal MCP Task Completion score:",
            metric.score,
        )

        print(
            "\nLocal evaluator reason:",
            metric.reason,
        )

        # IMPORTANT:
        # No assert on metric.score.
        #
        # llama3.2 has already demonstrated
        # a false-negative on a deterministically
        # completed browser task.


@pytest.mark.anyio
async def test_real_browser_agent_step_efficiency():

    async with Client("http://localhost:8931/mcp") as client:
        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        todo_name = f"Efficiency-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        assert result.completed is True

        # Optimal trajectory:
        #
        # Step 1 = browser_type
        # Step 2 = browser_click

        optimal_steps = 2
        actual_steps = len(result.steps)

        efficiency_score = optimal_steps / actual_steps if actual_steps > 0 else 0.0

        print(
            "\nActual steps:",
            actual_steps,
        )

        print(
            "Optimal steps:",
            optimal_steps,
        )

        print(
            "Step efficiency:",
            efficiency_score,
        )

        assert actual_steps == 2
        assert efficiency_score == 1.0


@pytest.mark.anyio
async def test_browser_agent_quality_gate():

    async with Client("http://localhost:8931/mcp") as client:
        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        todo_name = f"QualityGate-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        # -----------------------------------------
        # REAL AGENT EXECUTION
        # -----------------------------------------

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        # =========================================
        # 1. TASK COMPLETION
        # =========================================

        task_completion = (
            result.completed
            and todo_name in result.final_snapshot
            and "[checked]" in result.final_snapshot
        )

        # =========================================
        # 2. TOOL SELECTION
        # =========================================

        correct_tools = (
            len(result.steps) == 2
            and result.steps[0].tool_name == "browser_type"
            and result.steps[1].tool_name == "browser_click"
        )

        # =========================================
        # 3. ARGUMENT CORRECTNESS
        # =========================================

        correct_arguments = (
            result.steps[0].arguments.get("text") == todo_name
            and result.steps[0].arguments.get("submit") is True
            and result.steps[1].arguments.get("button", "left") == "left"
        )

        # =========================================
        # 4. STEP EFFICIENCY
        # =========================================

        optimal_steps = 2
        actual_steps = len(result.steps)

        step_efficiency = optimal_steps / actual_steps if actual_steps > 0 else 0.0

        # =========================================
        # 5. MCP EXECUTION SUCCESS
        # =========================================

        mcp_execution_success = all(
            step.execution_result is not None and not step.execution_result.is_error
            for step in result.steps
        )

        # =========================================
        # 6. QUALITY REPORT
        # =========================================

        print("\n==============================")

        print("BROWSER AGENT QUALITY REPORT")

        print("==============================")

        print(
            "Task completion:",
            task_completion,
        )

        print(
            "Correct tools:",
            correct_tools,
        )

        print(
            "Correct arguments:",
            correct_arguments,
        )

        print(
            "Step efficiency:",
            step_efficiency,
        )

        print(
            "MCP execution:",
            mcp_execution_success,
        )

        # =========================================
        # 7. QUALITY GATES
        # =========================================

        assert task_completion is True

        assert correct_tools is True

        assert correct_arguments is True

        assert step_efficiency >= 1.0

        assert mcp_execution_success is True
