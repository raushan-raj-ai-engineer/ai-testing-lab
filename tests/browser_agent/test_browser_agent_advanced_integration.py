from uuid import uuid4

import pytest
from mcp import Client

from src.browser_agent.advanced_quality import (
    build_advanced_quality_report,
    passes_advanced_quality_gate,
)
from src.browser_agent.browser_agent import (
    run_browser_agent,
)


@pytest.mark.anyio
async def test_real_advanced_agent_quality_gate():

    async with Client("http://localhost:8931/mcp") as client:
        # =================================================
        # 1. Open fresh TodoMVC
        # =================================================

        navigate_result = await client.call_tool(
            "browser_navigate",
            {"url": ("https://demo.playwright.dev/todomvc")},
        )

        assert not navigate_result.is_error

        # =================================================
        # 2. Unique task
        # =================================================

        todo_name = f"Advanced-{uuid4().hex[:6]}"

        goal = f"Add {todo_name} and mark it complete"

        print(
            "\nAdvanced goal:",
            goal,
        )

        # =================================================
        # 3. REAL autonomous browser agent
        # =================================================

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
            max_execution_retries=1,
        )

        # =================================================
        # 4. Build advanced quality report
        # =================================================

        report = build_advanced_quality_report(
            goal=goal,
            result=result,
        )

        # =================================================
        # 5. Print AI decision trajectory
        # =================================================

        print("\n==============================")

        print("AI DECISION TRACE")

        print("==============================")

        for attempt in result.attempts:
            print(
                "Step:",
                attempt.step_number,
                "| Attempt:",
                attempt.attempt_number,
                "| Tool:",
                attempt.tool_name,
                "| Accepted:",
                attempt.accepted,
            )

            if attempt.rejection_reason is not None:
                print(
                    "  Rejection:",
                    attempt.rejection_reason,
                )

        # =================================================
        # 6. Print execution trajectory
        # =================================================

        print("\n==============================")

        print("MCP EXECUTION TRACE")

        print("==============================")

        for execution in result.execution_attempts:
            print(
                "Step:",
                execution.step_number,
                "| Execution:",
                (execution.execution_attempt_number),
                "| Tool:",
                execution.tool_name,
                "| Success:",
                execution.succeeded,
            )

        # =================================================
        # 7. Quality report
        # =================================================

        print("\n==============================")

        print("ADVANCED AGENT QUALITY REPORT")

        print("==============================")

        print(
            "Completed:",
            report.completed,
        )

        print(
            "Progress:",
            report.progress_status,
        )

        print(
            "Progress score:",
            report.progress_score,
        )

        print(
            "Executed steps:",
            report.executed_steps,
        )

        print(
            "Decision attempts:",
            report.decision_attempts,
        )

        print(
            "Rejected decisions:",
            report.rejected_attempts,
        )

        print(
            "Decision efficiency:",
            round(
                report.decision_efficiency,
                3,
            ),
        )

        print(
            "Execution attempts:",
            report.execution_attempts,
        )

        print(
            "Execution failures:",
            report.execution_failures,
        )

        print(
            "Duplicate decisions:",
            report.duplicate_decisions,
        )

        print(
            "Stuck loop:",
            report.stuck_loop,
        )

        # =================================================
        # 8. Deterministic business proof
        # =================================================

        assert result.completed is True

        assert todo_name in result.final_snapshot

        assert "[checked]" in result.final_snapshot

        # Optimal browser execution:
        #
        # browser_type
        # browser_click
        assert len(result.steps) == 2

        assert result.steps[0].tool_name == "browser_type"

        assert result.steps[1].tool_name == "browser_click"

        # =================================================
        # 9. Advanced release-style gate
        #
        # Local llama3.2 occasionally self-corrects,
        # therefore some rejected decisions are okay.
        #
        # But:
        #
        # task must complete
        # no stuck loop
        # progress must be 100%
        # decision quality cannot be terrible
        # execution reliability cannot collapse
        # =================================================

        assert (
            passes_advanced_quality_gate(
                report=report,
                min_decision_efficiency=0.30,
                max_execution_failures=1,
            )
            is True
        )
