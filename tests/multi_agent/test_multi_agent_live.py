from uuid import uuid4

import pytest
from mcp import Client

from src.multi_agent.orchestrator import (
    run_multi_agent_system,
)
from src.multi_agent.quality import (
    build_multi_agent_quality_report,
    passes_multi_agent_quality_gate,
)

pytestmark = [
    pytest.mark.multi_agent_live,
    pytest.mark.asyncio,
]


async def test_real_multi_agent_browser_handoff():

    unique_name = "Session17-" + uuid4().hex[:6]

    goal = f"Add {unique_name} and mark it complete"

    async with Client("http://localhost:8931/mcp") as client:
        result = await run_multi_agent_system(
            client=client,
            goal=goal,
        )

    report = build_multi_agent_quality_report(result)

    print("\n=================================")

    print("SESSION 17 MULTI-AGENT REPORT")

    print("=================================")

    print(
        "Goal:",
        goal,
    )

    print(
        "Completed:",
        report.completed,
    )

    print(
        "Tasks:",
        f"{report.completed_tasks}/{report.total_tasks}",
    )

    print(
        "Task completion rate:",
        report.task_completion_rate,
    )

    print(
        "Handoffs:",
        f"{report.actual_handoffs}/{report.expected_handoffs}",
    )

    print(
        "Handoff success rate:",
        report.handoff_success_rate,
    )

    print(
        "Context preserved:",
        report.context_preserved,
    )

    print(
        "Routing errors:",
        report.routing_errors,
    )

    print(
        "Execution errors:",
        report.execution_errors,
    )

    print(
        "Handoff loop:",
        report.handoff_loop,
    )

    print(
        "Browser quality passed:",
        report.browser_quality_passed,
    )

    print(
        "Release passed:",
        report.release_passed,
    )

    assert result.browser_result is not None

    assert unique_name in result.browser_result.final_snapshot

    assert result.browser_result.completed is True

    assert report.context_preserved is True

    assert report.handoff_loop is False

    assert report.routing_errors == 0

    assert passes_multi_agent_quality_gate(report) is True
