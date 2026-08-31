from uuid import uuid4

import pytest
from mcp import Client

from src.guardrails.quality import (
    build_runtime_guardrail_report,
)
from src.guardrails.safe_orchestrator import (
    run_guarded_multi_agent_system,
)

pytestmark = [
    pytest.mark.guardrail_live,
    pytest.mark.asyncio,
]


async def test_real_guarded_multi_agent_run():

    todo_name = "Session18-" + uuid4().hex[:6]

    goal = f"Add {todo_name} and mark it complete"

    async with Client("http://localhost:8931/mcp") as client:
        result = await run_guarded_multi_agent_system(
            client=client,
            goal=goal,
        )

    report = build_runtime_guardrail_report(result)

    print("\n================================")

    print("SESSION 18 SAFETY REPORT")

    print("================================")

    print(
        "Input allowed:",
        report.input_allowed,
    )

    print(
        "Handoffs safe:",
        report.handoffs_safe,
    )

    print(
        "Tools safe:",
        report.tools_safe,
    )

    print(
        "Multi-agent completed:",
        report.multi_agent_completed,
    )

    print(
        "Multi-agent quality:",
        report.multi_agent_quality_passed,
    )

    print(
        "Safety findings:",
        report.total_findings,
    )

    print(
        "Safety release:",
        report.safety_release_passed,
    )

    assert result.multi_agent_result is not None

    assert result.multi_agent_result.browser_result is not None

    assert todo_name in result.multi_agent_result.browser_result.final_snapshot

    assert report.total_findings == 0

    assert report.safety_release_passed is True
