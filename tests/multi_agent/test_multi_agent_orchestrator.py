import pytest

from src.browser_agent.browser_agent import (
    BrowserAgentRunResult,
)
from src.multi_agent.orchestrator import (
    run_multi_agent_system,
)
from src.multi_agent.quality import (
    build_multi_agent_quality_report,
)

pytestmark = [
    pytest.mark.multi_agent,
    pytest.mark.asyncio,
]


class FakeClient:
    pass


class FakeBrowserExecutor:
    async def execute(
        self,
        client,
        task,
    ):

        todo_name = task.payload["todo_name"]

        return BrowserAgentRunResult(
            completed=True,
            steps=[],
            final_snapshot=f"""
            - listitem:
              - checkbox "Toggle Todo" [checked] [ref=e21]
              - generic [ref=e22]: {todo_name}
            """,
            attempts=[],
            execution_attempts=[],
        )


class FailingBrowserExecutor:
    async def execute(
        self,
        client,
        task,
    ):

        raise RuntimeError("Browser unavailable")


async def test_multi_agent_orchestrator_completes():

    result = await run_multi_agent_system(
        client=FakeClient(),
        goal=("Add Buy milk and mark it complete"),
        browser_executor=(FakeBrowserExecutor()),
    )

    assert result.completed is True

    assert len(result.handoffs) == 2

    assert len(result.completed_task_ids) == 2

    report = build_multi_agent_quality_report(result)

    assert report.release_passed is True


async def test_browser_agent_failure_propagates_safely():

    result = await run_multi_agent_system(
        client=FakeClient(),
        goal=("Add Buy milk and mark it complete"),
        browser_executor=(FailingBrowserExecutor()),
    )

    assert result.completed is False

    assert result.browser_result is None

    assert len(result.errors) == 1

    assert "Browser agent failed" in result.errors[0]

    report = build_multi_agent_quality_report(result)

    assert report.release_passed is False
