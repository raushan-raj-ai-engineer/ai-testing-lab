from __future__ import annotations

from typing import Any

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
    build_advanced_quality_report,
)
from src.browser_agent.browser_agent import (
    BrowserAgentRunResult,
    run_browser_agent,
)
from src.multi_agent.models import (
    AgentTask,
)

TODO_URL = "https://demo.playwright.dev/todomvc"


class BrowserAgentExecutor:
    """
    Adapter around the autonomous browser agent
    built in Sessions 13-16.
    """

    def __init__(
        self,
        navigate_url: str | None = (TODO_URL),
    ) -> None:

        self.navigate_url = navigate_url

    async def execute(
        self,
        client: Any,
        task: AgentTask,
    ) -> BrowserAgentRunResult:

        goal = task.payload.get("goal")

        if (
            not isinstance(
                goal,
                str,
            )
            or not goal.strip()
        ):
            raise ValueError("Browser task requires goal")

        # =================================================
        # Fresh browser state
        # =================================================

        if self.navigate_url:
            result = await client.call_tool(
                "browser_navigate",
                {"url": (self.navigate_url)},
            )

            if result.is_error:
                raise RuntimeError("Browser agent could not navigate to TodoMVC")

        # =================================================
        # Existing autonomous browser agent
        # =================================================

        return await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=int(
                task.payload.get(
                    "max_steps",
                    5,
                )
            ),
            max_execution_retries=int(
                task.payload.get(
                    "max_execution_retries",
                    1,
                )
            ),
        )


class QualityAgentExecutor:
    """
    Agent responsible for evaluating the output
    produced by the browser agent.

    This is deterministic because deterministic
    evidence is preferred for release gating.
    """

    async def execute(
        self,
        task: AgentTask,
        browser_result: BrowserAgentRunResult,
    ) -> AdvancedAgentQualityReport:

        goal = task.payload.get("goal")

        if (
            not isinstance(
                goal,
                str,
            )
            or not goal.strip()
        ):
            raise ValueError("Quality task requires goal")

        return build_advanced_quality_report(
            goal=goal,
            result=browser_result,
        )
