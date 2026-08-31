from __future__ import annotations

from typing import Any, Protocol

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
    passes_advanced_quality_gate,
)
from src.browser_agent.browser_agent import (
    BrowserAgentRunResult,
)
from src.multi_agent.executors import (
    BrowserAgentExecutor,
    QualityAgentExecutor,
)
from src.multi_agent.models import (
    BROWSER_AGENT,
    BROWSER_TASK,
    PLANNER_AGENT,
    QUALITY_TASK,
    AgentTask,
    MultiAgentRunResult,
)
from src.multi_agent.planner import (
    build_multi_agent_plan,
    find_task,
)
from src.multi_agent.router import (
    build_handoff,
    validate_task_dependencies,
)


class BrowserExecutorProtocol(Protocol):
    async def execute(
        self,
        client: Any,
        task: AgentTask,
    ) -> BrowserAgentRunResult: ...


class QualityExecutorProtocol(Protocol):
    async def execute(
        self,
        task: AgentTask,
        browser_result: BrowserAgentRunResult,
    ) -> AdvancedAgentQualityReport: ...


async def run_multi_agent_system(
    client: Any,
    goal: str,
    browser_executor: (BrowserExecutorProtocol | None) = None,
    quality_executor: (QualityExecutorProtocol | None) = None,
) -> MultiAgentRunResult:
    """
    Execute:

        Planner
          ↓
        Browser Agent
          ↓
        Quality Agent

    Agent failures are captured in the result rather
    than converted into a false PASS.
    """

    browser_executor = browser_executor or BrowserAgentExecutor()

    quality_executor = quality_executor or QualityAgentExecutor()

    # =====================================================
    # PLANNER
    # =====================================================

    plan = build_multi_agent_plan(goal)

    browser_task = find_task(
        plan,
        BROWSER_TASK,
    )

    quality_task = find_task(
        plan,
        QUALITY_TASK,
    )

    handoffs = []

    completed_task_ids: list[str] = []

    routing_errors: list[str] = []

    errors: list[str] = []

    browser_result = None

    browser_quality_report = None

    correlation_id = f"correlation-{plan.id}"

    base_context = {
        "plan_id": plan.id,
        "original_goal": goal,
        "todo_name": plan.todo_name,
    }

    # =====================================================
    # PLANNER -> BROWSER AGENT
    # =====================================================

    try:
        validate_task_dependencies(
            task=browser_task,
            completed_task_ids=(completed_task_ids),
        )

        browser_handoff = build_handoff(
            correlation_id=(correlation_id),
            source_agent=(PLANNER_AGENT),
            task=browser_task,
            payload=(browser_task.payload),
            context=(base_context),
        )

        handoffs.append(browser_handoff)

    except ValueError as exc:
        routing_errors.append(str(exc))

        return MultiAgentRunResult(
            goal=goal,
            plan=plan,
            handoffs=handoffs,
            completed_task_ids=(completed_task_ids),
            browser_result=None,
            browser_quality_report=None,
            completed=False,
            routing_errors=(routing_errors),
            errors=errors,
        )

    # =====================================================
    # BROWSER AGENT EXECUTION
    # =====================================================

    try:
        browser_result = await browser_executor.execute(
            client=client,
            task=browser_task,
        )

        if browser_result.completed:
            completed_task_ids.append(browser_task.id)

    except Exception as exc:
        errors.append(f"Browser agent failed: {type(exc).__name__}: {exc}")

        return MultiAgentRunResult(
            goal=goal,
            plan=plan,
            handoffs=handoffs,
            completed_task_ids=(completed_task_ids),
            browser_result=None,
            browser_quality_report=None,
            completed=False,
            routing_errors=(routing_errors),
            errors=errors,
        )

    # =====================================================
    # BROWSER AGENT -> QUALITY AGENT
    # =====================================================

    try:
        validate_task_dependencies(
            task=quality_task,
            completed_task_ids=(completed_task_ids),
        )

        quality_payload = dict(quality_task.payload)

        quality_payload["browser_result"] = browser_result

        quality_handoff = build_handoff(
            correlation_id=(correlation_id),
            source_agent=(BROWSER_AGENT),
            task=quality_task,
            payload=(quality_payload),
            context=(base_context),
        )

        handoffs.append(quality_handoff)

    except ValueError as exc:
        routing_errors.append(str(exc))

        return MultiAgentRunResult(
            goal=goal,
            plan=plan,
            handoffs=handoffs,
            completed_task_ids=(completed_task_ids),
            browser_result=(browser_result),
            browser_quality_report=None,
            completed=False,
            routing_errors=(routing_errors),
            errors=errors,
        )

    # =====================================================
    # QUALITY AGENT EXECUTION
    # =====================================================

    try:
        browser_quality_report = await quality_executor.execute(
            task=quality_task,
            browser_result=(browser_result),
        )

        completed_task_ids.append(quality_task.id)

    except Exception as exc:
        errors.append(f"Quality agent failed: {type(exc).__name__}: {exc}")

        return MultiAgentRunResult(
            goal=goal,
            plan=plan,
            handoffs=handoffs,
            completed_task_ids=(completed_task_ids),
            browser_result=(browser_result),
            browser_quality_report=None,
            completed=False,
            routing_errors=(routing_errors),
            errors=errors,
        )

    # =====================================================
    # FINAL MULTI-AGENT COMPLETION
    # =====================================================

    browser_quality_passed = passes_advanced_quality_gate(
        browser_quality_report,
        min_decision_efficiency=float(
            quality_task.payload.get(
                "min_decision_efficiency",
                0.30,
            )
        ),
        max_execution_failures=int(
            quality_task.payload.get(
                "max_execution_failures",
                1,
            )
        ),
    )

    completed = (
        browser_result.completed
        and browser_quality_passed
        and len(completed_task_ids) == len(plan.tasks)
        and not routing_errors
        and not errors
    )

    return MultiAgentRunResult(
        goal=goal,
        plan=plan,
        handoffs=handoffs,
        completed_task_ids=(completed_task_ids),
        browser_result=(browser_result),
        browser_quality_report=(browser_quality_report),
        completed=completed,
        routing_errors=(routing_errors),
        errors=errors,
    )
