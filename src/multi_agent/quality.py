from __future__ import annotations

from src.browser_agent.advanced_quality import (
    passes_advanced_quality_gate,
)
from src.multi_agent.models import (
    MultiAgentQualityReport,
    MultiAgentRunResult,
)
from src.multi_agent.router import (
    has_handoff_loop,
)

EXPECTED_HANDOFFS = 2


def is_context_preserved(
    result: MultiAgentRunResult,
) -> bool:
    """
    Validate that important context survives every
    agent-to-agent transition.
    """

    if not result.handoffs:
        return False

    correlation_ids = {handoff.correlation_id for handoff in result.handoffs}

    if len(correlation_ids) != 1:
        return False

    for handoff in result.handoffs:
        context = handoff.context

        if context.get("plan_id") != result.plan.id:
            return False

        if context.get("original_goal") != result.goal:
            return False

        if context.get("todo_name") != result.plan.todo_name:
            return False

        if handoff.payload.get("goal") != result.goal:
            return False

    return True


def build_multi_agent_quality_report(
    result: MultiAgentRunResult,
) -> MultiAgentQualityReport:

    total_tasks = len(result.plan.tasks)

    completed_tasks = len(result.completed_task_ids)

    task_completion_rate = completed_tasks / total_tasks if total_tasks else 0.0

    actual_handoffs = len(result.handoffs)

    handoff_success_rate = min(
        (actual_handoffs / EXPECTED_HANDOFFS),
        1.0,
    )

    context_preserved = is_context_preserved(result)

    handoff_loop = has_handoff_loop(result.handoffs)

    browser_quality_passed = False

    if result.browser_quality_report is not None:
        browser_quality_passed = passes_advanced_quality_gate(
            result.browser_quality_report,
            min_decision_efficiency=(0.30),
            max_execution_failures=(1),
        )

    release_passed = all((
        result.completed,
        total_tasks > 0,
        completed_tasks == total_tasks,
        handoff_success_rate == 1.0,
        context_preserved,
        not result.routing_errors,
        not result.errors,
        not handoff_loop,
        browser_quality_passed,
    ))

    return MultiAgentQualityReport(
        completed=result.completed,
        total_tasks=total_tasks,
        completed_tasks=(completed_tasks),
        task_completion_rate=(task_completion_rate),
        expected_handoffs=(EXPECTED_HANDOFFS),
        actual_handoffs=(actual_handoffs),
        handoff_success_rate=(handoff_success_rate),
        context_preserved=(context_preserved),
        routing_errors=len(result.routing_errors),
        execution_errors=len(result.errors),
        handoff_loop=(handoff_loop),
        browser_quality_passed=(browser_quality_passed),
        release_passed=(release_passed),
    )


def passes_multi_agent_quality_gate(
    report: MultiAgentQualityReport,
) -> bool:

    return report.release_passed
