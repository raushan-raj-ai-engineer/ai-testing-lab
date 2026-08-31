from __future__ import annotations

from typing import Any

from src.multi_agent.models import (
    BROWSER_AGENT,
    BROWSER_TASK,
    QUALITY_AGENT,
    QUALITY_TASK,
    AgentTask,
    HandoffEnvelope,
)

ROUTES = {
    BROWSER_TASK: BROWSER_AGENT,
    QUALITY_TASK: QUALITY_AGENT,
}


def route_task(
    task: AgentTask,
) -> str:
    """
    Determine which agent owns a task.

    Important:
    The planner cannot arbitrarily send a browser
    task to the quality agent or vice versa.
    """

    expected_agent = ROUTES.get(task.task_type)

    if expected_agent is None:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    if task.assigned_agent != expected_agent:
        raise ValueError(
            "Wrong agent assignment for "
            f"task '{task.id}': "
            f"expected {expected_agent}, "
            f"got {task.assigned_agent}"
        )

    return expected_agent


def validate_task_dependencies(
    task: AgentTask,
    completed_task_ids: list[str],
) -> None:
    """
    Ensure agents do not execute out of order.
    """

    missing = [
        dependency
        for dependency in task.depends_on
        if dependency not in completed_task_ids
    ]

    if missing:
        raise ValueError("Task dependencies are not complete: " + ", ".join(missing))


def build_handoff(
    *,
    correlation_id: str,
    source_agent: str,
    task: AgentTask,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> HandoffEnvelope:

    target_agent = route_task(task)

    handoff = HandoffEnvelope(
        correlation_id=correlation_id,
        task_id=task.id,
        source_agent=source_agent,
        target_agent=target_agent,
        payload=dict(payload),
        context=dict(context),
    )

    validate_handoff(
        handoff=handoff,
        task=task,
    )

    return handoff


def validate_handoff(
    handoff: HandoffEnvelope,
    task: AgentTask,
) -> None:
    """
    Validate agent-to-agent handoff schema,
    routing and context preservation.

    This acts like API contract testing,
    but between AI agents.
    """

    # =====================================================
    # BASIC ENVELOPE
    # =====================================================

    if not handoff.correlation_id:
        raise ValueError("Handoff correlation_id is required")

    if not handoff.task_id:
        raise ValueError("Handoff task_id is required")

    if handoff.task_id != task.id:
        raise ValueError("Handoff task_id does not match task")

    if handoff.source_agent == handoff.target_agent:
        raise ValueError("Agent cannot hand off a task to itself")

    # =====================================================
    # ROUTING
    # =====================================================

    expected_target = route_task(task)

    if handoff.target_agent != expected_target:
        raise ValueError("Handoff routed to wrong agent")

    # =====================================================
    # COMMON CONTEXT
    # =====================================================

    required_context = (
        "plan_id",
        "original_goal",
        "todo_name",
    )

    for key in required_context:
        if key not in handoff.context:
            raise ValueError(f"Missing handoff context: {key}")

    original_goal = handoff.context["original_goal"]

    payload_goal = handoff.payload.get("goal")

    if payload_goal != original_goal:
        raise ValueError("Handoff goal does not match original context")

    # =====================================================
    # BROWSER AGENT CONTRACT
    # =====================================================

    if handoff.target_agent == BROWSER_AGENT:
        if (
            not isinstance(
                payload_goal,
                str,
            )
            or not payload_goal.strip()
        ):
            raise ValueError("Browser-agent handoff requires goal")

        if not handoff.payload.get("todo_name"):
            raise ValueError("Browser-agent handoff requires todo_name")

    # =====================================================
    # QUALITY AGENT CONTRACT
    # =====================================================

    if handoff.target_agent == QUALITY_AGENT:
        if "browser_result" not in handoff.payload:
            raise ValueError("Quality-agent handoff requires browser_result")

        if handoff.payload["browser_result"] is None:
            raise ValueError("Quality-agent browser_result cannot be None")


def handoff_signature(
    handoff: HandoffEnvelope,
) -> tuple[str, str, str]:

    return (
        handoff.source_agent,
        handoff.target_agent,
        handoff.task_id,
    )


def has_handoff_loop(
    handoffs: list[HandoffEnvelope],
    repeat_threshold: int = 3,
) -> bool:
    """
    Detect repeated agent-to-agent handoffs.

    Example bad loop:

        planner -> browser
        planner -> browser
        planner -> browser
    """

    if repeat_threshold < 2:
        raise ValueError("repeat_threshold must be >= 2")

    if not handoffs:
        return False

    previous_signature = None

    repeat_count = 0

    for handoff in handoffs:
        signature = handoff_signature(handoff)

        if signature == previous_signature:
            repeat_count += 1

        else:
            previous_signature = signature

            repeat_count = 1

        if repeat_count >= repeat_threshold:
            return True

    return False
