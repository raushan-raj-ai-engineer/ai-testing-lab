from __future__ import annotations

from typing import Any

from src.guardrails.handoff_guard import (
    evaluate_handoff_safety,
)
from src.guardrails.input_guard import (
    evaluate_user_goal,
)
from src.guardrails.models import (
    GuardedMultiAgentRunResult,
)
from src.guardrails.tool_guard import (
    GuardedMCPClient,
)
from src.multi_agent.orchestrator import (
    run_multi_agent_system,
)


async def run_guarded_multi_agent_system(
    client: Any,
    goal: str,
    browser_executor: Any | None = None,
    quality_executor: Any | None = None,
) -> GuardedMultiAgentRunResult:
    """
    Secure wrapper around Session 17.

    Security boundaries:

        User -> Input Guard
        Agent -> Handoff Guard
        Agent -> MCP Tool Guard
    """

    # =====================================================
    # INPUT GUARD
    # =====================================================

    input_decision = evaluate_user_goal(goal)

    if not input_decision.allowed:
        return GuardedMultiAgentRunResult(
            goal=goal,
            input_decision=(input_decision),
            multi_agent_result=None,
            handoff_decisions=[],
            tool_decisions=[],
            completed=False,
        )

    # =====================================================
    # MCP TOOL SECURITY PROXY
    # =====================================================

    guarded_client = GuardedMCPClient(client)

    # =====================================================
    # MULTI-AGENT EXECUTION
    # =====================================================

    result = await run_multi_agent_system(
        client=guarded_client,
        goal=goal,
        browser_executor=(browser_executor),
        quality_executor=(quality_executor),
    )

    # =====================================================
    # HANDOFF SECURITY VALIDATION
    # =====================================================

    handoff_decisions = []

    task_lookup = {task.id: task for task in result.plan.tasks}

    for handoff in result.handoffs:
        task = task_lookup.get(handoff.task_id)

        if task is None:
            continue

        decision = evaluate_handoff_safety(
            handoff=handoff,
            task=task,
        )

        handoff_decisions.append(decision)

    handoffs_safe = all(decision.allowed for decision in handoff_decisions)

    tools_safe = all(decision.allowed for decision in guarded_client.decisions)

    completed = result.completed and handoffs_safe and tools_safe

    return GuardedMultiAgentRunResult(
        goal=goal,
        input_decision=(input_decision),
        multi_agent_result=(result),
        handoff_decisions=(handoff_decisions),
        tool_decisions=list(guarded_client.decisions),
        completed=completed,
    )
