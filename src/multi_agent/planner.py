from __future__ import annotations

from uuid import uuid4

from src.browser_agent.browser_agent import (
    extract_todo_from_full_goal,
)
from src.multi_agent.models import (
    BROWSER_AGENT,
    BROWSER_TASK,
    QUALITY_AGENT,
    QUALITY_TASK,
    AgentTask,
    MultiAgentPlan,
)


def build_multi_agent_plan(
    goal: str,
) -> MultiAgentPlan:
    """
    Deterministic planner for Session 17.

    Supported goal:

        Add <todo> and mark it complete

    The planner converts one user goal into
    two agent tasks:

        1. Browser agent executes browser work
        2. Quality agent evaluates the result

    We intentionally keep the planner deterministic
    in Session 17 so that handoff/orchestration
    failures are distinguishable from planner
    hallucinations.
    """

    if not goal.strip():
        raise ValueError("Multi-agent goal cannot be empty")

    todo_name = extract_todo_from_full_goal(goal)

    if not todo_name:
        raise ValueError("Unsupported multi-agent goal")

    plan_id = f"plan-{uuid4().hex[:8]}"

    browser_task_id = f"{plan_id}-browser"

    quality_task_id = f"{plan_id}-quality"

    browser_task = AgentTask(
        id=browser_task_id,
        task_type=BROWSER_TASK,
        assigned_agent=BROWSER_AGENT,
        description=("Execute the requested TodoMVC browser task"),
        payload={
            "goal": goal,
            "todo_name": todo_name,
            "max_steps": 5,
            "max_execution_retries": 1,
        },
        depends_on=(),
    )

    quality_task = AgentTask(
        id=quality_task_id,
        task_type=QUALITY_TASK,
        assigned_agent=QUALITY_AGENT,
        description=("Evaluate browser-agent execution quality and task completion"),
        payload={
            "goal": goal,
            "todo_name": todo_name,
            "min_decision_efficiency": 0.30,
            "max_execution_failures": 1,
        },
        depends_on=(browser_task_id,),
    )

    return MultiAgentPlan(
        id=plan_id,
        goal=goal,
        todo_name=todo_name,
        tasks=(
            browser_task,
            quality_task,
        ),
    )


def find_task(
    plan: MultiAgentPlan,
    task_type: str,
) -> AgentTask:

    for task in plan.tasks:
        if task.task_type == task_type:
            return task

    raise ValueError(f"Plan does not contain task type '{task_type}'")
