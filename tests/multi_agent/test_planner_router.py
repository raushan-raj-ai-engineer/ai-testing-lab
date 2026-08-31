import pytest

from src.multi_agent.models import (
    BROWSER_AGENT,
    BROWSER_TASK,
    QUALITY_AGENT,
    AgentTask,
    HandoffEnvelope,
)
from src.multi_agent.planner import (
    build_multi_agent_plan,
)
from src.multi_agent.router import (
    build_handoff,
    has_handoff_loop,
    route_task,
    validate_handoff,
    validate_task_dependencies,
)

pytestmark = pytest.mark.multi_agent


def test_planner_creates_browser_and_quality_tasks():

    plan = build_multi_agent_plan("Add Buy milk and mark it complete")

    assert len(plan.tasks) == 2

    browser_task = plan.tasks[0]

    quality_task = plan.tasks[1]

    assert browser_task.assigned_agent == BROWSER_AGENT

    assert quality_task.assigned_agent == QUALITY_AGENT

    assert quality_task.depends_on == (browser_task.id,)


def test_router_routes_browser_task():

    plan = build_multi_agent_plan("Add Buy milk and mark it complete")

    assert route_task(plan.tasks[0]) == BROWSER_AGENT


def test_router_rejects_wrong_agent():

    task = AgentTask(
        id="task-1",
        task_type=BROWSER_TASK,
        assigned_agent=QUALITY_AGENT,
        description="wrong route",
        payload={
            "goal": ("Add Buy milk and mark it complete"),
            "todo_name": "Buy milk",
        },
    )

    with pytest.raises(
        ValueError,
        match="Wrong agent assignment",
    ):
        route_task(task)


def test_quality_task_cannot_execute_before_browser():

    plan = build_multi_agent_plan("Add Buy milk and mark it complete")

    quality_task = plan.tasks[1]

    with pytest.raises(
        ValueError,
        match="dependencies",
    ):
        validate_task_dependencies(
            task=quality_task,
            completed_task_ids=[],
        )


def test_handoff_preserves_original_context():

    plan = build_multi_agent_plan("Add Buy milk and mark it complete")

    browser_task = plan.tasks[0]

    handoff = build_handoff(
        correlation_id="correlation-1",
        source_agent="planner_agent",
        task=browser_task,
        payload=browser_task.payload,
        context={
            "plan_id": plan.id,
            "original_goal": plan.goal,
            "todo_name": plan.todo_name,
        },
    )

    assert handoff.context["original_goal"] == plan.goal

    assert handoff.context["todo_name"] == "Buy milk"


def test_handoff_rejects_context_hallucination():

    plan = build_multi_agent_plan("Add Buy milk and mark it complete")

    browser_task = plan.tasks[0]

    handoff = HandoffEnvelope(
        correlation_id="correlation-1",
        task_id=browser_task.id,
        source_agent="planner_agent",
        target_agent=(BROWSER_AGENT),
        payload={
            "goal": ("Add Buy coffee and mark it complete"),
            "todo_name": "Buy milk",
        },
        context={
            "plan_id": plan.id,
            "original_goal": plan.goal,
            "todo_name": plan.todo_name,
        },
    )

    with pytest.raises(
        ValueError,
        match=("Handoff goal does not match"),
    ):
        validate_handoff(
            handoff=handoff,
            task=browser_task,
        )


def test_handoff_loop_detected():

    plan = build_multi_agent_plan("Add Buy milk and mark it complete")

    browser_task = plan.tasks[0]

    handoff = build_handoff(
        correlation_id="correlation-1",
        source_agent="planner_agent",
        task=browser_task,
        payload=browser_task.payload,
        context={
            "plan_id": plan.id,
            "original_goal": plan.goal,
            "todo_name": plan.todo_name,
        },
    )

    handoffs = [
        handoff,
        handoff,
        handoff,
    ]

    assert has_handoff_loop(handoffs) is True
