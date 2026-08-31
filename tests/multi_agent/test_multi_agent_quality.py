import pytest

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
)
from src.browser_agent.browser_agent import (
    BrowserAgentRunResult,
)
from src.multi_agent.models import (
    MultiAgentRunResult,
)
from src.multi_agent.planner import (
    build_multi_agent_plan,
)
from src.multi_agent.quality import (
    build_multi_agent_quality_report,
    passes_multi_agent_quality_gate,
)
from src.multi_agent.router import (
    build_handoff,
)

pytestmark = pytest.mark.multi_agent


def completed_browser_result():

    return BrowserAgentRunResult(
        completed=True,
        steps=[],
        final_snapshot="""
        - listitem:
          - checkbox "Toggle Todo" [checked] [ref=e21]
          - generic [ref=e22]: Buy milk
        """,
        attempts=[],
        execution_attempts=[],
    )


def good_browser_quality():

    return AdvancedAgentQualityReport(
        completed=True,
        progress_status="completed",
        progress_score=1.0,
        executed_steps=2,
        decision_attempts=2,
        rejected_attempts=0,
        decision_efficiency=1.0,
        execution_attempts=2,
        execution_failures=0,
        duplicate_decisions=0,
        stuck_loop=False,
    )


def build_good_result():

    goal = "Add Buy milk and mark it complete"

    plan = build_multi_agent_plan(goal)

    browser_task = plan.tasks[0]

    quality_task = plan.tasks[1]

    context = {
        "plan_id": plan.id,
        "original_goal": goal,
        "todo_name": "Buy milk",
    }

    correlation_id = f"correlation-{plan.id}"

    browser_result = completed_browser_result()

    browser_handoff = build_handoff(
        correlation_id=(correlation_id),
        source_agent="planner_agent",
        task=browser_task,
        payload=browser_task.payload,
        context=context,
    )

    quality_payload = dict(quality_task.payload)

    quality_payload["browser_result"] = browser_result

    quality_handoff = build_handoff(
        correlation_id=(correlation_id),
        source_agent="browser_agent",
        task=quality_task,
        payload=quality_payload,
        context=context,
    )

    return MultiAgentRunResult(
        goal=goal,
        plan=plan,
        handoffs=[
            browser_handoff,
            quality_handoff,
        ],
        completed_task_ids=[
            browser_task.id,
            quality_task.id,
        ],
        browser_result=(browser_result),
        browser_quality_report=(good_browser_quality()),
        completed=True,
        routing_errors=[],
        errors=[],
    )


def test_good_multi_agent_run_passes():

    result = build_good_result()

    report = build_multi_agent_quality_report(result)

    assert report.completed is True

    assert report.task_completion_rate == 1.0

    assert report.handoff_success_rate == 1.0

    assert report.context_preserved is True

    assert report.handoff_loop is False

    assert report.release_passed is True

    assert passes_multi_agent_quality_gate(report) is True


def test_multi_agent_gate_fails_on_execution_error():

    result = build_good_result()

    result.completed = False

    result.errors.append("browser failed")

    report = build_multi_agent_quality_report(result)

    assert report.execution_errors == 1

    assert report.release_passed is False


def test_multi_agent_gate_fails_when_context_corrupted():

    result = build_good_result()

    original = result.handoffs[1]

    from src.multi_agent.models import (
        HandoffEnvelope,
    )

    corrupted = HandoffEnvelope(
        correlation_id=(original.correlation_id),
        task_id=(original.task_id),
        source_agent=(original.source_agent),
        target_agent=(original.target_agent),
        payload=(original.payload),
        context={
            **original.context,
            "original_goal": "wrong goal",
        },
    )

    result.handoffs[1] = corrupted

    report = build_multi_agent_quality_report(result)

    assert report.context_preserved is False

    assert report.release_passed is False
