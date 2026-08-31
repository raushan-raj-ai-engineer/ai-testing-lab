from dataclasses import dataclass

import pytest
from mcp.types import TextContent

import src.browser_agent.browser_agent as browser_agent
from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
    calculate_decision_efficiency,
    classify_goal_progress,
    count_duplicate_decisions,
    detect_trajectory_regressions,
    has_stuck_loop,
    passes_advanced_quality_gate,
)
from src.browser_agent.browser_agent import (
    BrowserAgentAttempt,
    validate_browser_action,
)

# =========================================================
# FAKE ACCESSIBILITY SNAPSHOTS
# =========================================================


EMPTY_SNAPSHOT = """
- heading "todos" [ref=e7]
- textbox "What needs to be done?" [ref=e8]
"""


ACTIVE_SNAPSHOT = """
- checkbox "Mark all as complete" [ref=e16]
- listitem [ref=e20]:
    - checkbox "Toggle Todo" [ref=e21]
    - generic [ref=e22]: Buy milk
- link "All" [ref=e30]
- link "Active" [ref=e31]
- link "Completed" [ref=e32]
"""


COMPLETED_SNAPSHOT = """
- checkbox "Mark all as complete" [checked] [ref=e16]
- listitem [ref=e20]:
    - checkbox "Toggle Todo" [checked] [ref=e21]
    - generic [ref=e22]: Buy milk
- link "All" [ref=e30]
- link "Active" [ref=e31]
- link "Completed" [ref=e32]
"""


# =========================================================
# FAKE TOOL DEFINITIONS
# =========================================================


def build_tools():

    return [
        {
            "type": "function",
            "function": {
                "name": "browser_type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "text": {
                            "type": "string",
                        },
                        "submit": {
                            "type": "boolean",
                        },
                    },
                    "required": [
                        "target",
                        "text",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "button": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "target",
                    ],
                },
            },
        },
    ]


# =========================================================
# 1. REJECTED ATTEMPT TRACING
# =========================================================


def test_rejected_attempt_is_traced(
    monkeypatch,
):

    ai_responses = iter([
        # Wrong tool
        (
            "browser_type",
            {
                "target": "e22",
                "text": "Buy milk",
                "submit": True,
            },
        ),
        # Correct tool
        (
            "browser_click",
            {
                "target": "e21",
                "button": "left",
            },
        ),
    ])

    def fake_choose_browser_action(
        goal,
        snapshot,
        tools,
    ):

        return next(ai_responses)

    monkeypatch.setattr(
        browser_agent,
        "choose_browser_action",
        fake_choose_browser_action,
    )

    attempts: list[BrowserAgentAttempt] = []

    tool_name, arguments = browser_agent.choose_valid_browser_action(
        goal=("Mark Buy milk as complete"),
        snapshot=ACTIVE_SNAPSHOT,
        tools=build_tools(),
        max_attempts=3,
        step_number=2,
        attempt_trace=attempts,
    )

    assert tool_name == "browser_click"

    assert arguments["target"] == "e21"

    assert len(attempts) == 2

    # First AI decision rejected
    assert attempts[0].accepted is False

    assert attempts[0].tool_name == "browser_type"

    assert "browser_click" in (attempts[0].rejection_reason or "")

    # Second AI decision accepted
    assert attempts[1].accepted is True

    assert attempts[1].tool_name == "browser_click"


# =========================================================
# 2. DECISION EFFICIENCY
# =========================================================


def test_retry_efficiency_score():

    attempts = [
        BrowserAgentAttempt(
            step_number=1,
            attempt_number=1,
            goal="Add Buy milk",
            tool_name="browser_type",
            arguments={},
            accepted=False,
            rejection_reason="bad",
        ),
        BrowserAgentAttempt(
            step_number=1,
            attempt_number=2,
            goal="Add Buy milk",
            tool_name="browser_type",
            arguments={},
            accepted=True,
        ),
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=1,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={},
            accepted=True,
        ),
    ]

    score = calculate_decision_efficiency(attempts)

    # 2 accepted / 3 total
    assert score == pytest.approx(2 / 3)


# =========================================================
# 3. DUPLICATE DECISION
# =========================================================


def test_duplicate_decision_detected():

    attempts = [
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=1,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={
                "target": "e32",
                "button": "left",
            },
            accepted=False,
            rejection_reason=("wrong target"),
        ),
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=2,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={
                "target": "e32",
                "button": "left",
            },
            accepted=False,
            rejection_reason=("wrong target"),
        ),
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=3,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={
                "target": "e21",
                "button": "left",
            },
            accepted=True,
        ),
    ]

    duplicates = count_duplicate_decisions(attempts)

    assert duplicates == 1


# =========================================================
# 4. STUCK LOOP
# =========================================================


def test_stuck_loop_detected():

    attempts = [
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=i,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={
                "target": "e32",
                "button": "left",
            },
            accepted=False,
            rejection_reason=("wrong target"),
        )
        for i in range(
            1,
            4,
        )
    ]

    assert (
        has_stuck_loop(
            attempts,
            repeat_threshold=3,
        )
        is True
    )


def test_corrected_action_not_stuck_loop():

    attempts = [
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=1,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={
                "target": "e32",
            },
            accepted=False,
            rejection_reason="wrong",
        ),
        BrowserAgentAttempt(
            step_number=2,
            attempt_number=2,
            goal=("Mark Buy milk as complete"),
            tool_name="browser_click",
            arguments={
                "target": "e21",
            },
            accepted=True,
        ),
    ]

    assert has_stuck_loop(attempts) is False


# =========================================================
# 5. PARTIAL COMPLETION
# =========================================================


def test_task_not_started():

    progress = classify_goal_progress(
        goal=("Add Buy milk and mark it complete"),
        snapshot=EMPTY_SNAPSHOT,
    )

    assert progress.status == "not_started"

    assert progress.completed_actions == 0

    assert progress.score == 0.0


def test_partial_task_completion():

    progress = classify_goal_progress(
        goal=("Add Buy milk and mark it complete"),
        snapshot=ACTIVE_SNAPSHOT,
    )

    assert progress.status == "partial"

    assert progress.completed_actions == 1

    assert progress.total_actions == 2

    assert progress.score == 0.5


def test_full_task_completion():

    progress = classify_goal_progress(
        goal=("Add Buy milk and mark it complete"),
        snapshot=COMPLETED_SNAPSHOT,
    )

    assert progress.status == "completed"

    assert progress.completed_actions == 2

    assert progress.score == 1.0


# =========================================================
# 6. ADVERSARIAL BROWSER STATE
# =========================================================


def test_global_checkbox_rejected():

    with pytest.raises(
        ValueError,
        match=("correct checkbox target is e21"),
    ):
        validate_browser_action(
            goal=("Mark Buy milk as complete"),
            tool_name=("browser_click"),
            arguments={
                # Wrong:
                # global mark-all checkbox
                "target": "e16",
                "button": "left",
            },
            snapshot=ACTIVE_SNAPSHOT,
        )


def test_completed_filter_rejected():

    with pytest.raises(
        ValueError,
        match=("correct checkbox target is e21"),
    ):
        validate_browser_action(
            goal=("Mark Buy milk as complete"),
            tool_name=("browser_click"),
            arguments={
                # Wrong:
                # Completed filter
                "target": "e32",
                "button": "left",
            },
            snapshot=ACTIVE_SNAPSHOT,
        )


def test_correct_checkbox_accepted():

    # Should NOT raise
    validate_browser_action(
        goal=("Mark Buy milk as complete"),
        tool_name="browser_click",
        arguments={
            "target": "e21",
            "button": "left",
        },
        snapshot=ACTIVE_SNAPSHOT,
    )


# =========================================================
# 7. FAKE MCP RESULT
# =========================================================


@dataclass
class FakeToolResult:
    text: str = ""
    is_error: bool = False

    @property
    def content(self):

        return [
            TextContent(
                type="text",
                text=self.text,
            )
        ]


# =========================================================
# 8. FAKE RECOVERY CLIENT
# =========================================================


class FakeRecoveryClient:
    def __init__(self):

        self.state = "missing"

        self.type_calls = 0

    async def call_tool(
        self,
        tool_name,
        arguments,
    ):

        # -------------------------------------------------
        # Snapshot
        # -------------------------------------------------

        if tool_name == "browser_snapshot":
            if self.state == "missing":
                return FakeToolResult(text=EMPTY_SNAPSHOT)

            if self.state == "active":
                return FakeToolResult(text=ACTIVE_SNAPSHOT)

            return FakeToolResult(text=COMPLETED_SNAPSHOT)

        # -------------------------------------------------
        # browser_type
        # -------------------------------------------------

        if tool_name == "browser_type":
            self.type_calls += 1

            # First execution intentionally fails
            if self.type_calls == 1:
                return FakeToolResult(
                    text=("temporary browser execution failure"),
                    is_error=True,
                )

            self.state = "active"

            return FakeToolResult(text="typed")

        # -------------------------------------------------
        # browser_click
        # -------------------------------------------------

        if tool_name == "browser_click":
            self.state = "completed"

            return FakeToolResult(text="clicked")

        raise AssertionError(f"Unexpected tool: {tool_name}")


# =========================================================
# 9. TOOL FAILURE RECOVERY
# =========================================================


@pytest.mark.anyio
async def test_agent_recovers_from_explicit_tool_failure(
    monkeypatch,
):

    # =====================================================
    # 1. Fake tool discovery
    # =====================================================

    async def fake_discover_tools(
        client,
    ):
        return build_tools()

    # =====================================================
    # 2. Fake AI decision
    #
    # Real Ollama call nahi karenge.
    #
    # Missing todo:
    #     browser_type
    #
    # Active todo:
    #     browser_click
    # =====================================================

    def fake_valid_action(
        goal,
        snapshot,
        tools,
        max_attempts=3,
        step_number=0,
        attempt_trace=None,
    ):

        if goal.startswith("Add "):
            tool_name = "browser_type"

            arguments = {
                "target": "e8",
                "text": "Buy milk",
                "submit": True,
            }

        else:
            tool_name = "browser_click"

            arguments = {
                "target": "e21",
                "button": "left",
            }

        # ---------------------------------------------
        # Session 15 decision tracing
        # ---------------------------------------------

        if attempt_trace is not None:
            attempt_trace.append(
                BrowserAgentAttempt(
                    step_number=step_number,
                    attempt_number=1,
                    goal=goal,
                    tool_name=tool_name,
                    arguments=dict(arguments),
                    accepted=True,
                    rejection_reason=None,
                )
            )

        return (
            tool_name,
            arguments,
        )

    # =====================================================
    # 3. Patch real dependencies
    # =====================================================

    monkeypatch.setattr(
        browser_agent,
        "discover_browser_agent_tools",
        fake_discover_tools,
    )

    monkeypatch.setattr(
        browser_agent,
        "choose_valid_browser_action",
        fake_valid_action,
    )

    # =====================================================
    # 4. Fake MCP client
    #
    # Expected behavior:
    #
    # browser_type attempt 1 -> FAIL
    # browser_type attempt 2 -> PASS
    # browser_click           -> PASS
    # =====================================================

    client = FakeRecoveryClient()

    # =====================================================
    # 5. Execute autonomous agent
    # =====================================================

    result = await browser_agent.run_browser_agent(
        client=client,
        goal=("Add Buy milk and mark it complete"),
        max_steps=5,
        # One retry allowed after explicit MCP failure
        max_execution_retries=1,
    )

    # =====================================================
    # 6. Final task completion
    # =====================================================

    assert result.completed is True

    assert "Buy milk" in result.final_snapshot

    assert "[checked]" in result.final_snapshot

    # =====================================================
    # 7. Successful browser steps
    #
    # Only successful actions are stored here.
    #
    # Step 1:
    #     browser_type
    #
    # Step 2:
    #     browser_click
    # =====================================================

    assert len(result.steps) == 2

    assert result.steps[0].tool_name == "browser_type"

    assert result.steps[1].tool_name == "browser_click"

    # =====================================================
    # 8. AI decision attempts
    #
    # Both AI decisions were valid immediately.
    # The failure happened at MCP execution layer,
    # not at decision-validation layer.
    # =====================================================

    assert len(result.attempts) == 2

    assert result.attempts[0].accepted is True

    assert result.attempts[1].accepted is True

    # =====================================================
    # 9. MCP execution attempts
    #
    # Expected:
    #
    # 1. browser_type -> FAIL
    # 2. browser_type -> PASS
    # 3. browser_click -> PASS
    # =====================================================

    assert len(result.execution_attempts) == 3

    first_execution = result.execution_attempts[0]

    second_execution = result.execution_attempts[1]

    third_execution = result.execution_attempts[2]

    # ---------------------------------------------
    # First execution failed
    # ---------------------------------------------

    assert first_execution.tool_name == "browser_type"

    assert first_execution.succeeded is False

    assert first_execution.error is not None

    assert "temporary browser execution failure" in first_execution.error

    # ---------------------------------------------
    # Retry succeeded
    # ---------------------------------------------

    assert second_execution.tool_name == "browser_type"

    assert second_execution.succeeded is True

    assert second_execution.error is None

    # ---------------------------------------------
    # Completion click succeeded
    # ---------------------------------------------

    assert third_execution.tool_name == "browser_click"

    assert third_execution.succeeded is True

    assert third_execution.error is None


# =========================================================
# 10. TRAJECTORY REGRESSION
# =========================================================


def test_trajectory_regression_detected():

    baseline = AdvancedAgentQualityReport(
        completed=True,
        progress_status=("completed"),
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

    current = AdvancedAgentQualityReport(
        completed=True,
        progress_status=("completed"),
        progress_score=1.0,
        # Regression
        executed_steps=3,
        decision_attempts=5,
        rejected_attempts=3,
        # Regression
        decision_efficiency=0.4,
        execution_attempts=4,
        # Regression
        execution_failures=1,
        # Regression
        duplicate_decisions=1,
        stuck_loop=False,
    )

    regressions = detect_trajectory_regressions(
        current=current,
        baseline=baseline,
    )

    assert "Executed step count increased" in regressions

    assert "Decision efficiency regressed" in regressions

    assert "Execution failures increased" in regressions

    assert "Duplicate decisions increased" in regressions


# =========================================================
# 11. GOOD QUALITY GATE
# =========================================================


def test_good_report_passes_quality_gate():

    report = AdvancedAgentQualityReport(
        completed=True,
        progress_status=("completed"),
        progress_score=1.0,
        executed_steps=2,
        decision_attempts=3,
        rejected_attempts=1,
        decision_efficiency=(2 / 3),
        execution_attempts=2,
        execution_failures=0,
        duplicate_decisions=0,
        stuck_loop=False,
    )

    assert passes_advanced_quality_gate(report) is True


# =========================================================
# 12. BAD QUALITY GATE
# =========================================================


def test_stuck_agent_fails_quality_gate():

    report = AdvancedAgentQualityReport(
        completed=False,
        progress_status=("partial"),
        progress_score=0.5,
        executed_steps=1,
        decision_attempts=6,
        rejected_attempts=5,
        decision_efficiency=(1 / 6),
        execution_attempts=1,
        execution_failures=0,
        duplicate_decisions=4,
        stuck_loop=True,
    )

    assert passes_advanced_quality_gate(report) is False
