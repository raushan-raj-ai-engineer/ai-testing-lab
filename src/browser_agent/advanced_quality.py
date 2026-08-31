import json
from dataclasses import dataclass

from src.browser_agent.browser_agent import (
    BrowserAgentAttempt,
    BrowserAgentRunResult,
    extract_todo_from_full_goal,
    get_todo_state,
)

# =========================================================
# GOAL PROGRESS
# =========================================================


@dataclass(frozen=True)
class GoalProgress:
    status: str

    completed_actions: int
    total_actions: int

    score: float


def classify_goal_progress(
    goal: str,
    snapshot: str,
) -> GoalProgress:
    """
    For our TodoMVC autonomous goal:

        1. Add todo
        2. Complete todo

    missing:
        0 / 2

    active:
        1 / 2

    completed:
        2 / 2
    """

    todo_name = extract_todo_from_full_goal(goal)

    if todo_name is None:
        raise ValueError("Unsupported autonomous goal for progress calculation")

    state = get_todo_state(
        todo_name,
        snapshot,
    )

    if state == "missing":
        return GoalProgress(
            status="not_started",
            completed_actions=0,
            total_actions=2,
            score=0.0,
        )

    if state == "active":
        return GoalProgress(
            status="partial",
            completed_actions=1,
            total_actions=2,
            score=0.5,
        )

    if state == "completed":
        return GoalProgress(
            status="completed",
            completed_actions=2,
            total_actions=2,
            score=1.0,
        )

    raise ValueError(f"Unknown todo state: {state}")


# =========================================================
# DECISION / RETRY EFFICIENCY
# =========================================================


def calculate_decision_efficiency(
    attempts: list[BrowserAgentAttempt],
) -> float:
    """
    accepted decisions / total AI decisions

    Example:

        Attempt 1 ❌
        Attempt 2 ✅
        Attempt 3 ✅

        2 / 3 = 0.667
    """

    if not attempts:
        return 1.0

    accepted = sum(1 for attempt in attempts if attempt.accepted)

    return accepted / len(attempts)


def count_rejected_attempts(
    attempts: list[BrowserAgentAttempt],
) -> int:

    return sum(1 for attempt in attempts if not attempt.accepted)


# =========================================================
# MCP EXECUTION RELIABILITY
# =========================================================


def count_execution_failures(
    result: BrowserAgentRunResult,
) -> int:

    return sum(1 for attempt in result.execution_attempts if not attempt.succeeded)


# =========================================================
# DECISION SIGNATURE
# =========================================================


def _attempt_signature(
    attempt: BrowserAgentAttempt,
) -> str:
    """
    Build deterministic representation of a decision.

    Goal is included so identical actions for different
    subgoals are not automatically considered duplicates.
    """

    arguments = json.dumps(
        attempt.arguments,
        sort_keys=True,
        default=str,
    )

    return f"{attempt.goal}::{attempt.tool_name}::{arguments}"


# =========================================================
# DUPLICATE DECISION DETECTION
# =========================================================


def count_duplicate_decisions(
    attempts: list[BrowserAgentAttempt],
) -> int:
    """
    Count consecutive identical decisions.

    Example:

        click e32 ❌
        click e32 ❌
        click e21 ✅

    duplicate_count = 1
    """

    if len(attempts) < 2:
        return 0

    duplicate_count = 0

    previous_signature: str | None = None

    for attempt in attempts:
        current_signature = _attempt_signature(attempt)

        if previous_signature is not None and current_signature == previous_signature:
            duplicate_count += 1

        previous_signature = current_signature

    return duplicate_count


# =========================================================
# STUCK LOOP DETECTION
# =========================================================


def has_stuck_loop(
    attempts: list[BrowserAgentAttempt],
    repeat_threshold: int = 3,
) -> bool:
    """
    Agent is considered stuck if it repeats exactly the
    same decision several consecutive times.

    Example:

        click e32
        click e32
        click e32

    threshold=3
    => stuck
    """

    if repeat_threshold < 2:
        raise ValueError("repeat_threshold must be >= 2")

    if not attempts:
        return False

    previous_signature: str | None = None

    repeat_count = 0

    for attempt in attempts:
        signature = _attempt_signature(attempt)

        if signature == previous_signature:
            repeat_count += 1

        else:
            repeat_count = 1

            previous_signature = signature

        if repeat_count >= repeat_threshold:
            return True

    return False


# =========================================================
# ADVANCED QUALITY REPORT
# =========================================================


@dataclass(frozen=True)
class AdvancedAgentQualityReport:
    completed: bool

    progress_status: str
    progress_score: float

    executed_steps: int

    decision_attempts: int
    rejected_attempts: int
    decision_efficiency: float

    execution_attempts: int
    execution_failures: int

    duplicate_decisions: int
    stuck_loop: bool


def build_advanced_quality_report(
    goal: str,
    result: BrowserAgentRunResult,
) -> AdvancedAgentQualityReport:

    progress = classify_goal_progress(
        goal=goal,
        snapshot=result.final_snapshot,
    )

    rejected_attempts = count_rejected_attempts(result.attempts)

    decision_efficiency = calculate_decision_efficiency(result.attempts)

    execution_failures = count_execution_failures(result)

    duplicate_decisions = count_duplicate_decisions(result.attempts)

    stuck_loop = has_stuck_loop(result.attempts)

    return AdvancedAgentQualityReport(
        completed=result.completed,
        progress_status=(progress.status),
        progress_score=(progress.score),
        executed_steps=len(result.steps),
        decision_attempts=len(result.attempts),
        rejected_attempts=(rejected_attempts),
        decision_efficiency=(decision_efficiency),
        execution_attempts=len(result.execution_attempts),
        execution_failures=(execution_failures),
        duplicate_decisions=(duplicate_decisions),
        stuck_loop=(stuck_loop),
    )


# =========================================================
# ADVANCED QUALITY GATE
# =========================================================


def passes_advanced_quality_gate(
    report: AdvancedAgentQualityReport,
    min_decision_efficiency: float = 0.30,
    max_execution_failures: int = 1,
) -> bool:
    """
    Hard production-style quality gate.

    Task MUST complete.

    Agent MUST NOT be stuck.

    Some self-correction is allowed.
    """

    if not report.completed:
        return False

    if report.progress_score < 1.0:
        return False

    if report.stuck_loop:
        return False

    if report.decision_efficiency < min_decision_efficiency:
        return False

    if report.execution_failures > max_execution_failures:
        return False

    return True


# =========================================================
# TRAJECTORY REGRESSION DETECTION
# =========================================================


def detect_trajectory_regressions(
    current: AdvancedAgentQualityReport,
    baseline: AdvancedAgentQualityReport,
    allowed_efficiency_drop: float = 0.15,
    allowed_extra_steps: int = 0,
) -> list[str]:
    """
    Compare current agent behavior against previous
    known-good baseline.
    """

    regressions: list[str] = []

    # -----------------------------------------------------
    # Task completion regression
    # -----------------------------------------------------

    if baseline.completed and not current.completed:
        regressions.append("Task completion regressed")

    # -----------------------------------------------------
    # Progress regression
    # -----------------------------------------------------

    if current.progress_score < baseline.progress_score:
        regressions.append("Goal progress regressed")

    # -----------------------------------------------------
    # Extra browser steps
    # -----------------------------------------------------

    if current.executed_steps > baseline.executed_steps + allowed_extra_steps:
        regressions.append("Executed step count increased")

    # -----------------------------------------------------
    # Decision efficiency regression
    # -----------------------------------------------------

    efficiency_drop = baseline.decision_efficiency - current.decision_efficiency

    if efficiency_drop > allowed_efficiency_drop:
        regressions.append("Decision efficiency regressed")

    # -----------------------------------------------------
    # MCP reliability regression
    # -----------------------------------------------------

    if current.execution_failures > baseline.execution_failures:
        regressions.append("Execution failures increased")

    # -----------------------------------------------------
    # Duplicate decision regression
    # -----------------------------------------------------

    if current.duplicate_decisions > baseline.duplicate_decisions:
        regressions.append("Duplicate decisions increased")

    # -----------------------------------------------------
    # Loop regression
    # -----------------------------------------------------

    if current.stuck_loop and not baseline.stuck_loop:
        regressions.append("Agent developed a stuck loop")

    return regressions
