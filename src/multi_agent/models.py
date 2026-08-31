from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
)
from src.browser_agent.browser_agent import (
    BrowserAgentRunResult,
)

# =========================================================
# AGENT NAMES
# =========================================================

PLANNER_AGENT = "planner_agent"
BROWSER_AGENT = "browser_agent"
QUALITY_AGENT = "quality_agent"


# =========================================================
# TASK TYPES
# =========================================================

BROWSER_TASK = "browser"
QUALITY_TASK = "quality"


# =========================================================
# AGENT TASK
# =========================================================


@dataclass(frozen=True)
class AgentTask:
    id: str

    task_type: str

    assigned_agent: str

    description: str

    payload: dict[str, Any]

    depends_on: tuple[str, ...] = ()


# =========================================================
# MULTI-AGENT PLAN
# =========================================================


@dataclass(frozen=True)
class MultiAgentPlan:
    id: str

    goal: str

    todo_name: str

    tasks: tuple[AgentTask, ...]


# =========================================================
# HANDOFF ENVELOPE
# =========================================================


@dataclass(frozen=True)
class HandoffEnvelope:
    correlation_id: str

    task_id: str

    source_agent: str

    target_agent: str

    payload: dict[str, Any]

    context: dict[str, Any]


# =========================================================
# MULTI-AGENT RUN RESULT
# =========================================================


@dataclass
class MultiAgentRunResult:
    goal: str

    plan: MultiAgentPlan

    handoffs: list[HandoffEnvelope]

    completed_task_ids: list[str]

    browser_result: BrowserAgentRunResult | None

    browser_quality_report: AdvancedAgentQualityReport | None

    completed: bool

    routing_errors: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)


# =========================================================
# MULTI-AGENT QUALITY REPORT
# =========================================================


@dataclass(frozen=True)
class MultiAgentQualityReport:
    completed: bool

    total_tasks: int

    completed_tasks: int

    task_completion_rate: float

    expected_handoffs: int

    actual_handoffs: int

    handoff_success_rate: float

    context_preserved: bool

    routing_errors: int

    execution_errors: int

    handoff_loop: bool

    browser_quality_passed: bool

    release_passed: bool
