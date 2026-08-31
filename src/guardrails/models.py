from __future__ import annotations

from dataclasses import dataclass, field

from src.multi_agent.models import (
    MultiAgentRunResult,
)

# =========================================================
# SAFETY CATEGORIES
# =========================================================

PROMPT_INJECTION = "prompt_injection"
SECRET_EXFILTRATION = "secret_exfiltration"
SCOPE_VIOLATION = "scope_violation"
UNAUTHORIZED_TOOL = "unauthorized_tool"
UNSAFE_NAVIGATION = "unsafe_navigation"
DESTRUCTIVE_ACTION = "destructive_action"
HANDOFF_TAMPERING = "handoff_tampering"
TOOL_ARGUMENT_VIOLATION = "tool_argument_violation"
INPUT_VALIDATION = "input_validation"


# =========================================================
# FINDING
# =========================================================


@dataclass(frozen=True)
class GuardrailFinding:
    category: str

    severity: str

    message: str

    evidence: str | None = None


# =========================================================
# DECISION
# =========================================================


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool

    findings: tuple[GuardrailFinding, ...] = ()

    @property
    def categories(
        self,
    ) -> tuple[str, ...]:

        return tuple(finding.category for finding in self.findings)


# =========================================================
# ADVERSARIAL DATASET
# =========================================================


@dataclass(frozen=True)
class AdversarialCase:
    id: str

    prompt: str

    expected_allowed: bool

    expected_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdversarialCaseEvaluation:
    case_id: str

    expected_allowed: bool

    actual_allowed: bool

    expected_categories: tuple[str, ...]

    actual_categories: tuple[str, ...]

    passed: bool


# =========================================================
# DATASET QUALITY
# =========================================================


@dataclass(frozen=True)
class GuardrailQualityReport:
    total_cases: int

    passed_cases: int

    failed_cases: int

    attack_cases: int

    benign_cases: int

    blocked_attacks: int

    missed_attacks: int

    false_positives: int

    detection_rate: float

    false_positive_rate: float

    release_failures: tuple[str, ...]

    release_passed: bool


# =========================================================
# GUARDED MULTI-AGENT RESULT
# =========================================================


@dataclass
class GuardedMultiAgentRunResult:
    goal: str

    input_decision: GuardrailDecision

    multi_agent_result: MultiAgentRunResult | None

    handoff_decisions: list[GuardrailDecision] = field(default_factory=list)

    tool_decisions: list[GuardrailDecision] = field(default_factory=list)

    completed: bool = False


# =========================================================
# RUNTIME SAFETY REPORT
# =========================================================


@dataclass(frozen=True)
class RuntimeGuardrailQualityReport:
    input_allowed: bool

    handoffs_safe: bool

    tools_safe: bool

    multi_agent_completed: bool

    multi_agent_quality_passed: bool

    total_findings: int

    safety_release_passed: bool
