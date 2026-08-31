from __future__ import annotations

from typing import Any

from src.guardrails.input_guard import (
    scan_untrusted_text,
)
from src.guardrails.models import (
    HANDOFF_TAMPERING,
    GuardrailDecision,
    GuardrailFinding,
)
from src.multi_agent.models import (
    BROWSER_AGENT,
    PLANNER_AGENT,
    QUALITY_AGENT,
    AgentTask,
    HandoffEnvelope,
)
from src.multi_agent.router import (
    validate_handoff,
)

ALLOWED_AGENT_ROUTES = {
    (
        PLANNER_AGENT,
        BROWSER_AGENT,
    ),
    (
        BROWSER_AGENT,
        QUALITY_AGENT,
    ),
}


def _extract_strings(
    value: Any,
) -> list[str]:

    strings: list[str] = []

    if isinstance(
        value,
        str,
    ):
        strings.append(value)

    elif isinstance(
        value,
        dict,
    ):
        for nested_value in value.values():
            strings.extend(_extract_strings(nested_value))

    elif isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        for nested_value in value:
            strings.extend(_extract_strings(nested_value))

    return strings


def evaluate_handoff_safety(
    handoff: HandoffEnvelope,
    task: AgentTask,
) -> GuardrailDecision:
    """
    Security validation for agent-to-agent
    communication.

    A handoff is treated as an untrusted boundary.
    """

    findings: list[GuardrailFinding] = []

    # =====================================================
    # ROUTE POLICY
    # =====================================================

    route = (
        handoff.source_agent,
        handoff.target_agent,
    )

    if route not in ALLOWED_AGENT_ROUTES:
        findings.append(
            GuardrailFinding(
                category=(HANDOFF_TAMPERING),
                severity="critical",
                message=("Unauthorized agent-to-agent route"),
                evidence=(f"{handoff.source_agent} -> {handoff.target_agent}"),
            )
        )

    # =====================================================
    # EXISTING HANDOFF CONTRACT
    # =====================================================

    try:
        validate_handoff(
            handoff=handoff,
            task=task,
        )

    except ValueError as exc:
        findings.append(
            GuardrailFinding(
                category=(HANDOFF_TAMPERING),
                severity="high",
                message=("Handoff contract validation failed"),
                evidence=str(exc),
            )
        )

    # =====================================================
    # SCAN PAYLOAD + CONTEXT FOR INJECTION
    # =====================================================

    untrusted_strings = _extract_strings(handoff.payload) + _extract_strings(
        handoff.context
    )

    for text in untrusted_strings:
        findings.extend(scan_untrusted_text(text))

    return GuardrailDecision(
        allowed=not findings,
        findings=tuple(findings),
    )
