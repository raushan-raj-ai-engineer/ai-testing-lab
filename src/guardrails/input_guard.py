from __future__ import annotations

import re

from src.browser_agent.browser_agent import (
    extract_todo_from_full_goal,
)
from src.guardrails.models import (
    INPUT_VALIDATION,
    PROMPT_INJECTION,
    SCOPE_VIOLATION,
    SECRET_EXFILTRATION,
    UNAUTHORIZED_TOOL,
    UNSAFE_NAVIGATION,
    GuardrailDecision,
    GuardrailFinding,
)

# =========================================================
# LOW-LEVEL THREAT PATTERNS
# =========================================================


PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+)?"
        r"(?:previous|prior|above)\s+"
        r"(?:instructions?|rules?|prompts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+(?:the\s+)?"
        r"(?:previous|system|developer)\s+"
        r"(?:instructions?|message|prompt)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|dump)\s+"
        r"(?:the\s+)?"
        r"(?:system|developer|hidden)\s+"
        r"(?:prompt|instructions?|message)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbypass\s+(?:the\s+)?"
        r"(?:guardrails?|safety|policy|rules?)\b",
        re.IGNORECASE,
    ),
)


SECRET_EXFILTRATION_PATTERNS = (
    re.compile(
        r"\b(?:reveal|show|print|dump|send|"
        r"export|copy)\b"
        r".{0,50}"
        r"\b(?:api[_ -]?key|password|secret|"
        r"access[_ -]?token|auth[_ -]?token|"
        r"cookies?|environment variables?)\b",
        re.IGNORECASE,
    ),
)


UNAUTHORIZED_TOOL_PATTERNS = (
    re.compile(
        r"\b(?:use|call|invoke|run)\s+"
        r"(?:the\s+)?"
        r"(?:browser_evaluate|browser_run_code|"
        r"shell|terminal|exec|javascript)\b",
        re.IGNORECASE,
    ),
)


EXTERNAL_NAVIGATION_PATTERNS = (
    re.compile(
        r"\b(?:navigate|open|visit|go)\s+"
        r"(?:to\s+)?https?://",
        re.IGNORECASE,
    ),
)


# =========================================================
# GENERIC TEXT SCANNER
# =========================================================


def scan_untrusted_text(
    text: str,
) -> tuple[GuardrailFinding, ...]:

    findings: list[GuardrailFinding] = []

    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)

        if match:
            findings.append(
                GuardrailFinding(
                    category=(PROMPT_INJECTION),
                    severity="critical",
                    message=("Prompt-injection pattern detected"),
                    evidence=(match.group(0)),
                )
            )

            break

    for pattern in SECRET_EXFILTRATION_PATTERNS:
        match = pattern.search(text)

        if match:
            findings.append(
                GuardrailFinding(
                    category=(SECRET_EXFILTRATION),
                    severity="critical",
                    message=("Sensitive-data exfiltration request detected"),
                    evidence=(match.group(0)),
                )
            )

            break

    for pattern in UNAUTHORIZED_TOOL_PATTERNS:
        match = pattern.search(text)

        if match:
            findings.append(
                GuardrailFinding(
                    category=(UNAUTHORIZED_TOOL),
                    severity="high",
                    message=("Request attempts to force an unauthorized tool"),
                    evidence=(match.group(0)),
                )
            )

            break

    for pattern in EXTERNAL_NAVIGATION_PATTERNS:
        match = pattern.search(text)

        if match:
            findings.append(
                GuardrailFinding(
                    category=(UNSAFE_NAVIGATION),
                    severity="high",
                    message=("External navigation request detected"),
                    evidence=(match.group(0)),
                )
            )

            break

    return tuple(findings)


# =========================================================
# USER-GOAL GUARDRAIL
# =========================================================


def evaluate_user_goal(
    goal: str,
) -> GuardrailDecision:
    """
    Safety gate before the planner sees the request.

    Session 18 authorized functional scope:

        Add <todo> and mark it complete

    Anything else must be rejected instead of
    allowing the agent to guess.
    """

    findings: list[GuardrailFinding] = []

    if not isinstance(
        goal,
        str,
    ):
        return GuardrailDecision(
            allowed=False,
            findings=(
                GuardrailFinding(
                    category=(INPUT_VALIDATION),
                    severity="high",
                    message=("Goal must be a string"),
                ),
            ),
        )

    clean_goal = goal.strip()

    if not clean_goal:
        return GuardrailDecision(
            allowed=False,
            findings=(
                GuardrailFinding(
                    category=(INPUT_VALIDATION),
                    severity="high",
                    message=("Goal cannot be empty"),
                ),
            ),
        )

    if len(clean_goal) > 500:
        findings.append(
            GuardrailFinding(
                category=(INPUT_VALIDATION),
                severity="medium",
                message=("Goal exceeds maximum supported length"),
            )
        )

    if any(
        ord(character) < 32 and character not in "\t\n\r" for character in clean_goal
    ):
        findings.append(
            GuardrailFinding(
                category=(INPUT_VALIDATION),
                severity="high",
                message=("Goal contains unsupported control characters"),
            )
        )

    findings.extend(scan_untrusted_text(clean_goal))

    # =====================================================
    # STRICT AUTHORIZED FUNCTIONAL SCOPE
    # =====================================================

    todo_name = extract_todo_from_full_goal(clean_goal)

    if not todo_name:
        findings.append(
            GuardrailFinding(
                category=(SCOPE_VIOLATION),
                severity="high",
                message=("Request is outside the authorized TodoMVC task scope"),
            )
        )

    elif len(todo_name) > 160:
        findings.append(
            GuardrailFinding(
                category=(INPUT_VALIDATION),
                severity="medium",
                message=("Todo text exceeds maximum supported length"),
            )
        )

    return GuardrailDecision(
        allowed=not findings,
        findings=tuple(findings),
    )
