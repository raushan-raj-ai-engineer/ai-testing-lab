from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from src.guardrails.input_guard import (
    scan_untrusted_text,
)
from src.guardrails.models import (
    TOOL_ARGUMENT_VIOLATION,
    UNAUTHORIZED_TOOL,
    UNSAFE_NAVIGATION,
    GuardrailDecision,
    GuardrailFinding,
)

TODO_HOST = "demo.playwright.dev"

TODO_PATH_PREFIX = "/todomvc"


ALLOWED_BROWSER_TOOLS = frozenset({
    "browser_navigate",
    "browser_snapshot",
    "browser_type",
    "browser_click",
})


ALLOWED_ARGUMENTS = {
    "browser_navigate": {
        "url",
    },
    "browser_snapshot": set(),
    "browser_type": {
        "target",
        "text",
        "submit",
        "slowly",
        "element",
    },
    "browser_click": {
        "target",
        "button",
        "doubleClick",
        "modifiers",
        "element",
    },
}


def evaluate_tool_call_safety(
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ]
    | None,
) -> GuardrailDecision:
    """
    Enforce a least-privilege tool policy.

    The agent gets only the capabilities required
    for the TodoMVC task.
    """

    findings: list[GuardrailFinding] = []

    arguments = dict(arguments or {})

    # =====================================================
    # TOOL ALLOWLIST
    # =====================================================

    if tool_name not in ALLOWED_BROWSER_TOOLS:
        findings.append(
            GuardrailFinding(
                category=(UNAUTHORIZED_TOOL),
                severity="critical",
                message=("Tool is not allowed by the browser-agent policy"),
                evidence=tool_name,
            )
        )

        return GuardrailDecision(
            allowed=False,
            findings=tuple(findings),
        )

    # =====================================================
    # ARGUMENT ALLOWLIST
    # =====================================================

    allowed_arguments = ALLOWED_ARGUMENTS[tool_name]

    unknown_arguments = set(arguments) - allowed_arguments

    if unknown_arguments:
        findings.append(
            GuardrailFinding(
                category=(TOOL_ARGUMENT_VIOLATION),
                severity="high",
                message=("Tool call contains unauthorized arguments"),
                evidence=", ".join(sorted(unknown_arguments)),
            )
        )

    # =====================================================
    # NAVIGATION DOMAIN POLICY
    # =====================================================

    if tool_name == "browser_navigate":
        url = arguments.get("url")

        if (
            not isinstance(
                url,
                str,
            )
            or not url.strip()
        ):
            findings.append(
                GuardrailFinding(
                    category=(TOOL_ARGUMENT_VIOLATION),
                    severity="high",
                    message=("browser_navigate requires a valid URL"),
                )
            )

        else:
            parsed = urlparse(url)

            safe_scheme = parsed.scheme == "https"

            safe_host = parsed.hostname == TODO_HOST

            safe_path = parsed.path.startswith(TODO_PATH_PREFIX)

            if not (safe_scheme and safe_host and safe_path):
                findings.append(
                    GuardrailFinding(
                        category=(UNSAFE_NAVIGATION),
                        severity="critical",
                        message=(
                            "Browser navigation "
                            "outside the authorized "
                            "TodoMVC origin is blocked"
                        ),
                        evidence=url,
                    )
                )

    # =====================================================
    # TYPED TEXT IS ALSO UNTRUSTED
    # =====================================================

    if tool_name == "browser_type":
        text = arguments.get("text")

        if isinstance(
            text,
            str,
        ):
            findings.extend(scan_untrusted_text(text))

    return GuardrailDecision(
        allowed=not findings,
        findings=tuple(findings),
    )


class GuardedMCPClient:
    """
    Security proxy around the real MCP client.

    Tool policy is checked BEFORE execution.
    """

    def __init__(
        self,
        client: Any,
    ) -> None:

        self._client = client

        self.decisions: list[GuardrailDecision] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[
            str,
            Any,
        ]
        | None = None,
    ) -> Any:

        decision = evaluate_tool_call_safety(
            tool_name=name,
            arguments=arguments,
        )

        self.decisions.append(decision)

        if not decision.allowed:
            details = "; ".join(finding.message for finding in decision.findings)

            raise PermissionError(f"Guardrail blocked MCP tool '{name}': {details}")

        return await self._client.call_tool(
            name,
            arguments or {},
        )

    async def list_tools(
        self,
    ) -> Any:

        return await self._client.list_tools()

    def __getattr__(
        self,
        name: str,
    ) -> Any:

        return getattr(
            self._client,
            name,
        )
