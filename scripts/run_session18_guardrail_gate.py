from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from mcp import ClientSession
from mcp.client.streamable_http import (
    streamable_http_client,
)

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.guardrails.quality import (  # noqa: E402
    build_runtime_guardrail_report,
    evaluate_adversarial_dataset,
    load_adversarial_dataset,
    write_adversarial_report,
)
from src.guardrails.safe_orchestrator import (  # noqa: E402
    run_guarded_multi_agent_system,
)

# ============================================================
# FILES
# ============================================================

DATASET = ROOT / "config" / "session18_adversarial_cases.json"

REPORT = ROOT / "reports" / "session18_guardrail_report.json"


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run Session 18 deterministic and optional live AI safety gate."),
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=("Run the real Playwright MCP multi-agent safety test."),
    )

    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8931/mcp",
        help=("Playwright MCP Streamable HTTP endpoint."),
    )

    return parser.parse_args()


# ============================================================
# MCP SESSION
# ============================================================


@asynccontextmanager
async def open_mcp_session(
    mcp_url: str,
) -> AsyncIterator[ClientSession]:
    """
    Open a Streamable HTTP MCP connection.

    The installed MCP SDK returns:

        (
            read_stream,
            write_stream,
            get_session_id,
        )

    ClientSession only needs the first two values.
    """

    async with streamable_http_client(
        mcp_url,
    ) as streams:
        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            # MCP protocol handshake.
            await session.initialize()

            yield session


# ============================================================
# LIVE SAFETY TEST
# ============================================================


async def run_live(
    mcp_url: str,
) -> bool:
    """
    Run a real guarded multi-agent workflow
    against Playwright MCP.

    This is intentionally separate from the
    deterministic adversarial dataset gate.
    """

    todo_name = "Session18-" + uuid4().hex[:6]

    goal = f"Add {todo_name} and mark it complete"

    print()
    print("================================")
    print("LIVE SAFETY EXECUTION")
    print("================================")

    print(
        "MCP URL:",
        mcp_url,
    )

    print(
        "Goal:",
        goal,
    )

    # --------------------------------------------------------
    # OPEN REAL MCP CONNECTION
    # --------------------------------------------------------

    async with open_mcp_session(
        mcp_url,
    ) as client:
        result = await run_guarded_multi_agent_system(
            client=client,
            goal=goal,
        )

    # --------------------------------------------------------
    # BUILD LIVE SAFETY REPORT
    # --------------------------------------------------------

    report = build_runtime_guardrail_report(
        result,
    )

    # --------------------------------------------------------
    # PRINT LIVE REPORT
    # --------------------------------------------------------

    print()
    print("================================")
    print("LIVE SAFETY QUALITY REPORT")
    print("================================")

    print(
        "Input allowed:",
        report.input_allowed,
    )

    print(
        "Handoffs safe:",
        report.handoffs_safe,
    )

    print(
        "Tools safe:",
        report.tools_safe,
    )

    print(
        "Multi-agent completed:",
        report.multi_agent_completed,
    )

    print(
        "Multi-agent quality passed:",
        report.multi_agent_quality_passed,
    )

    print(
        "Findings:",
        report.total_findings,
    )

    print(
        "Safety release:",
        report.safety_release_passed,
    )

    return report.safety_release_passed


# ============================================================
# MAIN QUALITY GATE
# ============================================================


async def async_main() -> int:
    args = parse_args()

    # ========================================================
    # 1. LOAD ADVERSARIAL DATASET
    # ========================================================

    cases = load_adversarial_dataset(
        DATASET,
    )

    # ========================================================
    # 2. RUN DETERMINISTIC ADVERSARIAL EVALUATION
    # ========================================================

    evaluations, report = evaluate_adversarial_dataset(
        cases,
    )

    # ========================================================
    # 3. WRITE JSON REPORT
    # ========================================================

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_adversarial_report(
        evaluations=evaluations,
        report=report,
        path=REPORT,
    )

    # ========================================================
    # 4. PRINT DETERMINISTIC REPORT
    # ========================================================

    print()
    print("================================")
    print("SESSION 18 ADVERSARIAL GATE")
    print("================================")

    print(
        "Total cases:",
        report.total_cases,
    )

    print(
        "Passed:",
        report.passed_cases,
    )

    print(
        "Failed:",
        report.failed_cases,
    )

    print(
        "Attack cases:",
        report.attack_cases,
    )

    print(
        "Blocked attacks:",
        report.blocked_attacks,
    )

    print(
        "Missed attacks:",
        report.missed_attacks,
    )

    print(
        "False positives:",
        report.false_positives,
    )

    print(
        "Detection rate:",
        f"{report.detection_rate:.2%}",
    )

    print(
        "False-positive rate:",
        (f"{report.false_positive_rate:.2%}"),
    )

    # ========================================================
    # 5. PRINT FAILED CASE DETAILS
    # ========================================================

    for evaluation in evaluations:
        if evaluation.passed:
            continue

        print()
        print(
            "FAILED:",
            evaluation.case_id,
        )

        print(
            "Expected allowed:",
            evaluation.expected_allowed,
        )

        print(
            "Actual allowed:",
            evaluation.actual_allowed,
        )

        print(
            "Expected categories:",
            evaluation.expected_categories,
        )

        print(
            "Actual categories:",
            evaluation.actual_categories,
        )

    # ========================================================
    # 6. HARD DETERMINISTIC RELEASE GATE
    # ========================================================

    if not report.release_passed:
        print()
        print("SAFETY RELEASE: FAIL ❌")

        return 1

    print()
    print("ADVERSARIAL GATE: PASS ✅")

    # ========================================================
    # 7. OPTIONAL LIVE MCP GATE
    # ========================================================

    if args.live:
        live_passed = await run_live(
            args.mcp_url,
        )

        if not live_passed:
            print()
            print("LIVE SAFETY RELEASE: FAIL ❌")

            return 1

        print()
        print("LIVE SAFETY RELEASE: PASS ✅")

    # ========================================================
    # ALL QUALITY GATES PASSED
    # ========================================================

    return 0


# ============================================================
# ENTRY POINT
# ============================================================


def main():
    raise SystemExit(
        asyncio.run(
            async_main(),
        )
    )


if __name__ == "__main__":
    main()
