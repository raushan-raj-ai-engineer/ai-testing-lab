from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from mcp import Client

from src.multi_agent.orchestrator import (
    run_multi_agent_system,
)
from src.multi_agent.quality import (
    build_multi_agent_quality_report,
    passes_multi_agent_quality_gate,
)

# =========================================================
# PROJECT ROOT
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=("Run Session 17 multi-agent browser quality flow")
    )

    parser.add_argument(
        "--goal",
        default=None,
    )

    parser.add_argument(
        "--mcp-url",
        default=("http://localhost:8931/mcp"),
    )

    return parser.parse_args()


async def async_main() -> int:

    args = parse_args()

    goal = args.goal

    if goal is None:
        todo_name = "Session17-" + uuid4().hex[:6]

        goal = f"Add {todo_name} and mark it complete"

    print("\n=================================")

    print("SESSION 17 MULTI-AGENT RUN")

    print("=================================")

    print(
        "Goal:",
        goal,
    )

    async with Client(args.mcp_url) as client:
        result = await run_multi_agent_system(
            client=client,
            goal=goal,
        )

    report = build_multi_agent_quality_report(result)

    print("\n=================================")

    print("MULTI-AGENT QUALITY REPORT")

    print("=================================")

    print(
        "Plan:",
        result.plan.id,
    )

    print(
        "Total tasks:",
        report.total_tasks,
    )

    print(
        "Completed tasks:",
        report.completed_tasks,
    )

    print(
        "Task completion rate:",
        f"{report.task_completion_rate:.2%}",
    )

    print(
        "Expected handoffs:",
        report.expected_handoffs,
    )

    print(
        "Actual handoffs:",
        report.actual_handoffs,
    )

    print(
        "Handoff success rate:",
        f"{report.handoff_success_rate:.2%}",
    )

    print(
        "Context preserved:",
        report.context_preserved,
    )

    print(
        "Routing errors:",
        report.routing_errors,
    )

    print(
        "Execution errors:",
        report.execution_errors,
    )

    print(
        "Handoff loop:",
        report.handoff_loop,
    )

    print(
        "Browser quality passed:",
        report.browser_quality_passed,
    )

    if result.routing_errors:
        print("\nROUTING ERRORS:")

        for error in result.routing_errors:
            print(
                " -",
                error,
            )

    if result.errors:
        print("\nEXECUTION ERRORS:")

        for error in result.errors:
            print(
                " -",
                error,
            )

    if passes_multi_agent_quality_gate(report):
        print("\nMULTI-AGENT RELEASE: PASS ✅")

        return 0

    print("\nMULTI-AGENT RELEASE: FAIL ❌")

    return 1


def main():

    exit_code = asyncio.run(async_main())

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
