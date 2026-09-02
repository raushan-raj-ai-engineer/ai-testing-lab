from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

# =========================================================
# PROJECT ROOT
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# =========================================================
# MCP IMPORTS
# =========================================================

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import (  # noqa: E402
    streamable_http_client,
)

# =========================================================
# PROJECT IMPORTS
# =========================================================
from src.multi_agent.orchestrator import (  # noqa: E402
    run_multi_agent_system,
)
from src.multi_agent.quality import (  # noqa: E402
    build_multi_agent_quality_report,
    passes_multi_agent_quality_gate,
)

# =========================================================
# ARGUMENTS
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run Session 17 multi-agent browser quality flow")
    )

    parser.add_argument(
        "--goal",
        default=None,
        help=("Optional multi-agent goal. A TodoMVC goal is generated if omitted."),
    )

    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8931/mcp",
        help=("Playwright MCP Streamable HTTP endpoint."),
    )

    return parser.parse_args()


# =========================================================
# MCP CONNECTION
# =========================================================


@asynccontextmanager
async def open_mcp_session(
    mcp_url: str,
) -> AsyncIterator[ClientSession]:
    """
    Open and initialize an MCP Streamable HTTP session.

    New MCP SDK versions use:

        streamable_http_client
                ↓
        ClientSession
                ↓
        initialize()

    instead of the old:

        Client(url)
    """

    async with streamable_http_client(
        mcp_url,
    ) as streams:
        # Current MCP Streamable HTTP transport may return
        # more than two items.
        #
        # ClientSession only needs:
        #
        # streams[0] -> read stream
        # streams[1] -> write stream

        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            yield session


# =========================================================
# MAIN
# =========================================================


async def async_main() -> int:
    args = parse_args()

    goal = args.goal

    # =====================================================
    # GENERATE DEFAULT GOAL
    # =====================================================

    if goal is None:
        todo_name = "Session17-" + uuid4().hex[:6]

        goal = f"Add {todo_name} and mark it complete"

    # =====================================================
    # PRINT RUN INFORMATION
    # =====================================================

    print()
    print("=================================")
    print("SESSION 17 MULTI-AGENT RUN")
    print("=================================")

    print(
        "Goal:",
        goal,
    )

    print(
        "MCP URL:",
        args.mcp_url,
    )

    # =====================================================
    # CONNECT TO PLAYWRIGHT MCP
    # =====================================================

    async with open_mcp_session(
        args.mcp_url,
    ) as client:
        result = await run_multi_agent_system(
            client=client,
            goal=goal,
        )

    # =====================================================
    # BUILD QUALITY REPORT
    # =====================================================

    report = build_multi_agent_quality_report(
        result,
    )

    # =====================================================
    # PRINT QUALITY REPORT
    # =====================================================

    print()
    print("=================================")
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
        (f"{report.task_completion_rate:.2%}"),
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
        (f"{report.handoff_success_rate:.2%}"),
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

    # =====================================================
    # ROUTING ERRORS
    # =====================================================

    if result.routing_errors:
        print()
        print("ROUTING ERRORS:")

        for error in result.routing_errors:
            print(
                " -",
                error,
            )

    # =====================================================
    # EXECUTION ERRORS
    # =====================================================

    if result.errors:
        print()
        print("EXECUTION ERRORS:")

        for error in result.errors:
            print(
                " -",
                error,
            )

    # =====================================================
    # RELEASE QUALITY GATE
    # =====================================================

    if passes_multi_agent_quality_gate(
        report,
    ):
        print()
        print("MULTI-AGENT RELEASE: PASS ✅")

        return 0

    print()
    print("MULTI-AGENT RELEASE: FAIL ❌")

    return 1


# =========================================================
# ENTRY POINT
# =========================================================


def main():
    exit_code = asyncio.run(async_main())

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
