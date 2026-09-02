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
from src.observability.quality import (  # noqa: E402
    build_observability_report,
    load_observability_baseline,
    load_observability_thresholds,
    write_observability_baseline,
    write_observability_report,
)
from src.observability.runner import (  # noqa: E402
    run_observed_guarded_system,
)

# =========================================================
# MCP SESSION
# =========================================================


@asynccontextmanager
async def open_mcp_session(
    mcp_url: str,
) -> AsyncIterator[ClientSession]:
    """
    Open Playwright MCP using the current MCP SDK.

    Replaces the old:

        async with Client(url)

    with:

        streamable_http_client
            ↓
        ClientSession
            ↓
        initialize()
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
            await session.initialize()

            yield session


# =========================================================
# OBSERVABILITY FLOW
# =========================================================


async def go(
    args,
) -> int:

    # -----------------------------------------------------
    # LOAD THRESHOLDS
    # -----------------------------------------------------

    thresholds = load_observability_thresholds(
        ROOT / "config" / "session19_observability_thresholds.json"
    )

    # -----------------------------------------------------
    # LOAD BASELINE
    # -----------------------------------------------------

    baseline = load_observability_baseline(
        ROOT / "config" / "session19_observability_baseline.json"
    )

    # -----------------------------------------------------
    # GENERATE TEST GOAL
    # -----------------------------------------------------

    todo_name = "Session19-" + uuid4().hex[:6]

    goal = f"Add {todo_name} and mark it complete"

    print()
    print("=================================")
    print("SESSION 19 OBSERVABILITY RUN")
    print("=================================")

    print(
        "Goal:",
        goal,
    )

    print(
        "MCP URL:",
        args.mcp_url,
    )

    # -----------------------------------------------------
    # RUN REAL MCP WORKFLOW
    # -----------------------------------------------------

    async with open_mcp_session(
        args.mcp_url,
    ) as client:
        observed_run = await run_observed_guarded_system(
            client,
            goal,
        )

    # -----------------------------------------------------
    # BUILD OBSERVABILITY REPORT
    # -----------------------------------------------------

    report = build_observability_report(
        observed_run,
        thresholds,
        baseline,
    )

    # -----------------------------------------------------
    # ENSURE REPORT DIRECTORY EXISTS
    # -----------------------------------------------------

    report_path = Path(args.report)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # WRITE REPORT
    # -----------------------------------------------------

    write_observability_report(
        run=observed_run,
        report=report,
        path=report_path,
    )

    # -----------------------------------------------------
    # PRINT REPORT
    # -----------------------------------------------------

    print()
    print("=================================")
    print("SESSION 19 OBSERVABILITY")
    print("=================================")

    print(
        "Trace:",
        report.trace_id,
    )

    print(
        "Total latency ms:",
        round(
            report.total_latency_ms,
            2,
        ),
    )

    print(
        "MCP calls:",
        report.mcp_call_count,
    )

    print(
        "Retry overhead:",
        f"{report.retry_overhead_ratio:.2%}",
    )

    print(
        "Usage available:",
        report.usage_metrics_available,
    )

    print(
        "Release:",
        ("PASS ✅" if report.release_passed else "FAIL ❌"),
    )

    # -----------------------------------------------------
    # OPTIONAL BASELINE UPDATE
    # -----------------------------------------------------

    if args.update_baseline and report.release_passed:
        baseline_path = ROOT / "config" / "session19_observability_baseline.json"

        write_observability_baseline(
            report=report,
            path=baseline_path,
        )

        print(
            "Baseline updated:",
            baseline_path,
        )

    # -----------------------------------------------------
    # RELEASE RESULT
    # -----------------------------------------------------

    return 0 if report.release_passed else 1


# =========================================================
# CLI
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run Session 19 observability quality gate."),
    )

    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8931/mcp",
        help=("Playwright MCP Streamable HTTP endpoint."),
    )

    parser.add_argument(
        "--report",
        default=str(ROOT / "reports" / "session19_observability_report.json"),
        help="Output observability report path.",
    )

    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=("Update observability baseline only when the quality gate passes."),
    )

    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================


def main():
    args = parse_args()

    exit_code = asyncio.run(
        go(
            args,
        )
    )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
