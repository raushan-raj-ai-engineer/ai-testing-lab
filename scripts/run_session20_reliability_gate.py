from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
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
)
from src.observability.runner import (  # noqa: E402
    run_observed_guarded_system,
)
from src.reliability.quality import (  # noqa: E402
    build_reliability_report,
    load_reliability_thresholds,
)
from src.reliability.runner import (  # noqa: E402
    run_repeated_async,
)

# =========================================================
# MCP SESSION
# =========================================================


@asynccontextmanager
async def open_mcp_session(
    mcp_url: str,
) -> AsyncIterator[ClientSession]:
    """
    Open and initialize Playwright MCP using
    the current MCP Python SDK.

    Replaces the old:

        async with Client(url)

    pattern.
    """

    async with streamable_http_client(
        mcp_url,
    ) as streams:
        # Current transport may return 3 values.
        # ClientSession needs only read/write streams.

        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            yield session


# =========================================================
# RELIABILITY FLOW
# =========================================================


async def go(
    args,
) -> int:

    # -----------------------------------------------------
    # LOAD OBSERVABILITY CONFIG
    # -----------------------------------------------------

    observability_thresholds = load_observability_thresholds(
        ROOT / "config" / "session19_observability_thresholds.json"
    )

    observability_baseline = load_observability_baseline(
        ROOT / "config" / "session19_observability_baseline.json"
    )

    # -----------------------------------------------------
    # LOAD RELIABILITY CONFIG
    # -----------------------------------------------------

    reliability_thresholds = load_reliability_thresholds(
        ROOT / "config" / "session20_reliability_thresholds.json"
    )

    # -----------------------------------------------------
    # PRINT RUN INFO
    # -----------------------------------------------------

    print()
    print("=================================")
    print("SESSION 20 RELIABILITY RUN")
    print("=================================")

    print(
        "Runs:",
        args.runs,
    )

    print(
        "MCP URL:",
        args.mcp_url,
    )

    # -----------------------------------------------------
    # OPEN REAL MCP SESSION
    # -----------------------------------------------------

    async with open_mcp_session(
        args.mcp_url,
    ) as client:
        # -------------------------------------------------
        # ONE RELIABILITY ITERATION
        # -------------------------------------------------

        async def operation():

            todo_name = "Session20-" + uuid4().hex[:6]

            goal = f"Add {todo_name} and mark it complete"

            observed_run = await run_observed_guarded_system(
                client,
                goal,
            )

            observability_report = build_observability_report(
                observed_run,
                observability_thresholds,
                observability_baseline,
            )

            return (
                observed_run,
                observability_report,
            )

        # -------------------------------------------------
        # RUN MULTIPLE ITERATIONS
        # -------------------------------------------------

        runs = await run_repeated_async(
            operation,
            runs=args.runs,
            success=lambda value: value[1].release_passed,
            telemetry=lambda value: {
                "decision_attempts": (value[0].decision_attempts),
                "rejected_decisions": (value[0].rejected_decisions),
                "execution_attempts": (value[0].execution_attempts),
                "execution_failures": (value[0].execution_failures),
            },
        )

    # =====================================================
    # BUILD RELIABILITY REPORT
    # =====================================================

    report = build_reliability_report(
        runs,
        reliability_thresholds,
    )

    # =====================================================
    # WRITE REPORT
    # =====================================================

    output_path = ROOT / "reports" / "session20_reliability_report.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            {
                "report": asdict(
                    report,
                ),
                "runs": [asdict(run) for run in runs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # PRINT QUALITY REPORT
    # =====================================================

    print()
    print("=================================")
    print("SESSION 20 RELIABILITY")
    print("=================================")

    print(
        "Pass rate:",
        f"{report.pass_rate:.2%}",
    )

    print(
        "Flake rate:",
        f"{report.flake_rate:.2%}",
    )

    print(
        "Latency CV:",
        round(
            report.latency_cv,
            3,
        ),
    )

    print(
        "Report:",
        output_path,
    )

    print(
        "Release:",
        ("PASS ✅" if report.release_passed else "FAIL ❌"),
    )

    # =====================================================
    # RELEASE RESULT
    # =====================================================

    return 0 if report.release_passed else 1


# =========================================================
# CLI
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run Session 20 reliability quality gate."),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help=("Number of repeated executions used for reliability measurement."),
    )

    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8931/mcp",
        help=("Playwright MCP Streamable HTTP endpoint."),
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
