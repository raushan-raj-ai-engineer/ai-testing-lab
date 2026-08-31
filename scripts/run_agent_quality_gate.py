from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.browser_agent.dataset_quality import (
    detect_dataset_regressions,
    load_dataset_baseline,
    load_quality_dataset,
    summarize_dataset,
    write_quality_report,
)
from src.browser_agent.dataset_runner import (
    execute_live_quality_dataset,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run browser-agent golden dataset and enforce release quality gates."
        )
    )

    parser.add_argument(
        "--dataset",
        default=("config/agent_quality_dataset.json"),
    )

    parser.add_argument(
        "--baseline",
        default=("config/agent_quality_baseline.json"),
    )

    parser.add_argument(
        "--report",
        default=("reports/agent_quality_report.json"),
    )

    parser.add_argument(
        "--mcp-url",
        default=("http://localhost:8931/mcp"),
    )

    return parser.parse_args()


async def async_main() -> int:

    args = parse_args()

    dataset = load_quality_dataset(args.dataset)

    print("\n======================================")

    print("AGENT QUALITY DATASET")

    print("======================================")

    print(
        "Dataset:",
        dataset.dataset_name,
    )

    print(
        "Scenarios:",
        len(dataset.scenarios),
    )

    # =====================================================
    # Execute real golden dataset
    # =====================================================

    evaluations = await execute_live_quality_dataset(
        dataset=dataset,
        mcp_url=args.mcp_url,
    )

    # =====================================================
    # First summary
    # =====================================================

    initial_summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
    )

    # =====================================================
    # Baseline regression
    # =====================================================

    baseline_path = Path(args.baseline)

    regressions: list[str] = []

    if baseline_path.exists():
        baseline = load_dataset_baseline(baseline_path)

        regressions = detect_dataset_regressions(
            summary=initial_summary,
            baseline=baseline,
        )

    # =====================================================
    # Final summary including regression gate
    # =====================================================

    summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
        regressions=regressions,
    )

    # =====================================================
    # JSON report
    # =====================================================

    write_quality_report(
        summary=summary,
        evaluations=evaluations,
        output_path=args.report,
    )

    # =====================================================
    # Console report
    # =====================================================

    print("\n======================================")

    print("AGENT QUALITY RELEASE REPORT")

    print("======================================")

    print(
        "Total scenarios:",
        summary.total_scenarios,
    )

    print(
        "Passed scenarios:",
        summary.passed_scenarios,
    )

    print(
        "Failed scenarios:",
        summary.failed_scenarios,
    )

    print(
        "Pass rate:",
        f"{summary.pass_rate:.2%}",
    )

    print(
        "Average decision efficiency:",
        round(
            summary.average_decision_efficiency,
            3,
        ),
    )

    print(
        "Total execution failures:",
        summary.total_execution_failures,
    )

    print(
        "Duplicate decisions:",
        summary.total_duplicate_decisions,
    )

    print(
        "Stuck-loop cases:",
        summary.stuck_loop_cases,
    )

    print(
        "Regressions:",
        len(summary.regressions),
    )

    # =====================================================
    # Failed scenarios
    # =====================================================

    for evaluation in evaluations:
        if evaluation.passed:
            continue

        print(
            "\nFAILED SCENARIO:",
            evaluation.scenario_id,
        )

        for failure in evaluation.failures:
            print(
                " -",
                failure,
            )

    # =====================================================
    # Regressions
    # =====================================================

    if summary.regressions:
        print("\nREGRESSIONS:")

        for regression in summary.regressions:
            print(
                " -",
                regression,
            )

    # =====================================================
    # Release failures
    # =====================================================

    if summary.release_failures:
        print("\nRELEASE GATE FAILURES:")

        for failure in summary.release_failures:
            print(
                " -",
                failure,
            )

    print(
        "\nJSON report:",
        args.report,
    )

    # =====================================================
    # CI exit code
    # =====================================================

    if summary.release_passed:
        print("\nRELEASE DECISION: PASS ✅")

        return 0

    print("\nRELEASE DECISION: FAIL ❌")

    return 1


def main():

    try:
        exit_code = asyncio.run(async_main())

    except KeyboardInterrupt:
        exit_code = 130

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
