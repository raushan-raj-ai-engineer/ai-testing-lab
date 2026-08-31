from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from mcp import Client

from src.guardrails.quality import (
    build_runtime_guardrail_report,
    evaluate_adversarial_dataset,
    load_adversarial_dataset,
    write_adversarial_report,
)
from src.guardrails.safe_orchestrator import (
    run_guarded_multi_agent_system,
)

DATASET = ROOT / "config" / "session18_adversarial_cases.json"


REPORT = ROOT / "reports" / "session18_guardrail_report.json"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--live",
        action="store_true",
    )

    parser.add_argument(
        "--mcp-url",
        default=("http://localhost:8931/mcp"),
    )

    return parser.parse_args()


async def run_live(
    mcp_url: str,
) -> bool:

    todo_name = "Session18-" + uuid4().hex[:6]

    goal = f"Add {todo_name} and mark it complete"

    async with Client(mcp_url) as client:
        result = await run_guarded_multi_agent_system(
            client=client,
            goal=goal,
        )

    report = build_runtime_guardrail_report(result)

    print("\n================================")

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


async def async_main() -> int:

    args = parse_args()

    cases = load_adversarial_dataset(DATASET)

    evaluations, report = evaluate_adversarial_dataset(cases)

    write_adversarial_report(
        evaluations=evaluations,
        report=report,
        path=REPORT,
    )

    print("\n================================")

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
        f"{report.false_positive_rate:.2%}",
    )

    for evaluation in evaluations:
        if not evaluation.passed:
            print(
                "\nFAILED:",
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

    if not report.release_passed:
        print("\nSAFETY RELEASE: FAIL ❌")

        return 1

    print("\nADVERSARIAL GATE: PASS ✅")

    if args.live:
        live_passed = await run_live(args.mcp_url)

        if not live_passed:
            print("\nLIVE SAFETY RELEASE: FAIL ❌")

            return 1

        print("\nLIVE SAFETY RELEASE: PASS ✅")

    return 0


def main():

    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
