from pathlib import Path

import pytest

from src.browser_agent.dataset_quality import (
    detect_dataset_regressions,
    load_dataset_baseline,
    load_quality_dataset,
    summarize_dataset,
)
from src.browser_agent.dataset_runner import (
    execute_live_quality_dataset,
)

pytestmark = [
    pytest.mark.agent_live,
    pytest.mark.anyio,
]


ROOT = Path(__file__).resolve().parents[2]


DATASET_PATH = ROOT / "config" / "agent_quality_dataset.json"


BASELINE_PATH = ROOT / "config" / "agent_quality_baseline.json"


async def test_live_agent_dataset_release_gate():

    dataset = load_quality_dataset(DATASET_PATH)

    evaluations = await execute_live_quality_dataset(
        dataset=dataset,
    )

    initial_summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
    )

    baseline = load_dataset_baseline(BASELINE_PATH)

    regressions = detect_dataset_regressions(
        summary=initial_summary,
        baseline=baseline,
    )

    summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
        regressions=regressions,
    )

    print("\n==============================")

    print("LIVE DATASET QUALITY REPORT")

    print("==============================")

    print(
        "Pass rate:",
        summary.pass_rate,
    )

    print(
        "Average decision efficiency:",
        summary.average_decision_efficiency,
    )

    print(
        "Execution failures:",
        summary.total_execution_failures,
    )

    print(
        "Stuck loops:",
        summary.stuck_loop_cases,
    )

    print(
        "Regressions:",
        summary.regressions,
    )

    assert summary.release_passed is True, (
        f"Agent dataset release gate failed: {summary.release_failures}"
    )
