from pathlib import Path

import pytest

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
)
from src.browser_agent.dataset_quality import (
    AgentScenario,
    DatasetBaseline,
    ScenarioThresholds,
    detect_dataset_regressions,
    evaluate_scenario_report,
    load_quality_dataset,
    summarize_dataset,
)

pytestmark = pytest.mark.agent_quality


ROOT = Path(__file__).resolve().parents[2]


DATASET_PATH = ROOT / "config" / "agent_quality_dataset.json"


def good_report(
    decision_efficiency: float = 1.0,
) -> AdvancedAgentQualityReport:

    return AdvancedAgentQualityReport(
        completed=True,
        progress_status=("completed"),
        progress_score=1.0,
        executed_steps=2,
        decision_attempts=2,
        rejected_attempts=0,
        decision_efficiency=(decision_efficiency),
        execution_attempts=2,
        execution_failures=0,
        duplicate_decisions=0,
        stuck_loop=False,
    )


def bad_report() -> AdvancedAgentQualityReport:

    return AdvancedAgentQualityReport(
        completed=False,
        progress_status="partial",
        progress_score=0.5,
        executed_steps=4,
        decision_attempts=6,
        rejected_attempts=5,
        decision_efficiency=(1 / 6),
        execution_attempts=4,
        execution_failures=2,
        duplicate_decisions=3,
        stuck_loop=True,
    )


# =========================================================
# DATASET LOAD
# =========================================================


def test_agent_quality_dataset_loads():

    dataset = load_quality_dataset(DATASET_PATH)

    assert dataset.dataset_name == "todomvc_browser_agent_v1"

    assert len(dataset.scenarios) == 3


# =========================================================
# GOOD SCENARIO
# =========================================================


def test_good_scenario_passes():

    scenario = AgentScenario(
        id="good",
        todo_text="Buy milk",
        thresholds=(ScenarioThresholds()),
    )

    evaluation = evaluate_scenario_report(
        scenario=scenario,
        goal=("Add Buy milk and mark it complete"),
        report=good_report(),
    )

    assert evaluation.passed is True

    assert evaluation.failures == []


# =========================================================
# BAD SCENARIO
# =========================================================


def test_bad_scenario_fails():

    scenario = AgentScenario(
        id="bad",
        todo_text="Buy milk",
        thresholds=(ScenarioThresholds()),
    )

    evaluation = evaluate_scenario_report(
        scenario=scenario,
        goal=("Add Buy milk and mark it complete"),
        report=bad_report(),
    )

    assert evaluation.passed is False

    assert len(evaluation.failures) >= 5


# =========================================================
# DATASET PASS RATE
# =========================================================


def test_dataset_summary_passes():

    dataset = load_quality_dataset(DATASET_PATH)

    evaluations = []

    for scenario in dataset.scenarios:
        evaluations.append(
            evaluate_scenario_report(
                scenario=scenario,
                goal=(f"Add {scenario.todo_text} and mark it complete"),
                report=good_report(
                    decision_efficiency=0.8,
                ),
            )
        )

    summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
    )

    assert summary.total_scenarios == 3

    assert summary.passed_scenarios == 3

    assert summary.failed_scenarios == 0

    assert summary.pass_rate == 1.0

    assert summary.release_passed is True


# =========================================================
# DATASET RELEASE FAILURE
# =========================================================


def test_dataset_release_fails_when_case_fails():

    dataset = load_quality_dataset(DATASET_PATH)

    evaluations = []

    for index, scenario in enumerate(dataset.scenarios):
        report = bad_report() if index == 0 else good_report()

        evaluations.append(
            evaluate_scenario_report(
                scenario=scenario,
                goal=(f"Add {scenario.todo_text} and mark it complete"),
                report=report,
            )
        )

    summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
    )

    assert summary.pass_rate < 1.0

    assert summary.release_passed is False


# =========================================================
# BASELINE REGRESSION
# =========================================================


def test_dataset_regression_detected():

    dataset = load_quality_dataset(DATASET_PATH)

    evaluations = []

    for scenario in dataset.scenarios:
        evaluations.append(
            evaluate_scenario_report(
                scenario=scenario,
                goal=(f"Add {scenario.todo_text} and mark it complete"),
                report=good_report(
                    decision_efficiency=0.4,
                ),
            )
        )

    summary = summarize_dataset(
        dataset=dataset,
        evaluations=evaluations,
    )

    baseline = DatasetBaseline(
        dataset_name=(dataset.dataset_name),
        min_pass_rate=1.0,
        # Deliberately higher
        min_average_decision_efficiency=(0.8),
        max_total_execution_failures=0,
        max_stuck_loop_cases=0,
    )

    regressions = detect_dataset_regressions(
        summary=summary,
        baseline=baseline,
    )

    assert "Average decision efficiency regressed" in regressions


def test_failed_scenario_report_fails_release():

    scenario = AgentScenario(
        id="broken",
        todo_text="Broken task",
        thresholds=(ScenarioThresholds()),
    )

    report = AdvancedAgentQualityReport(
        completed=False,
        progress_status="error",
        progress_score=0.0,
        executed_steps=0,
        decision_attempts=0,
        rejected_attempts=0,
        decision_efficiency=0.0,
        execution_attempts=0,
        execution_failures=1,
        duplicate_decisions=0,
        stuck_loop=False,
    )

    evaluation = evaluate_scenario_report(
        scenario=scenario,
        goal=("Add Broken task and mark it complete"),
        report=report,
    )

    assert evaluation.passed is False

    assert "Task was not completed" in evaluation.failures
