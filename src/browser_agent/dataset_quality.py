from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
)

# =========================================================
# CONFIG MODELS
# =========================================================


@dataclass(frozen=True)
class ScenarioThresholds:
    min_progress_score: float = 1.0
    max_executed_steps: int = 2
    min_decision_efficiency: float = 0.30
    max_execution_failures: int = 1
    max_duplicate_decisions: int = 1
    allow_stuck_loop: bool = False


@dataclass(frozen=True)
class AgentScenario:
    id: str
    todo_text: str

    max_steps: int = 5
    max_execution_retries: int = 1

    thresholds: ScenarioThresholds = ScenarioThresholds()


@dataclass(frozen=True)
class DatasetReleaseThresholds:
    min_pass_rate: float = 1.0

    min_average_decision_efficiency: float = 0.30

    max_total_execution_failures: int = 3

    max_stuck_loop_cases: int = 0

    max_regressions: int = 0


@dataclass(frozen=True)
class AgentQualityDataset:
    dataset_name: str

    scenarios: list[AgentScenario]

    release_thresholds: DatasetReleaseThresholds


# =========================================================
# BASELINE
# =========================================================


@dataclass(frozen=True)
class DatasetBaseline:
    dataset_name: str

    min_pass_rate: float

    min_average_decision_efficiency: float

    max_total_execution_failures: int

    max_stuck_loop_cases: int


# =========================================================
# PER-SCENARIO RESULT
# =========================================================


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario_id: str

    goal: str

    passed: bool

    failures: list[str]

    report: AdvancedAgentQualityReport


# =========================================================
# DATASET SUMMARY
# =========================================================


@dataclass(frozen=True)
class DatasetQualitySummary:
    dataset_name: str

    total_scenarios: int

    passed_scenarios: int

    failed_scenarios: int

    pass_rate: float

    average_decision_efficiency: float

    total_execution_failures: int

    total_duplicate_decisions: int

    stuck_loop_cases: int

    regressions: list[str]

    release_failures: list[str]

    release_passed: bool


# =========================================================
# LOAD DATASET
# =========================================================


def load_quality_dataset(
    path: str | Path,
) -> AgentQualityDataset:

    dataset_path = Path(path)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))

    release_raw = raw["release_thresholds"]

    release_thresholds = DatasetReleaseThresholds(
        min_pass_rate=release_raw.get(
            "min_pass_rate",
            1.0,
        ),
        min_average_decision_efficiency=(
            release_raw.get(
                "min_average_decision_efficiency",
                0.30,
            )
        ),
        max_total_execution_failures=(
            release_raw.get(
                "max_total_execution_failures",
                3,
            )
        ),
        max_stuck_loop_cases=(
            release_raw.get(
                "max_stuck_loop_cases",
                0,
            )
        ),
        max_regressions=(
            release_raw.get(
                "max_regressions",
                0,
            )
        ),
    )

    scenarios: list[AgentScenario] = []

    for item in raw["scenarios"]:
        threshold_raw = item.get(
            "thresholds",
            {},
        )

        thresholds = ScenarioThresholds(
            min_progress_score=(
                threshold_raw.get(
                    "min_progress_score",
                    1.0,
                )
            ),
            max_executed_steps=(
                threshold_raw.get(
                    "max_executed_steps",
                    2,
                )
            ),
            min_decision_efficiency=(
                threshold_raw.get(
                    "min_decision_efficiency",
                    0.30,
                )
            ),
            max_execution_failures=(
                threshold_raw.get(
                    "max_execution_failures",
                    1,
                )
            ),
            max_duplicate_decisions=(
                threshold_raw.get(
                    "max_duplicate_decisions",
                    1,
                )
            ),
            allow_stuck_loop=(
                threshold_raw.get(
                    "allow_stuck_loop",
                    False,
                )
            ),
        )

        scenarios.append(
            AgentScenario(
                id=item["id"],
                todo_text=(item["todo_text"]),
                max_steps=item.get(
                    "max_steps",
                    5,
                ),
                max_execution_retries=(
                    item.get(
                        "max_execution_retries",
                        1,
                    )
                ),
                thresholds=thresholds,
            )
        )

    return AgentQualityDataset(
        dataset_name=raw["dataset_name"],
        scenarios=scenarios,
        release_thresholds=(release_thresholds),
    )


# =========================================================
# LOAD BASELINE
# =========================================================


def load_dataset_baseline(
    path: str | Path,
) -> DatasetBaseline:

    baseline_path = Path(path)

    raw = json.loads(baseline_path.read_text(encoding="utf-8"))

    return DatasetBaseline(
        dataset_name=raw["dataset_name"],
        min_pass_rate=raw["min_pass_rate"],
        min_average_decision_efficiency=(raw["min_average_decision_efficiency"]),
        max_total_execution_failures=(raw["max_total_execution_failures"]),
        max_stuck_loop_cases=(raw["max_stuck_loop_cases"]),
    )


# =========================================================
# SCENARIO QUALITY GATE
# =========================================================


def evaluate_scenario_report(
    scenario: AgentScenario,
    goal: str,
    report: AdvancedAgentQualityReport,
) -> ScenarioEvaluation:

    failures: list[str] = []

    thresholds = scenario.thresholds

    # -----------------------------------------------------
    # Task completion
    # -----------------------------------------------------

    if not report.completed:
        failures.append("Task was not completed")

    # -----------------------------------------------------
    # Partial/full completion
    # -----------------------------------------------------

    if report.progress_score < thresholds.min_progress_score:
        failures.append(
            "Progress score below threshold: "
            f"{report.progress_score:.3f} "
            "< "
            f"{thresholds.min_progress_score:.3f}"
        )

    # -----------------------------------------------------
    # Successful browser actions
    # -----------------------------------------------------

    if report.executed_steps > thresholds.max_executed_steps:
        failures.append(
            "Executed steps exceeded threshold: "
            f"{report.executed_steps} "
            "> "
            f"{thresholds.max_executed_steps}"
        )

    # -----------------------------------------------------
    # AI retry/decision efficiency
    # -----------------------------------------------------

    if report.decision_efficiency < thresholds.min_decision_efficiency:
        failures.append(
            "Decision efficiency below threshold: "
            f"{report.decision_efficiency:.3f} "
            "< "
            f"{thresholds.min_decision_efficiency:.3f}"
        )

    # -----------------------------------------------------
    # MCP execution reliability
    # -----------------------------------------------------

    if report.execution_failures > thresholds.max_execution_failures:
        failures.append(
            "Execution failures exceeded threshold: "
            f"{report.execution_failures} "
            "> "
            f"{thresholds.max_execution_failures}"
        )

    # -----------------------------------------------------
    # Duplicate decisions
    # -----------------------------------------------------

    if report.duplicate_decisions > thresholds.max_duplicate_decisions:
        failures.append(
            "Duplicate decisions exceeded threshold: "
            f"{report.duplicate_decisions} "
            "> "
            f"{thresholds.max_duplicate_decisions}"
        )

    # -----------------------------------------------------
    # Stuck loop
    # -----------------------------------------------------

    if report.stuck_loop and not thresholds.allow_stuck_loop:
        failures.append("Agent entered a stuck loop")

    return ScenarioEvaluation(
        scenario_id=scenario.id,
        goal=goal,
        passed=not failures,
        failures=failures,
        report=report,
    )


# =========================================================
# BASELINE REGRESSION
# =========================================================


def detect_dataset_regressions(
    summary: DatasetQualitySummary,
    baseline: DatasetBaseline,
) -> list[str]:

    regressions: list[str] = []

    if summary.dataset_name != baseline.dataset_name:
        regressions.append("Dataset name does not match baseline")

        return regressions

    if summary.pass_rate < baseline.min_pass_rate:
        regressions.append("Dataset pass rate regressed")

    if summary.average_decision_efficiency < baseline.min_average_decision_efficiency:
        regressions.append("Average decision efficiency regressed")

    if summary.total_execution_failures > baseline.max_total_execution_failures:
        regressions.append("Total execution failures regressed")

    if summary.stuck_loop_cases > baseline.max_stuck_loop_cases:
        regressions.append("Stuck-loop count regressed")

    return regressions


# =========================================================
# DATASET SUMMARY
# =========================================================


def summarize_dataset(
    dataset: AgentQualityDataset,
    evaluations: list[ScenarioEvaluation],
    regressions: list[str] | None = None,
) -> DatasetQualitySummary:

    regressions = list(regressions or [])

    total = len(evaluations)

    passed = sum(1 for evaluation in evaluations if evaluation.passed)

    failed = total - passed

    pass_rate = passed / total if total > 0 else 0.0

    efficiencies = [evaluation.report.decision_efficiency for evaluation in evaluations]

    average_efficiency = mean(efficiencies) if efficiencies else 0.0

    total_execution_failures = sum(
        evaluation.report.execution_failures for evaluation in evaluations
    )

    total_duplicate_decisions = sum(
        evaluation.report.duplicate_decisions for evaluation in evaluations
    )

    stuck_loop_cases = sum(
        1 for evaluation in evaluations if evaluation.report.stuck_loop
    )

    thresholds = dataset.release_thresholds

    release_failures: list[str] = []

    # -----------------------------------------------------
    # Dataset pass rate
    # -----------------------------------------------------

    if pass_rate < thresholds.min_pass_rate:
        release_failures.append("Dataset pass rate below release threshold")

    # -----------------------------------------------------
    # Average AI decision efficiency
    # -----------------------------------------------------

    if average_efficiency < thresholds.min_average_decision_efficiency:
        release_failures.append("Average decision efficiency below release threshold")

    # -----------------------------------------------------
    # MCP failures
    # -----------------------------------------------------

    if total_execution_failures > thresholds.max_total_execution_failures:
        release_failures.append(
            "Total MCP execution failures exceeded release threshold"
        )

    # -----------------------------------------------------
    # Stuck loops
    # -----------------------------------------------------

    if stuck_loop_cases > thresholds.max_stuck_loop_cases:
        release_failures.append("Stuck-loop cases exceeded release threshold")

    # -----------------------------------------------------
    # Baseline regressions
    # -----------------------------------------------------

    if len(regressions) > thresholds.max_regressions:
        release_failures.append(
            "Trajectory/dataset regressions exceeded release threshold"
        )

    return DatasetQualitySummary(
        dataset_name=(dataset.dataset_name),
        total_scenarios=total,
        passed_scenarios=passed,
        failed_scenarios=failed,
        pass_rate=pass_rate,
        average_decision_efficiency=(average_efficiency),
        total_execution_failures=(total_execution_failures),
        total_duplicate_decisions=(total_duplicate_decisions),
        stuck_loop_cases=(stuck_loop_cases),
        regressions=regressions,
        release_failures=(release_failures),
        release_passed=(not release_failures),
    )


# =========================================================
# JSON SERIALIZATION
# =========================================================


def build_quality_report_payload(
    summary: DatasetQualitySummary,
    evaluations: list[ScenarioEvaluation],
) -> dict[str, Any]:

    return {
        "generated_at": (datetime.now(timezone.utc).isoformat()),
        "summary": asdict(summary),
        "scenarios": [asdict(evaluation) for evaluation in evaluations],
    }


def write_quality_report(
    summary: DatasetQualitySummary,
    evaluations: list[ScenarioEvaluation],
    output_path: str | Path,
) -> None:

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = build_quality_report_payload(
        summary=summary,
        evaluations=evaluations,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
