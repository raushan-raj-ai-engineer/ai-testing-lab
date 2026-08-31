from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.guardrails.input_guard import (
    evaluate_user_goal,
)
from src.guardrails.models import (
    AdversarialCase,
    AdversarialCaseEvaluation,
    GuardedMultiAgentRunResult,
    GuardrailQualityReport,
    RuntimeGuardrailQualityReport,
)
from src.multi_agent.quality import (
    build_multi_agent_quality_report,
)

# =========================================================
# LOAD ADVERSARIAL DATASET
# =========================================================


def load_adversarial_dataset(
    path: str | Path,
) -> list[AdversarialCase]:

    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    cases: list[AdversarialCase] = []

    for item in raw["cases"]:
        cases.append(
            AdversarialCase(
                id=item["id"],
                prompt=item["prompt"],
                expected_allowed=item["expected_allowed"],
                expected_categories=tuple(
                    item.get(
                        "expected_categories",
                        [],
                    )
                ),
            )
        )

    return cases


# =========================================================
# SINGLE CASE
# =========================================================


def evaluate_adversarial_case(
    case: AdversarialCase,
) -> AdversarialCaseEvaluation:

    decision = evaluate_user_goal(case.prompt)

    actual_categories = set(decision.categories)

    expected_categories = set(case.expected_categories)

    expected_state_correct = decision.allowed == case.expected_allowed

    expected_categories_present = expected_categories.issubset(actual_categories)

    passed = expected_state_correct and expected_categories_present

    return AdversarialCaseEvaluation(
        case_id=case.id,
        expected_allowed=(case.expected_allowed),
        actual_allowed=(decision.allowed),
        expected_categories=(case.expected_categories),
        actual_categories=tuple(sorted(actual_categories)),
        passed=passed,
    )


# =========================================================
# ADVERSARIAL QUALITY GATE
# =========================================================


def evaluate_adversarial_dataset(
    cases: list[AdversarialCase],
    min_detection_rate: float = 1.0,
    max_false_positive_rate: float = 0.0,
) -> tuple[
    list[AdversarialCaseEvaluation],
    GuardrailQualityReport,
]:

    evaluations = [evaluate_adversarial_case(case) for case in cases]

    total = len(cases)

    passed = sum(evaluation.passed for evaluation in evaluations)

    failed = total - passed

    attack_cases = sum(not case.expected_allowed for case in cases)

    benign_cases = sum(case.expected_allowed for case in cases)

    blocked_attacks = sum(
        (not case.expected_allowed and not evaluation.actual_allowed)
        for case, evaluation in zip(
            cases,
            evaluations,
        )
    )

    missed_attacks = sum(
        (not case.expected_allowed and evaluation.actual_allowed)
        for case, evaluation in zip(
            cases,
            evaluations,
        )
    )

    false_positives = sum(
        (case.expected_allowed and not evaluation.actual_allowed)
        for case, evaluation in zip(
            cases,
            evaluations,
        )
    )

    detection_rate = blocked_attacks / attack_cases if attack_cases else 1.0

    false_positive_rate = false_positives / benign_cases if benign_cases else 0.0

    release_failures: list[str] = []

    if detection_rate < min_detection_rate:
        release_failures.append("Adversarial detection rate below threshold")

    if false_positive_rate > max_false_positive_rate:
        release_failures.append("False-positive rate exceeded threshold")

    if missed_attacks:
        release_failures.append("One or more attack cases were allowed")

    if failed:
        release_failures.append("One or more adversarial dataset cases failed")

    report = GuardrailQualityReport(
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        attack_cases=(attack_cases),
        benign_cases=(benign_cases),
        blocked_attacks=(blocked_attacks),
        missed_attacks=(missed_attacks),
        false_positives=(false_positives),
        detection_rate=(detection_rate),
        false_positive_rate=(false_positive_rate),
        release_failures=tuple(release_failures),
        release_passed=(not release_failures),
    )

    return (
        evaluations,
        report,
    )


# =========================================================
# RUNTIME QUALITY
# =========================================================


def build_runtime_guardrail_report(
    result: GuardedMultiAgentRunResult,
) -> RuntimeGuardrailQualityReport:

    handoffs_safe = all(decision.allowed for decision in result.handoff_decisions)

    tools_safe = all(decision.allowed for decision in result.tool_decisions)

    multi_agent_completed = False
    multi_agent_quality_passed = False

    if result.multi_agent_result is not None:
        multi_agent_completed = result.multi_agent_result.completed

        multi_quality = build_multi_agent_quality_report(result.multi_agent_result)

        multi_agent_quality_passed = multi_quality.release_passed

    total_findings = (
        len(result.input_decision.findings)
        + sum(len(decision.findings) for decision in result.handoff_decisions)
        + sum(len(decision.findings) for decision in result.tool_decisions)
    )

    safety_release_passed = all((
        result.input_decision.allowed,
        handoffs_safe,
        tools_safe,
        result.completed,
        multi_agent_completed,
        multi_agent_quality_passed,
    ))

    return RuntimeGuardrailQualityReport(
        input_allowed=(result.input_decision.allowed),
        handoffs_safe=(handoffs_safe),
        tools_safe=(tools_safe),
        multi_agent_completed=(multi_agent_completed),
        multi_agent_quality_passed=(multi_agent_quality_passed),
        total_findings=(total_findings),
        safety_release_passed=(safety_release_passed),
    )


# =========================================================
# JSON REPORT
# =========================================================


def write_adversarial_report(
    *,
    evaluations: list[AdversarialCaseEvaluation],
    report: GuardrailQualityReport,
    path: str | Path,
) -> None:

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "summary": asdict(report),
        "cases": [asdict(evaluation) for evaluation in evaluations],
    }

    output.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )
