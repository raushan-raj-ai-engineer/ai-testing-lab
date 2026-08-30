from dataclasses import dataclass, field

from deepeval.test_case import LLMTestCase

from src.llm_client import classify_ticket
from src.metrics import reason_correctness


@dataclass
class EvaluationResult:
    total: int
    passed: int
    failed: int
    accuracy: float


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str]


@dataclass
class CaseEvaluation:
    case_id: str

    deterministic_passed: bool

    deterministic_failures: list[str] = field(default_factory=list)

    actual_category: str | None = None
    actual_priority: str | None = None
    actual_needs_human: bool | None = None
    actual_reason: str | None = None

    semantic_score: float | None = None
    semantic_passed: bool | None = None
    semantic_reason: str | None = None


@dataclass
class DatasetEvaluation:
    total_cases: int
    deterministic_pass_rate: float
    semantic_cases_evaluated: int
    semantic_pass_rate: float
    average_semantic_score: float
    cases: list[CaseEvaluation]


def evaluate_case(
    case: dict,
) -> CaseEvaluation:

    result = classify_ticket(case["ticket"])

    (
        deterministic_passed,
        deterministic_failures,
    ) = evaluate_deterministic(
        case,
        result,
    )

    semantic_score = None
    semantic_passed = None
    semantic_reason = None

    if "expected_reason" in case:
        (
            semantic_score,
            semantic_passed,
            semantic_reason,
        ) = evaluate_semantic(
            case,
            result,
        )

    return CaseEvaluation(
        case_id=case["id"],
        deterministic_passed=deterministic_passed,
        deterministic_failures=deterministic_failures,
        actual_category=result.category,
        actual_priority=result.priority,
        actual_needs_human=result.needs_human,
        actual_reason=result.reason,
        semantic_score=semantic_score,
        semantic_passed=semantic_passed,
        semantic_reason=semantic_reason,
    )


def evaluate_dataset(
    dataset: list[dict],
) -> DatasetEvaluation:

    results = [evaluate_case(case) for case in dataset]

    total = len(results)

    deterministic_passed = sum(result.deterministic_passed for result in results)

    semantic_results = [
        result for result in results if result.semantic_passed is not None
    ]

    semantic_passed = sum(result.semantic_passed is True for result in semantic_results)

    semantic_scores = [
        result.semantic_score
        for result in semantic_results
        if result.semantic_score is not None
    ]

    deterministic_pass_rate = deterministic_passed / total if total else 0

    semantic_pass_rate = (
        semantic_passed / len(semantic_results) if semantic_results else 0
    )

    average_semantic_score = (
        sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    )

    return DatasetEvaluation(
        total_cases=total,
        deterministic_pass_rate=deterministic_pass_rate,
        semantic_cases_evaluated=len(semantic_results),
        semantic_pass_rate=semantic_pass_rate,
        average_semantic_score=average_semantic_score,
        cases=results,
    )


def evaluate_deterministic(
    case: dict,
    result,
) -> tuple[bool, list[str]]:

    failures = []

    if result.category != case["expected_category"]:
        failures.append("category")

    if "expected_priority" in case and result.priority != case["expected_priority"]:
        failures.append("priority")

    if (
        "expected_needs_human" in case
        and result.needs_human != case["expected_needs_human"]
    ):
        failures.append("needs_human")

    return (
        len(failures) == 0,
        failures,
    )


def evaluate_semantic(
    case: dict,
    result,
) -> tuple[float, bool, str | None]:

    test_case = LLMTestCase(
        input=case["ticket"],
        actual_output=result.reason,
        expected_output=case["expected_reason"],
    )

    reason_correctness.measure(test_case)

    score = float(reason_correctness.score or 0)

    passed = score >= 0.8

    return (
        score,
        passed,
        reason_correctness.reason,
    )
