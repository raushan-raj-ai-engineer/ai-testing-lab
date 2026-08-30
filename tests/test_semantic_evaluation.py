from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from src.llm_client import classify_ticket
from src.metrics import (
    REASON_CORRECTNESS_THRESHOLD,
    build_reason_correctness_metric,
)


def test_real_classifier_reason():

    ticket = "My credit card was charged twice for the same order."

    result = classify_ticket(ticket)

    test_case = LLMTestCase(
        input=ticket,
        actual_output=result.reason,
        expected_output=("The customer has been charged twice for the same purchase."),
    )

    metric = build_reason_correctness_metric()

    assert_test(
        test_case,
        [metric],
    )


def test_correct_paraphrase_passes():

    test_case = LLMTestCase(
        input=("My credit card was charged twice for the same order."),
        actual_output=(
            "The customer appears to have been billed twice for one purchase."
        ),
        expected_output=("The customer reports a duplicate charge."),
    )

    metric = build_reason_correctness_metric()

    assert_test(
        test_case,
        [metric],
    )


def test_wrong_reason_is_rejected():

    test_case = LLMTestCase(
        input=("My credit card was charged twice for the same order."),
        actual_output=(
            "The customer cannot access their account because the password has expired."
        ),
        expected_output=("The customer reports a duplicate charge."),
    )

    metric = build_reason_correctness_metric()

    metric.measure(test_case)

    score = metric.score

    assert score is not None, "Semantic evaluator did not return a score."

    print("\nNegative semantic test")

    print(
        "Score:",
        score,
    )

    print(
        "Reason:",
        metric.reason,
    )

    assert score < REASON_CORRECTNESS_THRESHOLD, (
        "Evaluator incorrectly accepted "
        "a semantically wrong response. "
        f"Score: {score}, "
        f"Threshold: "
        f"{REASON_CORRECTNESS_THRESHOLD}"
    )
