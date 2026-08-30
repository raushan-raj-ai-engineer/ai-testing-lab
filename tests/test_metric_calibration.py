from deepeval.test_case import LLMTestCase

from src.metrics import build_calibration_metric

CALIBRATION_CASES = [
    # -------------------------
    # Known GOOD
    # -------------------------
    (
        "GOOD-1",
        "My credit card was charged twice for the same order.",
        "The customer was billed twice for one purchase.",
        "The customer reports a duplicate charge.",
        True,
    ),
    (
        "GOOD-2",
        "My credit card was charged twice for the same order.",
        "There is a duplicate transaction for the purchase.",
        "The customer reports a duplicate charge.",
        True,
    ),
    (
        "GOOD-3",
        "My credit card was charged twice for the same order.",
        "The same purchase appears to have been charged two times.",
        "The customer reports a duplicate charge.",
        True,
    ),
    (
        "GOOD-4",
        "I forgot my password and cannot login.",
        "The customer is having a login or password issue.",
        "The customer cannot access their account.",
        True,
    ),
    (
        "GOOD-5",
        "The application crashes whenever I open settings.",
        "The application crashes when the customer opens settings.",
        "The customer reports an application crash.",
        True,
    ),
    # -------------------------
    # Known BAD
    # -------------------------
    (
        "BAD-1",
        "My credit card was charged twice for the same order.",
        "The customer's password expired.",
        "The customer reports a duplicate charge.",
        False,
    ),
    (
        "BAD-2",
        "My credit card was charged twice for the same order.",
        "The package is delayed.",
        "The customer reports a duplicate charge.",
        False,
    ),
    (
        "BAD-3",
        "I forgot my password and cannot login.",
        "The customer was charged twice.",
        "The customer cannot access their account.",
        False,
    ),
    (
        "BAD-4",
        "The application crashes whenever I open settings.",
        "The customer wants a refund.",
        "The customer reports an application crash.",
        False,
    ),
    (
        "BAD-5",
        "My credit card was charged twice for the same order.",
        "There is no problem with the transaction.",
        "The customer reports a duplicate charge.",
        False,
    ),
]


def test_reason_metric_calibration():

    for (
        case_id,
        input_text,
        actual_output,
        expected_output,
        human_expected_pass,
    ) in CALIBRATION_CASES:
        metric = build_calibration_metric()

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
        )

        metric.measure(test_case)

        print(
            f"\n{case_id}"
            f"\nHuman expected pass: {human_expected_pass}"
            f"\nScore: {metric.score}"
            f"\nReason: {metric.reason}"
        )
