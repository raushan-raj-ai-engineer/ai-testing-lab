from deepeval.test_case import (
    LLMTestCase,
    RetrievedContextData,
)

from src.rag_metrics import (
    build_contextual_precision_metric,
    build_contextual_recall_metric,
)


def make_retrieval_context(
    *texts: str,
) -> list[str | RetrievedContextData]:

    context: list[str | RetrievedContextData] = list(texts)

    return context


# ============================================================
# CONTEXTUAL PRECISION
#
# Same relevant document exists in both cases.
#
# GOOD:
# Relevant document ranked first.
#
# BAD:
# Irrelevant document ranked first.
# ============================================================


def test_contextual_precision_ranking():

    question = "How long do I have to request a refund?"

    expected_output = "Customers can request a refund within 30 days of purchase."

    actual_output = "Customers can request a refund within 30 days of purchase."

    refund_context = "Customers can request a refund within 30 days of purchase."

    shipping_context = "Standard shipping normally takes 3 to 5 business days."

    # --------------------------------------------------------
    # GOOD ranking
    # --------------------------------------------------------

    good_context = make_retrieval_context(
        refund_context,
        shipping_context,
    )

    good_test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=good_context,
    )

    good_metric = build_contextual_precision_metric()

    good_metric.measure(good_test_case)

    good_score = good_metric.score

    assert good_score is not None

    # --------------------------------------------------------
    # BAD ranking
    # --------------------------------------------------------

    bad_context = make_retrieval_context(
        shipping_context,
        refund_context,
    )

    bad_test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=bad_context,
    )

    bad_metric = build_contextual_precision_metric()

    bad_metric.measure(bad_test_case)

    bad_score = bad_metric.score

    assert bad_score is not None

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== CONTEXTUAL PRECISION TEST =====")

    print("\nGOOD ranking:")

    print(good_context)

    print(
        "Score:",
        good_score,
    )

    print(
        "Reason:",
        good_metric.reason,
    )

    print("\nBAD ranking:")

    print(bad_context)

    print(
        "Score:",
        bad_score,
    )

    print(
        "Reason:",
        bad_metric.reason,
    )

    # --------------------------------------------------------
    # Expected behaviour
    # --------------------------------------------------------

    assert good_score > bad_score, (
        "Contextual Precision evaluator failed "
        "to distinguish good ranking from bad ranking.\n"
        f"Good score: {good_score}\n"
        f"Bad score: {bad_score}"
    )


# ============================================================
# CONTEXTUAL RECALL
#
# Ideal answer requires TWO facts:
#
# 1. Refund = 30 days
# 2. Shipping = 3 to 5 business days
#
# FULL retrieval contains both.
# INCOMPLETE retrieval contains only refund.
# ============================================================


def test_contextual_recall_detects_missing_information():

    question = "What is the refund window and how long does standard shipping take?"

    expected_output = (
        "Customers can request a refund "
        "within 30 days of purchase, and "
        "standard shipping normally takes "
        "3 to 5 business days."
    )

    actual_output = expected_output

    refund_context = "Customers can request a refund within 30 days of purchase."

    shipping_context = "Standard shipping normally takes 3 to 5 business days."

    # --------------------------------------------------------
    # FULL retrieval
    # --------------------------------------------------------

    complete_context = make_retrieval_context(
        refund_context,
        shipping_context,
    )

    complete_test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=complete_context,
    )

    complete_metric = build_contextual_recall_metric()

    complete_metric.measure(complete_test_case)

    complete_score = complete_metric.score

    assert complete_score is not None

    # --------------------------------------------------------
    # INCOMPLETE retrieval
    # --------------------------------------------------------

    incomplete_context = make_retrieval_context(
        refund_context,
    )

    incomplete_test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=incomplete_context,
    )

    incomplete_metric = build_contextual_recall_metric()

    incomplete_metric.measure(incomplete_test_case)

    incomplete_score = incomplete_metric.score

    assert incomplete_score is not None

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== CONTEXTUAL RECALL TEST =====")

    print("\nCOMPLETE retrieval:")

    print(complete_context)

    print(
        "Score:",
        complete_score,
    )

    print(
        "Reason:",
        complete_metric.reason,
    )

    print("\nINCOMPLETE retrieval:")

    print(incomplete_context)

    print(
        "Score:",
        incomplete_score,
    )

    print(
        "Reason:",
        incomplete_metric.reason,
    )

    # --------------------------------------------------------
    # Expected behaviour
    # --------------------------------------------------------

    assert complete_score > incomplete_score, (
        "Contextual Recall evaluator failed "
        "to detect missing retrieval evidence.\n"
        f"Complete score: {complete_score}\n"
        f"Incomplete score: {incomplete_score}"
    )
