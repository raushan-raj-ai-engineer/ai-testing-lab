from src.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
)


def test_precision_at_1_good_ranking():

    retrieved = [
        "REFUND_POLICY",
        "SHIPPING_POLICY",
    ]

    relevant = {"REFUND_POLICY"}

    score = precision_at_k(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=1,
    )

    assert score == 1.0


def test_precision_at_1_bad_ranking():

    retrieved = [
        "SHIPPING_POLICY",
        "REFUND_POLICY",
    ]

    relevant = {"REFUND_POLICY"}

    score = precision_at_k(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=1,
    )

    assert score == 0.0


def test_complete_retrieval_recall():

    retrieved = [
        "REFUND_POLICY",
        "SHIPPING_POLICY",
    ]

    relevant = {
        "REFUND_POLICY",
        "SHIPPING_POLICY",
    }

    score = recall_at_k(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=2,
    )

    assert score == 1.0


def test_incomplete_retrieval_recall():

    retrieved = [
        "REFUND_POLICY",
    ]

    relevant = {
        "REFUND_POLICY",
        "SHIPPING_POLICY",
    }

    score = recall_at_k(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=2,
    )

    assert score == 0.5
