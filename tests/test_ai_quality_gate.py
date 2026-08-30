import json
from pathlib import Path

from src.evaluator import evaluate_dataset
from src.rag.rag_service import answer_question

DATASET_PATH = Path(__file__).parent.parent / "data" / "ticket_golden_dataset.json"


def load_dataset() -> list[dict]:

    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# AI QUALITY GATE
# ============================================================


def test_ai_quality_gate():

    dataset = load_dataset()

    evaluation = evaluate_dataset(dataset)

    assert evaluation.deterministic_pass_rate >= 0.95, (
        "Deterministic quality gate failed. "
        f"Actual: "
        f"{evaluation.deterministic_pass_rate:.2%}"
    )

    assert evaluation.semantic_pass_rate >= 0.80, (
        "Semantic pass-rate quality gate failed. "
        f"Actual: "
        f"{evaluation.semantic_pass_rate:.2%}"
    )

    assert evaluation.average_semantic_score >= 0.80, (
        "Average semantic score quality gate failed. "
        f"Actual: "
        f"{evaluation.average_semantic_score:.2f}"
    )


# ============================================================
# BASIC RAG TESTS
# ============================================================


def test_refund_question():

    result = answer_question("How long do I have to request a refund?")

    assert result.document_ids == ["REFUND_POLICY"]

    assert "30" in result.answer


def test_password_question():

    result = answer_question("I cannot login. How can I reset my password?")

    assert result.document_ids == ["PASSWORD_POLICY"]

    assert "password" in result.answer.lower()


def test_shipping_question():

    result = answer_question("How many days does standard shipping take?")

    assert result.document_ids == ["SHIPPING_POLICY"]

    assert "3" in result.answer

    assert "5" in result.answer


def test_unknown_question():

    result = answer_question("What is the CEO's salary?")

    assert result.document_ids == []

    assert "don't know" in result.answer.lower()


# ============================================================
# RETRIEVAL CONTEXT TESTS
# ============================================================


def test_refund_retrieval_context():

    result = answer_question("How long do I have to request a refund?")

    assert result.retrieval_context

    assert len(result.retrieval_context) == 1

    context = str(result.retrieval_context[0])

    assert "30 days" in context


def test_unknown_question_has_no_context():

    result = answer_question("What is the CEO's salary?")

    assert result.document_ids == []

    assert result.retrieval_context == []
