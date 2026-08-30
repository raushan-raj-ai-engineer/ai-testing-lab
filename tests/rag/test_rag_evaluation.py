from deepeval.test_case import (
    LLMTestCase,
    RetrievedContextData,
)

from src.rag.rag_service import answer_question
from src.rag_metrics import (
    ANSWER_RELEVANCY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    GROUNDING_THRESHOLD,
    build_answer_relevancy_metric,
    build_faithfulness_metric,
    build_strict_grounding_metric,
)


def make_retrieval_context(
    *texts: str,
) -> list[str | RetrievedContextData]:

    context: list[str | RetrievedContextData] = list(texts)

    return context


# ============================================================
# 1. REAL RAG SYSTEM
# Correct retrieval + correct answer
# ============================================================


def test_refund_rag_quality():

    # --------------------------------------------------------
    # Arrange
    # --------------------------------------------------------

    question = "How long do I have to request a refund?"

    # --------------------------------------------------------
    # Act
    # --------------------------------------------------------

    result = answer_question(question)

    # --------------------------------------------------------
    # Deterministic retrieval validation
    # --------------------------------------------------------

    assert result.document_ids == ["REFUND_POLICY"], (
        f"Wrong document retrieved. Actual documents: {result.document_ids}"
    )

    assert result.retrieval_context, (
        "Expected retrieval context, but no context was returned."
    )

    assert result.answer.strip(), "Expected a non-empty RAG answer."

    # --------------------------------------------------------
    # Build DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(
        input=question,
        actual_output=result.answer,
        retrieval_context=result.retrieval_context,
    )

    # --------------------------------------------------------
    # Create fresh metrics
    # --------------------------------------------------------

    faithfulness = build_faithfulness_metric()

    answer_relevancy = build_answer_relevancy_metric()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    faithfulness.measure(test_case)

    answer_relevancy.measure(test_case)

    # --------------------------------------------------------
    # Safely read scores
    # --------------------------------------------------------

    faithfulness_score = faithfulness.score

    answer_relevancy_score = answer_relevancy.score

    assert faithfulness_score is not None, "Faithfulness metric did not return a score."

    assert answer_relevancy_score is not None, (
        "Answer relevancy metric did not return a score."
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== REAL RAG TEST =====")

    print(
        "Question:",
        question,
    )

    print(
        "Retrieved documents:",
        result.document_ids,
    )

    print(
        "Retrieved context:",
        result.retrieval_context,
    )

    print(
        "Actual answer:",
        result.answer,
    )

    print(
        "Faithfulness score:",
        faithfulness_score,
    )

    print(
        "Faithfulness reason:",
        faithfulness.reason,
    )

    print(
        "Answer relevancy score:",
        answer_relevancy_score,
    )

    print(
        "Answer relevancy reason:",
        answer_relevancy.reason,
    )

    # --------------------------------------------------------
    # Quality gates
    # --------------------------------------------------------

    assert faithfulness_score >= FAITHFULNESS_THRESHOLD, (
        "Faithfulness quality gate failed.\n"
        f"Score: {faithfulness_score}\n"
        f"Threshold: {FAITHFULNESS_THRESHOLD}\n"
        f"Reason: {faithfulness.reason}"
    )

    assert answer_relevancy_score >= ANSWER_RELEVANCY_THRESHOLD, (
        "Answer relevancy quality gate failed.\n"
        f"Score: {answer_relevancy_score}\n"
        f"Threshold: {ANSWER_RELEVANCY_THRESHOLD}\n"
        f"Reason: {answer_relevancy.reason}"
    )


# ============================================================
# 2. UNFAITHFUL ANSWER
#
# Context says 30 days
# Answer says 90 days
#
# Expected:
# Faithfulness score should be BELOW threshold
# ============================================================


def test_unfaithful_answer_is_detected():

    # --------------------------------------------------------
    # Arrange
    # --------------------------------------------------------

    question = "How long do I have to request a refund?"

    retrieval_context = make_retrieval_context(
        ("Customers can request a refund within 30 days of purchase.")
    )

    wrong_answer = "Customers can request a refund within 90 days of purchase."

    # --------------------------------------------------------
    # Build DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(
        input=question,
        actual_output=wrong_answer,
        retrieval_context=retrieval_context,
    )

    faithfulness = build_faithfulness_metric()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    faithfulness.measure(test_case)

    score = faithfulness.score

    assert score is not None, "Faithfulness metric did not return a score."

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== UNFAITHFUL ANSWER TEST =====")

    print(
        "Question:",
        question,
    )

    print(
        "Context:",
        retrieval_context,
    )

    print(
        "Wrong answer:",
        wrong_answer,
    )

    print(
        "Faithfulness score:",
        score,
    )

    print(
        "Faithfulness reason:",
        faithfulness.reason,
    )

    # --------------------------------------------------------
    # Expected rejection
    # --------------------------------------------------------

    assert score < FAITHFULNESS_THRESHOLD, (
        "Evaluator incorrectly accepted "
        "an unsupported answer.\n"
        f"Score: {score}\n"
        f"Threshold: {FAITHFULNESS_THRESHOLD}\n"
        f"Reason: {faithfulness.reason}"
    )


# ============================================================
# 3. IRRELEVANT ANSWER
#
# Question asks refund
# Answer talks about shipping
#
# Expected:
# Answer relevancy score should be BELOW threshold
# ============================================================


def test_irrelevant_answer_is_detected():

    # --------------------------------------------------------
    # Arrange
    # --------------------------------------------------------

    question = "How long do I have to request a refund?"

    retrieval_context = make_retrieval_context(
        ("Customers can request a refund within 30 days of purchase.")
    )

    irrelevant_answer = "Standard shipping normally takes 3 to 5 business days."

    # --------------------------------------------------------
    # Build DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(
        input=question,
        actual_output=irrelevant_answer,
        retrieval_context=retrieval_context,
    )

    answer_relevancy = build_answer_relevancy_metric()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    answer_relevancy.measure(test_case)

    score = answer_relevancy.score

    assert score is not None, "Answer relevancy metric did not return a score."

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n===== IRRELEVANT ANSWER TEST =====")

    print(
        "Question:",
        question,
    )

    print(
        "Context:",
        retrieval_context,
    )

    print(
        "Irrelevant answer:",
        irrelevant_answer,
    )

    print(
        "Answer relevancy score:",
        score,
    )

    print(
        "Answer relevancy reason:",
        answer_relevancy.reason,
    )

    # --------------------------------------------------------
    # Expected rejection
    # --------------------------------------------------------

    assert score < ANSWER_RELEVANCY_THRESHOLD, (
        "Evaluator incorrectly accepted "
        "an irrelevant answer.\n"
        f"Score: {score}\n"
        f"Threshold: "
        f"{ANSWER_RELEVANCY_THRESHOLD}\n"
        f"Reason: {answer_relevancy.reason}"
    )


# ============================================================
# 4. PARTIAL HALLUCINATION
#
# First sentence is supported by context
# Second sentence is invented
#
# Expected:
# Faithfulness score should be BELOW threshold
# ============================================================


def test_partially_hallucinated_answer_is_detected():

    question = "How long do I have to request a refund?"

    retrieval_context = make_retrieval_context(
        ("Customers can request a refund within 30 days of purchase.")
    )

    answer_with_extra_claim = (
        "Customers can request a refund "
        "within 30 days of purchase. "
        "Refunds are always processed "
        "within two hours."
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer_with_extra_claim,
        retrieval_context=retrieval_context,
    )

    grounding = build_strict_grounding_metric()

    grounding.measure(test_case)

    score = grounding.score

    assert score is not None

    print("\n===== PARTIAL HALLUCINATION TEST =====")

    print(
        "Score:",
        score,
    )

    print(
        "Reason:",
        grounding.reason,
    )

    assert score < GROUNDING_THRESHOLD, (
        "Evaluator failed to detect "
        "the unsupported additional claim.\n"
        f"Score: {score}\n"
        f"Threshold: {GROUNDING_THRESHOLD}\n"
        f"Reason: {grounding.reason}"
    )
