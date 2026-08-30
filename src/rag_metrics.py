from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import SingleTurnParams

from src.evaluation_model import evaluation_model

FAITHFULNESS_THRESHOLD = 0.8
ANSWER_RELEVANCY_THRESHOLD = 0.8
GROUNDING_THRESHOLD = 0.8


def build_faithfulness_metric() -> FaithfulnessMetric:

    return FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=evaluation_model,
        include_reason=True,
        async_mode=False,
    )


def build_answer_relevancy_metric() -> AnswerRelevancyMetric:

    return AnswerRelevancyMetric(
        threshold=ANSWER_RELEVANCY_THRESHOLD,
        model=evaluation_model,
        include_reason=True,
        async_mode=False,
    )


def build_strict_grounding_metric() -> GEval:

    return GEval(
        name="Strict RAG Grounding",
        evaluation_steps=[
            ("Identify every factual claim made in the actual output."),
            ("Check every factual claim against the retrieval context."),
            ("A factual claim is valid only when the retrieval context supports it."),
            (
                "Do not treat a claim as valid merely "
                "because it does not contradict the context."
            ),
            (
                "If the answer introduces factual information "
                "that is absent from the retrieval context, "
                "treat that information as unsupported."
            ),
            (
                "For example, if the retrieval context says "
                "refunds can be requested within 30 days but "
                "contains no information about processing time, "
                "then a claim that refunds are processed within "
                "two hours is unsupported."
            ),
            (
                "Give a high score only when all important "
                "factual claims in the actual output are "
                "grounded in the retrieval context."
            ),
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        threshold=GROUNDING_THRESHOLD,
        model=evaluation_model,
        async_mode=False,
    )


CONTEXTUAL_PRECISION_THRESHOLD = 0.8
CONTEXTUAL_RECALL_THRESHOLD = 0.8


def build_contextual_precision_metric(
    threshold: float | None = None,
) -> ContextualPrecisionMetric:

    return ContextualPrecisionMetric(
        threshold=threshold,
        model=evaluation_model,
        include_reason=True,
        async_mode=False,
    )


def build_contextual_recall_metric(
    threshold: float | None = None,
) -> ContextualRecallMetric:

    return ContextualRecallMetric(
        threshold=threshold,
        model=evaluation_model,
        include_reason=True,
        async_mode=False,
    )
