from dataclasses import dataclass

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import RetrievedContextData, SingleTurnParams
from ollama import chat

from src.evaluation_model import evaluation_model
from src.rag.knowledge_base import (
    DOCUMENTS,
    KnowledgeDocument,
)


@dataclass
class RAGResponse:
    answer: str
    document_ids: list[str]
    retrieval_context: list[str | RetrievedContextData]


def get_document_by_id(
    document_id: str,
) -> KnowledgeDocument:

    for document in DOCUMENTS:
        if document["id"] == document_id:
            return document

    raise ValueError(f"Knowledge document not found: {document_id}")


def retrieve(
    question: str,
) -> list[KnowledgeDocument]:

    question_lower = question.lower()

    if "refund" in question_lower:
        return [get_document_by_id("REFUND_POLICY")]

    if "password" in question_lower or "login" in question_lower:
        return [get_document_by_id("PASSWORD_POLICY")]

    if "shipping" in question_lower:
        return [get_document_by_id("SHIPPING_POLICY")]

    return []


def answer_question(
    question: str,
) -> RAGResponse:

    documents = retrieve(question)

    context = "\n".join(document["text"] for document in documents)

    prompt = f"""
Answer the question using ONLY the supplied context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the context does not contain the answer, reply exactly:
  I don't know based on the provided context.

Context:
{context}

Question:
{question}
"""

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
        },
    )

    content = response.message.content

    if content is None:
        raise ValueError("RAG model returned empty content.")

    answer = content.strip()

    if not answer:
        raise ValueError("RAG model returned blank content.")

    retrieval_context: list[str | RetrievedContextData] = [
        document["text"] for document in documents
    ]

    return RAGResponse(
        answer=answer,
        document_ids=[document["id"] for document in documents],
        retrieval_context=retrieval_context,
    )


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
            ("Read every factual claim made in the actual output."),
            (
                "For each factual claim, determine whether "
                "the retrieval context explicitly supports "
                "that claim."
            ),
            (
                "Do not treat a claim as correct merely "
                "because it does not contradict the context."
            ),
            ("A claim must have supporting evidence in the retrieval context."),
            (
                "If the actual output introduces any factual "
                "detail that is absent from the retrieval "
                "context, penalize the response."
            ),
            (
                "For example, if the context says refunds "
                "are available within 30 days but says "
                "nothing about processing time, a statement "
                "such as 'refunds are processed within two "
                "hours' is unsupported and must reduce "
                "the score."
            ),
            (
                "Give a high score only when all factual "
                "claims are grounded in the supplied "
                "retrieval context."
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
