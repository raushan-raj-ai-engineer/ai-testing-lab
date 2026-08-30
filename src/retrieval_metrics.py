def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(document_id in relevant_ids for document_id in top_k)

    return relevant_retrieved / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 1.0

    top_k = retrieved_ids[:k]

    relevant_retrieved = {
        document_id for document_id in top_k if document_id in relevant_ids
    }

    return len(relevant_retrieved) / len(relevant_ids)
