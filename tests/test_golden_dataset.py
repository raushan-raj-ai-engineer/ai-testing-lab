import json
from pathlib import Path

import pytest

from src.llm_client import classify_ticket

DATASET_PATH = Path(__file__).parent.parent / "data" / "ticket_golden_dataset.json"


def load_dataset() -> list[dict]:

    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


DATASET = load_dataset()


@pytest.mark.parametrize(
    "case",
    DATASET,
    ids=lambda case: case["id"],
)
def test_ticket_golden_dataset(case):

    result = classify_ticket(case["ticket"])

    assert result.category == case["expected_category"]

    if "expected_priority" in case:
        assert result.priority == case["expected_priority"]

    if "expected_needs_human" in case:
        assert result.needs_human == case["expected_needs_human"]
