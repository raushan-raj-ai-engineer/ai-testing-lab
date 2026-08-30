import json
from pathlib import Path

from src.evaluator import evaluate_dataset

DATASET_PATH = Path(__file__).parent.parent / "data" / "ticket_golden_dataset.json"

DETERMINISTIC_THRESHOLD = 0.95


def load_dataset() -> list[dict]:

    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_classification_quality_gate():

    dataset = load_dataset()

    evaluation = evaluate_dataset(dataset)

    print("\nDETERMINISTIC QUALITY REPORT\n-----------------------------")

    print(f"Total Cases: {evaluation.total_cases}")

    print(f"Deterministic Pass Rate: {evaluation.deterministic_pass_rate:.2%}")

    failed_cases = [case for case in evaluation.cases if not case.deterministic_passed]

    if failed_cases:
        print("\nFAILED DETERMINISTIC CASES")

        print("-----------------------------")

        for case in failed_cases:
            print(f"\nCase ID: {case.case_id}")

            print(
                "Failed fields:",
                case.deterministic_failures,
            )

            print(
                "Actual category:",
                case.actual_category,
            )

            print(
                "Actual priority:",
                case.actual_priority,
            )

            print(
                "Actual needs_human:",
                case.actual_needs_human,
            )

    assert evaluation.deterministic_pass_rate >= DETERMINISTIC_THRESHOLD, (
        "Deterministic quality gate failed. "
        f"Actual: "
        f"{evaluation.deterministic_pass_rate:.2%}, "
        f"Expected: >= "
        f"{DETERMINISTIC_THRESHOLD:.0%}"
    )
