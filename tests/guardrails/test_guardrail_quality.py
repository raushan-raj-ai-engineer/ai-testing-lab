from pathlib import Path

import pytest

from src.guardrails.models import (
    AdversarialCase,
)
from src.guardrails.quality import (
    evaluate_adversarial_dataset,
    load_adversarial_dataset,
)

pytestmark = pytest.mark.guardrail


ROOT = Path(__file__).resolve().parents[2]


DATASET_PATH = ROOT / "config" / "session18_adversarial_cases.json"


def test_adversarial_dataset_loads():

    cases = load_adversarial_dataset(DATASET_PATH)

    assert len(cases) >= 10


def test_session18_adversarial_gate_passes():

    cases = load_adversarial_dataset(DATASET_PATH)

    evaluations, report = evaluate_adversarial_dataset(cases)

    assert len(evaluations) == len(cases)

    assert report.missed_attacks == 0

    assert report.false_positives == 0

    assert report.detection_rate == 1.0

    assert report.release_passed is True


def test_quality_gate_detects_missed_attack():

    cases = [
        AdversarialCase(
            id="bad-ground-truth",
            prompt=("Add Buy milk and mark it complete"),
            # Deliberately claim benign input
            # should have been blocked.
            expected_allowed=False,
            expected_categories=(),
        )
    ]

    _, report = evaluate_adversarial_dataset(cases)

    assert report.missed_attacks == 1

    assert report.release_passed is False
