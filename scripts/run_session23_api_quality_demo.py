import json
import sys
from dataclasses import asdict
from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.ai_api_testing.contract import (  # noqa: E402
    validate_contract,
)
from src.ai_api_testing.models import (  # noqa: E402
    APIResponse,
)
from src.ai_api_testing.performance import (  # noqa: E402
    build_api_performance_report,
)

# =========================================================
# REPORT PATH
# =========================================================

REPORT_FILE = ROOT / "reports" / "session23_api_quality_report.json"


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # API CONTRACT
    # -----------------------------------------------------

    schema = {
        "required": [
            "answer",
            "confidence",
        ],
        "additionalProperties": False,
        "properties": {
            "answer": {
                "type": "string",
            },
            "confidence": {
                "type": "number",
            },
        },
    }

    # -----------------------------------------------------
    # SAMPLE AI API RESPONSE
    # -----------------------------------------------------

    payload = {
        "answer": "release passed",
        "confidence": 0.97,
    }

    # -----------------------------------------------------
    # CONTRACT VALIDATION
    # -----------------------------------------------------

    contract_issues = validate_contract(
        payload,
        schema,
    )

    # -----------------------------------------------------
    # SAMPLE PERFORMANCE RESPONSES
    #
    # Simulates 5 successful AI API calls with different
    # latency values.
    # -----------------------------------------------------

    latency_samples = (
        80,
        90,
        100,
        110,
        120,
    )

    responses = [
        APIResponse(
            200,
            payload,
            latency_ms,
        )
        for latency_ms in latency_samples
    ]

    # -----------------------------------------------------
    # PERFORMANCE QUALITY GATE
    # -----------------------------------------------------

    performance_report = build_api_performance_report(
        responses,
        min_success_rate=1.0,
        max_p95_latency_ms=200,
        max_latency_ms=250,
    )

    # -----------------------------------------------------
    # OVERALL RELEASE DECISION
    # -----------------------------------------------------

    release_passed = not contract_issues and performance_report.release_passed

    # -----------------------------------------------------
    # ENSURE REPORT DIRECTORY EXISTS
    # -----------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # WRITE JSON REPORT
    # -----------------------------------------------------

    REPORT_FILE.write_text(
        json.dumps(
            {
                "contract_issues": [asdict(issue) for issue in contract_issues],
                "performance": asdict(
                    performance_report,
                ),
                "release_passed": (release_passed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # PRINT QUALITY REPORT
    # -----------------------------------------------------

    print()
    print("=================================")
    print("SESSION 23 AI API QUALITY")
    print("=================================")

    print(
        "Contract issues:",
        contract_issues,
    )

    print(
        "Success rate:",
        f"{performance_report.success_rate:.2%}",
    )

    print(
        "P95 latency ms:",
        round(
            performance_report.p95_latency_ms,
            2,
        ),
    )

    print(
        "Max latency ms:",
        round(
            performance_report.max_latency_ms,
            2,
        ),
    )

    print(
        "Report:",
        REPORT_FILE,
    )

    print(
        "Release:",
        ("PASS ✅" if release_passed else "FAIL ❌"),
    )

    # -----------------------------------------------------
    # RELEASE GATE
    # -----------------------------------------------------

    raise SystemExit(0 if release_passed else 1)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
