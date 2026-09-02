from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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

from src.final_project.models import (  # noqa: E402
    ComponentGate,
)
from src.final_project.release_gate import (  # noqa: E402
    build_final_release_report,
)

# =========================================================
# REPORT PATHS
# =========================================================

REPORTS_DIR = ROOT / "reports"

SAFETY_REPORT = REPORTS_DIR / "session18_guardrail_report.json"

OBSERVABILITY_REPORT = REPORTS_DIR / "session19_observability_report.json"

RELIABILITY_REPORT = REPORTS_DIR / "session20_reliability_report.json"

EVALUATION_REPORT = REPORTS_DIR / "session21_evaluation_report.json"

AGENTIC_RAG_REPORT = REPORTS_DIR / "session22_agentic_rag_report.json"

AI_API_REPORT = REPORTS_DIR / "session23_api_quality_report.json"


# =========================================================
# JSON READER
# =========================================================


def read_json(
    path: Path,
) -> dict[str, Any]:
    """
    Read JSON report from disk.
    """

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


# =========================================================
# NESTED VALUE READER
# =========================================================


def get_nested_value(
    data: Any,
    path: tuple[str, ...],
) -> Any:
    """
    Read a nested JSON value using a tuple path.

    Example:

        ("summary", "release_passed")
    """

    value = data

    for key in path:
        value = value[key]

    return value


# =========================================================
# COMPONENT GATE BUILDER
# =========================================================


def build_component_gate(
    name: str,
    report_path: Path,
    pass_path: tuple[str, ...],
    score_path: tuple[str, ...] | None = None,
    critical: bool = True,
) -> ComponentGate:
    """
    Build one component quality gate from a report.

    Missing or invalid reports fail safely.
    """

    # -----------------------------------------------------
    # REPORT MISSING
    # -----------------------------------------------------

    if not report_path.exists():
        print(f"WARNING: Missing report for {name}: {report_path}")

        return ComponentGate(
            name,
            False,
            0.0,
            critical,
        )

    # -----------------------------------------------------
    # REPORT LOAD / VALIDATION
    # -----------------------------------------------------

    try:
        raw = read_json(
            report_path,
        )

        passed = bool(
            get_nested_value(
                raw,
                pass_path,
            )
        )

        # -------------------------------------------------
        # DEFAULT SCORE
        # -------------------------------------------------

        score = 1.0

        # -------------------------------------------------
        # OPTIONAL SCORE
        # -------------------------------------------------

        if score_path is not None:
            score = float(
                get_nested_value(
                    raw,
                    score_path,
                )
            )

        return ComponentGate(
            name,
            passed,
            score,
            critical,
        )

    # -----------------------------------------------------
    # FAIL CLOSED
    #
    # Corrupt/malformed report must never accidentally
    # produce a PASS.
    # -----------------------------------------------------

    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"WARNING: Invalid report for {name}: {error}")

        return ComponentGate(
            name,
            False,
            0.0,
            critical,
        )


# =========================================================
# MAIN
# =========================================================


def main() -> None:

    # =====================================================
    # BUILD COMPONENT QUALITY GATES
    # =====================================================

    components = [
        # -------------------------------------------------
        # SESSION 18
        # SECURITY / GUARDRAILS
        # -------------------------------------------------
        build_component_gate(
            name="safety",
            report_path=SAFETY_REPORT,
            pass_path=(
                "summary",
                "release_passed",
            ),
            score_path=(
                "summary",
                "detection_rate",
            ),
            critical=True,
        ),
        # -------------------------------------------------
        # SESSION 19
        # OBSERVABILITY
        # -------------------------------------------------
        build_component_gate(
            name="observability",
            report_path=OBSERVABILITY_REPORT,
            pass_path=(
                "report",
                "release_passed",
            ),
            score_path=None,
            critical=True,
        ),
        # -------------------------------------------------
        # SESSION 20
        # RELIABILITY
        # -------------------------------------------------
        build_component_gate(
            name="reliability",
            report_path=RELIABILITY_REPORT,
            pass_path=(
                "report",
                "release_passed",
            ),
            score_path=(
                "report",
                "pass_rate",
            ),
            critical=True,
        ),
        # -------------------------------------------------
        # SESSION 21
        # EVALUATION PIPELINE
        # -------------------------------------------------
        build_component_gate(
            name="evaluation_pipeline",
            report_path=EVALUATION_REPORT,
            pass_path=(
                "summary",
                "release_passed",
            ),
            score_path=(
                "summary",
                "average_score",
            ),
            critical=True,
        ),
        # -------------------------------------------------
        # SESSION 22
        # AGENTIC RAG
        # -------------------------------------------------
        build_component_gate(
            name="agentic_rag",
            report_path=AGENTIC_RAG_REPORT,
            pass_path=("release_passed",),
            score_path=("grounding_score",),
            critical=True,
        ),
        # -------------------------------------------------
        # SESSION 23
        # AI API QUALITY
        # -------------------------------------------------
        build_component_gate(
            name="ai_api",
            report_path=AI_API_REPORT,
            pass_path=("release_passed",),
            score_path=(
                "performance",
                "success_rate",
            ),
            critical=True,
        ),
    ]

    # =====================================================
    # BUILD FINAL RELEASE REPORT
    # =====================================================

    report = build_final_release_report(
        components,
    )

    # =====================================================
    # PRINT FINAL QUALITY REPORT
    # =====================================================

    print()
    print("======================================")
    print("SESSION 24 FINAL AI SDET RELEASE GATE")
    print("======================================")

    # =====================================================
    # COMPONENT RESULTS
    # =====================================================

    for component in components:
        status = "PASS ✅" if component.passed else "FAIL ❌"

        print(f"{component.name:<22} {status:<8} score={component.score:.2f}")

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print(
        "Critical failures:",
        report.critical_failures,
    )

    print(
        "Average score:",
        round(
            report.average_score,
            3,
        ),
    )

    # =====================================================
    # FINAL RELEASE DECISION
    # =====================================================

    if report.release_passed:
        print()
        print("FINAL RELEASE: PASS ✅")

        raise SystemExit(0)

    print()
    print("FINAL RELEASE: FAIL ❌")

    print()
    print("Tip:")

    print("Run Sessions 18-23 report-producing scripts before Session 24.")

    raise SystemExit(1)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
