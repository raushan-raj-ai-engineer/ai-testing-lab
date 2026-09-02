import sys
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

from src.eval_pipeline.dashboard import (  # noqa: E402
    write_html_dashboard,
)
from src.eval_pipeline.pipeline import (  # noqa: E402
    build_release_summary,
    load_eval_thresholds,
    load_records,
    write_release_report,
)

# =========================================================
# FILE PATHS
# =========================================================

RECORDS_FILE = ROOT / "config" / "session21_sample_records.json"

THRESHOLDS_FILE = ROOT / "config" / "session21_eval_thresholds.json"

REPORT_FILE = ROOT / "reports" / "session21_evaluation_report.json"

DASHBOARD_FILE = ROOT / "reports" / "session21_dashboard.html"


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # ENSURE REPORT DIRECTORY EXISTS
    # -----------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # LOAD SAMPLE EVALUATION RECORDS
    # -----------------------------------------------------

    records = load_records(
        RECORDS_FILE,
    )

    # -----------------------------------------------------
    # LOAD QUALITY THRESHOLDS
    # -----------------------------------------------------

    thresholds = load_eval_thresholds(
        THRESHOLDS_FILE,
    )

    # -----------------------------------------------------
    # BUILD RELEASE SUMMARY
    # -----------------------------------------------------

    summary = build_release_summary(
        records,
        thresholds,
    )

    # -----------------------------------------------------
    # WRITE JSON REPORT
    # -----------------------------------------------------

    write_release_report(
        summary=summary,
        records=records,
        path=REPORT_FILE,
    )

    # -----------------------------------------------------
    # WRITE HTML DASHBOARD
    # -----------------------------------------------------

    write_html_dashboard(
        summary=summary,
        records=records,
        path=DASHBOARD_FILE,
    )

    # -----------------------------------------------------
    # PRINT QUALITY RESULT
    # -----------------------------------------------------

    print()
    print("=================================")
    print("SESSION 21 EVAL PIPELINE")
    print("=================================")

    print(
        "Records:",
        len(records),
    )

    print(
        "JSON report:",
        REPORT_FILE,
    )

    print(
        "HTML dashboard:",
        DASHBOARD_FILE,
    )

    print(
        "Release:",
        ("PASS ✅" if summary.release_passed else "FAIL ❌"),
    )

    # -----------------------------------------------------
    # RELEASE GATE
    # -----------------------------------------------------

    raise SystemExit(0 if summary.release_passed else 1)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
