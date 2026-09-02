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

from src.agentic_rag.engine import (  # noqa: E402
    answer_with_retrieval,
)
from src.agentic_rag.models import (  # noqa: E402
    Document,
)
from src.agentic_rag.quality import (  # noqa: E402
    evaluate_agentic_rag,
)

# =========================================================
# REPORT PATH
# =========================================================

REPORT_FILE = ROOT / "reports" / "session22_agentic_rag_report.json"


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # CREATE SAMPLE KNOWLEDGE BASE
    # -----------------------------------------------------

    documents = [
        Document(
            "policy-1",
            ("The release process requires a deterministic quality gate."),
        ),
        Document(
            "policy-2",
            ("Live LLM evaluation should be separated from fast PR checks."),
        ),
        Document(
            "other",
            ("Coffee is available in the kitchen."),
        ),
    ]

    # -----------------------------------------------------
    # USER QUESTION
    # -----------------------------------------------------

    question = (
        "According to the policy documentation, what does the release process require?"
    )

    # -----------------------------------------------------
    # RUN AGENTIC RAG
    # -----------------------------------------------------

    result = answer_with_retrieval(
        question,
        documents,
        k=2,
    )

    # -----------------------------------------------------
    # EVALUATE RETRIEVAL + GROUNDING
    # -----------------------------------------------------

    quality = evaluate_agentic_rag(
        result,
        expected_retrieval=True,
        relevant_document_ids={
            "policy-1",
        },
    )

    # -----------------------------------------------------
    # ENSURE REPORT DIRECTORY EXISTS
    # -----------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # WRITE QUALITY REPORT
    # -----------------------------------------------------

    REPORT_FILE.write_text(
        json.dumps(
            asdict(
                quality,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # PRINT REPORT
    # -----------------------------------------------------

    print()
    print("=================================")
    print("SESSION 22 AGENTIC RAG")
    print("=================================")

    print(
        "Precision@K:",
        quality.precision_at_k,
    )

    print(
        "Recall@K:",
        quality.recall_at_k,
    )

    print(
        "Grounding:",
        quality.grounding_score,
    )

    print(
        "Report:",
        REPORT_FILE,
    )

    print(
        "Release:",
        ("PASS ✅" if quality.release_passed else "FAIL ❌"),
    )

    # -----------------------------------------------------
    # RELEASE GATE
    # -----------------------------------------------------

    raise SystemExit(0 if quality.release_passed else 1)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
