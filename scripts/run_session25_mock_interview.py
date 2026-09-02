import argparse
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

from src.interview_prep.engine import (  # noqa: E402
    load_question_bank,
    select_questions,
)

# =========================================================
# QUESTION BANK
# =========================================================

QUESTION_BANK_FILE = ROOT / "config" / "session25_interview_questions.json"


# =========================================================
# ARGUMENTS
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run Session 25 AI Testing mock interview."),
    )

    parser.add_argument(
        "--topic",
        default=None,
        help=("Optional interview topic filter."),
    )

    parser.add_argument(
        "--difficulty",
        default="senior",
        help=("Question difficulty level. Default: senior"),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help=("Number of interview questions to select."),
    )

    parser.add_argument(
        "--answers",
        action="store_true",
        help=("Show key points and model answers."),
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================


def main():
    args = parse_args()

    # -----------------------------------------------------
    # LOAD QUESTION BANK
    # -----------------------------------------------------

    question_bank = load_question_bank(
        QUESTION_BANK_FILE,
    )

    # -----------------------------------------------------
    # SELECT QUESTIONS
    # -----------------------------------------------------

    questions = select_questions(
        question_bank,
        topic=args.topic,
        difficulty=args.difficulty,
        count=args.count,
    )

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    print()
    print("=================================")
    print("SESSION 25 MOCK INTERVIEW")
    print("=================================")

    print(
        "Difficulty:",
        args.difficulty,
    )

    print(
        "Topic:",
        (args.topic if args.topic else "All"),
    )

    print(
        "Questions:",
        len(questions),
    )

    # -----------------------------------------------------
    # NO QUESTIONS FOUND
    # -----------------------------------------------------

    if not questions:
        print()
        print("No matching interview questions found.")

        raise SystemExit(1)

    # -----------------------------------------------------
    # PRINT QUESTIONS
    # -----------------------------------------------------

    for index, question in enumerate(
        questions,
        start=1,
    ):
        print()
        print(f"{index}. [{question.topic}] {question.question}")

        # -------------------------------------------------
        # OPTIONAL ANSWERS
        # -------------------------------------------------

        if args.answers:
            print()
            print("Key points:")

            for point in question.key_points:
                print(
                    " -",
                    point,
                )

            print()
            print("Model answer:")

            print(question.model_answer)

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    raise SystemExit(0)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
