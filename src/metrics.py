from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

from src.evaluation_model import evaluation_model

reason_correctness = GEval(
    name="Ticket Reason Correctness",
    criteria="""
Evaluate whether the actual output correctly expresses
the CORE support issue described in the input.

Evaluate semantic meaning, NOT exact wording.

Rules:

- Paraphrases must be accepted.
- Shorter explanations are acceptable when they preserve
  the important meaning.
- Do NOT require exact phrases from expected_output.
- Do NOT fail because minor non-essential details are omitted.
- For example, "duplicate charge", "charged twice",
  "billed two times", and "duplicate transaction"
  should be treated as semantically equivalent when the
  user's issue is duplicate billing.
- Do not require phrases such as "same order" or "my card"
  unless omitting them materially changes the meaning.

Fail when:
- The response identifies the wrong issue.
- The response contradicts the input.
- The response invents unsupported information.
- The response is so generic that it does not identify
  the customer's actual problem.

The expected output is a semantic reference,
not an exact answer that must be reproduced.
""",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.8,
    model=evaluation_model,
)


REASON_CORRECTNESS_THRESHOLD = 0.8


def build_reason_correctness_metric() -> GEval:

    return GEval(
        name="Ticket Reason Correctness",
        evaluation_steps=[
            ("Identify the core customer-support issue described in the input."),
            (
                "Compare the actual output with that core issue "
                "and with the expected output. Evaluate semantic "
                "meaning rather than exact wording."
            ),
            (
                "Treat equivalent phrases as correct paraphrases. "
                "For duplicate billing, phrases such as "
                "'duplicate charge', 'charged twice', "
                "'billed twice', 'duplicate transaction', "
                "and 'two charges for one purchase' should "
                "represent the same core issue."
            ),
            (
                "Do not penalize omission of minor details such as "
                "'same order', 'my card', or similar wording unless "
                "the omission changes the core meaning."
            ),
            (
                "Fail the response if it identifies a different "
                "support issue, contradicts the input, or invents "
                "unsupported information."
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=REASON_CORRECTNESS_THRESHOLD,
        model=evaluation_model,
        # Easier debugging while using local Ollama.
        async_mode=False,
    )


def build_calibration_metric() -> GEval:

    return GEval(
        name="Ticket Reason Correctness",
        evaluation_steps=[
            "Identify the core support issue in the input.",
            ("Determine whether the actual output expresses the same core issue."),
            (
                "Accept semantic paraphrases. "
                "'duplicate charge', 'charged twice', "
                "'billed twice', 'duplicate transaction', "
                "and 'two charges for one purchase' "
                "are equivalent for duplicate billing."
            ),
            ("Ignore minor omitted details when they do not change the support issue."),
            (
                "Reject outputs that identify a different issue, "
                "contradict the input, or invent a different problem."
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        # Calibration first — no pass/fail yet.
        threshold=None,
        model=evaluation_model,
        async_mode=False,
    )
