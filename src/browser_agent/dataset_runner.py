from __future__ import annotations

from uuid import uuid4

from mcp import Client

from src.browser_agent.advanced_quality import (
    AdvancedAgentQualityReport,
    build_advanced_quality_report,
)
from src.browser_agent.browser_agent import (
    BrowserAgentRunResult,
    run_browser_agent,
)
from src.browser_agent.dataset_quality import (
    AgentQualityDataset,
    ScenarioEvaluation,
    evaluate_scenario_report,
)

TODO_URL = "https://demo.playwright.dev/todomvc"


def build_failed_report() -> AdvancedAgentQualityReport:
    """
    Used when a scenario crashes before a normal
    BrowserAgentRunResult can be produced.
    """

    return AdvancedAgentQualityReport(
        completed=False,
        progress_status="error",
        progress_score=0.0,
        executed_steps=0,
        decision_attempts=0,
        rejected_attempts=0,
        decision_efficiency=0.0,
        execution_attempts=0,
        execution_failures=1,
        duplicate_decisions=0,
        stuck_loop=False,
    )


async def execute_live_quality_dataset(
    dataset: AgentQualityDataset,
    mcp_url: str = ("http://localhost:8931/mcp"),
) -> list[ScenarioEvaluation]:
    """
    Run every scenario independently.

    IMPORTANT:

    One scenario failure must NOT abort
    the complete dataset.
    """

    evaluations: list[ScenarioEvaluation] = []

    async with Client(mcp_url) as client:
        for scenario in dataset.scenarios:
            print("\n==================================")

            print(
                "DATASET SCENARIO:",
                scenario.id,
            )

            print("==================================")

            unique_suffix = uuid4().hex[:6]

            todo_name = f"{scenario.todo_text}-{unique_suffix}"

            goal = f"Add {todo_name} and mark it complete"

            print(
                "\nGoal:",
                goal,
            )

            try:
                # =========================================
                # 1. Fresh browser state
                # =========================================

                navigate_result = await client.call_tool(
                    "browser_navigate",
                    {"url": (TODO_URL)},
                )

                if navigate_result.is_error:
                    raise RuntimeError("TodoMVC navigation failed")

                # =========================================
                # 2. Real autonomous execution
                # =========================================

                result: BrowserAgentRunResult = await run_browser_agent(
                    client=client,
                    goal=goal,
                    max_steps=(scenario.max_steps),
                    max_execution_retries=(scenario.max_execution_retries),
                )

                # =========================================
                # 3. Deterministic final-state validation
                # =========================================

                if todo_name not in result.final_snapshot:
                    raise AssertionError("Todo missing from final browser state")

                if "[checked]" not in result.final_snapshot:
                    raise AssertionError("Todo exists but is not completed")

                # =========================================
                # 4. Advanced quality report
                # =========================================

                report = build_advanced_quality_report(
                    goal=goal,
                    result=result,
                )

                # =========================================
                # 5. Scenario-level gate
                # =========================================

                evaluation = evaluate_scenario_report(
                    scenario=scenario,
                    goal=goal,
                    report=report,
                )

            except Exception as exc:
                # =========================================
                # Scenario fails, dataset continues
                # =========================================

                error_message = f"{type(exc).__name__}: {exc}"

                print(
                    "\nSCENARIO ERROR:",
                    error_message,
                )

                report = build_failed_report()

                evaluation = ScenarioEvaluation(
                    scenario_id=(scenario.id),
                    goal=goal,
                    passed=False,
                    failures=[(f"Scenario execution failed: {error_message}")],
                    report=report,
                )

            # =============================================
            # Save result whether PASS or FAIL
            # =============================================

            evaluations.append(evaluation)

            print(
                "\nScenario passed:",
                evaluation.passed,
            )

            print(
                "Decision efficiency:",
                round(
                    evaluation.report.decision_efficiency,
                    3,
                ),
            )

            print(
                "Execution failures:",
                evaluation.report.execution_failures,
            )

            print(
                "Duplicate decisions:",
                evaluation.report.duplicate_decisions,
            )

            print(
                "Stuck loop:",
                evaluation.report.stuck_loop,
            )

            if evaluation.failures:
                print("Failures:")

                for failure in evaluation.failures:
                    print(
                        " -",
                        failure,
                    )

    return evaluations
