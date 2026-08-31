from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

mcp = MCPServer("AI Testing QA Server")


@mcp.tool()
def calculate_pass_rate(
    passed: Annotated[
        int,
        Field(
            ge=0,
            description="Number of passed test cases",
        ),
    ],
    total: Annotated[
        int,
        Field(
            gt=0,
            description="Total number of test cases",
        ),
    ],
) -> float:
    """
    Calculate the test pass rate as a percentage.
    """

    if passed > total:
        raise ValueError("Passed test count cannot be greater than total test count.")

    return round(
        (passed / total) * 100,
        2,
    )


@mcp.tool()
def release_decision(
    pass_rate: Annotated[
        float,
        Field(
            ge=0,
            le=100,
            description="Overall test pass rate percentage",
        ),
    ],
    critical_failures: Annotated[
        int,
        Field(
            ge=0,
            description="Number of critical test failures",
        ),
    ],
) -> str:
    """
    Decide whether a release should be blocked.

    Release is allowed only when:
    - pass rate is at least 95 percent
    - there are zero critical failures
    """

    if critical_failures > 0:
        return "BLOCK"

    if pass_rate < 95:
        return "BLOCK"

    return "RELEASE"
