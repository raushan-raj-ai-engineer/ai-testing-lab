import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from mcp.types import TextContent
from ollama import chat

MODEL = "llama3.2"


# =========================================================
# RESULT MODELS
# =========================================================


MODEL = "llama3.2"


@dataclass
class BrowserAgentStep:
    step_number: int
    goal: str
    tool_name: str
    arguments: dict[str, Any]

    execution_result: Any | None = None


@dataclass
class BrowserAgentAttempt:
    """
    One AI decision attempt.

    Example:

    Attempt 1:
        wrong tool -> rejected

    Attempt 2:
        correct tool -> accepted
    """

    step_number: int
    attempt_number: int
    goal: str

    tool_name: str | None
    arguments: dict[str, Any]

    accepted: bool

    rejection_reason: str | None = None


@dataclass
class BrowserAgentExecutionAttempt:
    """
    One real MCP execution attempt.

    AI decision can be valid,
    but MCP execution can still fail.
    """

    step_number: int

    execution_attempt_number: int

    goal: str

    tool_name: str

    arguments: dict[str, Any]

    succeeded: bool

    error: str | None = None

    execution_result: Any | None = None


@dataclass
class BrowserAgentRunResult:
    completed: bool

    # Successful browser actions
    steps: list[BrowserAgentStep]

    final_snapshot: str

    # Accepted + rejected AI decisions
    attempts: list[BrowserAgentAttempt] = field(default_factory=list)

    # Successful + failed MCP executions
    execution_attempts: list[BrowserAgentExecutionAttempt] = field(default_factory=list)


# =========================================================
# MCP RESULT HELPER
# =========================================================


def get_text_content(result) -> str:
    """
    Extract text from MCP CallToolResult.
    """

    parts: list[str] = []

    for block in result.content:
        if isinstance(
            block,
            TextContent,
        ):
            parts.append(block.text)

    return "\n".join(parts)


# =========================================================
# LLM BROWSER DECISION
# =========================================================


def choose_browser_action(
    goal: str,
    snapshot: str,
    tools: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """
    Ask Ollama to choose one browser action.

    Always returns:

        (
            tool_name,
            arguments,
        )

    tool_name:
        str | None

    arguments:
        dict[str, Any]
    """

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a browser automation agent.\n"
                    "\n"
                    "Rules:\n"
                    "1. Use only the browser tools provided.\n"
                    "2. Use element refs from the accessibility snapshot.\n"
                    "3. Never invent refs.\n"
                    "4. When adding a todo, type only the literal "
                    "todo text.\n"
                    "5. When adding or creating a todo, use "
                    "browser_type with submit=true.\n"
                    "6. When marking a specific todo complete, "
                    "click that todo's checkbox.\n"
                    "7. Do not click All, Active, or Completed "
                    "filters to complete a todo.\n"
                    "8. Do not click the global Mark all as "
                    "complete checkbox for a specific todo.\n"
                    "9. Normal checkbox/button clicks should use "
                    "the left mouse button.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal:\n{goal}\n\nCurrent accessibility snapshot:\n{snapshot}"
                ),
            },
        ],
        tools=tools,
        options={
            "temperature": 0,
        },
    )

    tool_calls = response.message.tool_calls

    # No tool selected
    if not tool_calls:
        return None, {}

    tool_call = tool_calls[0]

    name_value = tool_call.function.name

    # Type narrowing for Pylance
    if not isinstance(
        name_value,
        str,
    ):
        raise ValueError("LLM returned an invalid tool name")

    tool_name: str = name_value

    raw_arguments = tool_call.function.arguments

    arguments: dict[str, Any] = dict(raw_arguments or {})

    return (
        tool_name,
        arguments,
    )


# =========================================================
# ARGUMENT NORMALIZATION
# =========================================================


def normalize_browser_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:

    selected_tool = None

    for tool in tools:
        function = tool.get("function", {})

        if function.get("name") == tool_name:
            selected_tool = tool
            break

    if selected_tool is None:
        raise ValueError(f"Unknown browser tool: {tool_name}")

    function = selected_tool["function"]

    parameters = function.get(
        "parameters",
        {},
    )

    properties = parameters.get(
        "properties",
        {},
    )

    required = set(
        parameters.get(
            "required",
            [],
        )
    )

    normalized: dict[str, Any] = {}

    for arg_name, value in arguments.items():
        # -----------------------------------------
        # 1. Unknown argument
        # -----------------------------------------

        if arg_name not in properties:
            raise ValueError(f"Unknown argument '{arg_name}' for tool '{tool_name}'")

        property_schema = properties[arg_name]

        expected_type = property_schema.get("type")

        # -----------------------------------------
        # 2. None handling
        # -----------------------------------------

        if value is None:
            if arg_name in required:
                raise ValueError(f"{arg_name} is required")

            # Optional None should not be sent
            # to MCP.
            continue

        # -----------------------------------------
        # 3. Boolean normalization
        #
        # "true"  -> True
        # "false" -> False
        # -----------------------------------------

        if expected_type == "boolean" and isinstance(value, str):
            lowered = value.strip().lower()

            if lowered == "true":
                value = True

            elif lowered == "false":
                value = False

            else:
                raise ValueError(f"{arg_name} must be boolean, got {value!r}")

        # -----------------------------------------
        # 4. Array normalization
        #
        # "[]" -> []
        #
        # '["Alt"]' -> ["Alt"]
        # -----------------------------------------

        elif expected_type == "array" and isinstance(value, str):
            try:
                parsed_value = json.loads(value)

            except json.JSONDecodeError as exc:
                raise ValueError(f"{arg_name} must be an array, got {value!r}") from exc

            if not isinstance(
                parsed_value,
                list,
            ):
                raise ValueError(f"{arg_name} must be an array, got {value!r}")

            value = parsed_value

        # -----------------------------------------
        # 5. Basic type validation
        # -----------------------------------------

        if expected_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{arg_name} must be boolean, got {type(value).__name__}")

        if expected_type == "array" and not isinstance(value, list):
            raise ValueError(f"{arg_name} must be an array, got {type(value).__name__}")

        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"{arg_name} must be string, got {type(value).__name__}")

        if expected_type == "integer" and not isinstance(value, int):
            raise ValueError(f"{arg_name} must be integer, got {type(value).__name__}")

        if expected_type == "number" and not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(f"{arg_name} must be number, got {type(value).__name__}")

        normalized[arg_name] = value

    # ---------------------------------------------
    # 6. Required arguments must exist
    # ---------------------------------------------

    for required_arg in required:
        if required_arg not in normalized:
            raise ValueError(f"{required_arg} is required")

    return normalized


# =========================================================
# GOAL PARSING
# =========================================================


def extract_todo_from_add_goal(
    goal: str,
) -> str | None:
    """
    Example:

        Add Buy milk to the todo list

    Returns:

        Buy milk
    """

    patterns = [
        (
            r"(?:add|create)\s+(.+?)"
            r"\s+to\s+the\s+todo\s+list$"
        ),
        (
            r"(?:add|create)\s+(.+?)"
            r"\s+todo(?:\s+item)?$"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            goal.strip(),
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


def extract_todo_from_completion_goal(
    goal: str,
) -> str | None:
    """
    Supports:

        Mark Buy milk as complete
        Mark Buy milk complete

    Returns:

        Buy milk
    """

    match = re.search(
        r"mark\s+(.+?)\s+(?:as\s+)?complete$",
        goal.strip(),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).strip()


def extract_todo_from_full_goal(
    goal: str,
) -> str | None:
    """
    Supported examples:

        Add Buy milk and mark it complete

        Add Buy milk and mark it as complete

        Add Buy milk to the todo list
        and mark it complete

        Add Buy milk to the todo list
        and mark it as complete
    """

    patterns = [
        (
            r"add\s+(.+?)"
            r"\s+to\s+the\s+todo\s+list"
            r"\s+and\s+mark\s+it\s+"
            r"(?:as\s+)?complete$"
        ),
        (
            r"add\s+(.+?)"
            r"\s+and\s+mark\s+it\s+"
            r"(?:as\s+)?complete$"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            goal.strip(),
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


# =========================================================
# SNAPSHOT / ELEMENT GROUNDING
# =========================================================


def find_todo_checkbox_line(
    todo_name: str,
    snapshot: str,
) -> str | None:
    """
    Snapshot example:

        checkbox "Toggle Todo" [ref=e21]
        generic [ref=e22]: Buy milk

    Returns checkbox line belonging to Buy milk.
    """

    lines = snapshot.splitlines()

    for index, line in enumerate(lines):
        if todo_name.lower() not in line.lower():
            continue

        start_index = max(
            0,
            index - 5,
        )

        previous_lines = lines[start_index:index]

        for previous_line in reversed(previous_lines):
            previous_lower = previous_line.lower()

            if "checkbox" not in previous_lower:
                continue

            if "mark all as complete" in previous_lower:
                continue

            return previous_line

    return None


def find_todo_checkbox_ref(
    todo_name: str,
    snapshot: str,
) -> str | None:
    """
    Return the dynamic ref of a todo checkbox.
    """

    checkbox_line = find_todo_checkbox_line(
        todo_name=todo_name,
        snapshot=snapshot,
    )

    if checkbox_line is None:
        return None

    match = re.search(
        r"\[ref=(e\d+)\]",
        checkbox_line,
    )

    if not match:
        return None

    return match.group(1)


def get_todo_state(
    todo_name: str,
    snapshot: str,
) -> str:
    """
    Returns one of:

        missing
        active
        completed
    """

    checkbox_line = find_todo_checkbox_line(
        todo_name=todo_name,
        snapshot=snapshot,
    )

    if checkbox_line is None:
        return "missing"

    if "[checked]" in checkbox_line.lower():
        return "completed"

    return "active"


# =========================================================
# TASK COMPLETION / SUBGOAL PLANNING
# =========================================================


def determine_next_subgoal(
    goal: str,
    snapshot: str,
) -> str | None:
    """
    High-level goal:

        Add Buy milk and mark it complete

    Agent converts it into:

        Add Buy milk to the todo list

    then:

        Mark Buy milk as complete

    then:

        None

    None means task is finished.
    """

    todo_name = extract_todo_from_full_goal(goal)

    if todo_name is None:
        raise ValueError(
            "Unsupported autonomous goal. "
            "Expected something like: "
            "'Add Buy milk and mark it complete'"
        )

    state = get_todo_state(
        todo_name=todo_name,
        snapshot=snapshot,
    )

    print(
        "\nTask state:",
        state,
    )

    if state == "missing":
        return f"Add {todo_name} to the todo list"

    if state == "active":
        return f"Mark {todo_name} as complete"

    if state == "completed":
        return None

    raise ValueError(f"Unknown todo state: {state}")


def is_goal_complete(
    goal: str,
    snapshot: str,
) -> bool:

    next_goal = determine_next_subgoal(
        goal=goal,
        snapshot=snapshot,
    )

    return next_goal is None


# =========================================================
# SEMANTIC VALIDATION
# =========================================================


def validate_browser_action(
    goal: str,
    tool_name: str,
    arguments: dict[str, Any],
    snapshot: str,
) -> None:
    """
    Deterministic semantic validation before
    browser execution.
    """

    # =====================================================
    # ADD TODO VALIDATION
    # =====================================================

    add_todo = extract_todo_from_add_goal(goal)

    if add_todo is not None:
        if tool_name != "browser_type":
            raise ValueError("Adding a todo requires browser_type")

        if arguments.get("text") != add_todo:
            raise ValueError(f"Type only '{add_todo}' into the todo textbox")

        if arguments.get("submit") is not True:
            raise ValueError("Adding a todo requires submit=True")

    # =====================================================
    # COMPLETE TODO VALIDATION
    # =====================================================

    completion_todo = extract_todo_from_completion_goal(goal)

    if completion_todo is not None:
        if tool_name != "browser_click":
            raise ValueError("Completing a todo requires browser_click")

        button = arguments.get(
            "button",
            "left",
        )

        if button != "left":
            raise ValueError("Normal checkbox interaction requires button='left'")

        selected_target = arguments.get("target")

        if not selected_target:
            raise ValueError("Completing a todo requires a target")

        expected_target = find_todo_checkbox_ref(
            todo_name=completion_todo,
            snapshot=snapshot,
        )

        if expected_target is None:
            raise ValueError(f"Could not find checkbox for todo '{completion_todo}'")

        print("\nTodo grounding check:")

        print(
            "Todo:",
            completion_todo,
        )

        print(
            "Expected checkbox target:",
            expected_target,
        )

        print(
            "AI selected target:",
            selected_target,
        )

        if selected_target != expected_target:
            raise ValueError(
                f"For todo '{completion_todo}', "
                f"the correct checkbox target "
                f"is {expected_target}, "
                f"but you selected "
                f"{selected_target}. "
                f"Use browser_click with "
                f"target {expected_target}."
            )


# =========================================================
# SELF-CORRECTING ACTION SELECTION
# =========================================================


def choose_valid_browser_action(
    goal: str,
    snapshot: str,
    tools: list[dict[str, Any]],
    max_attempts: int = 3,
    step_number: int = 0,
    attempt_trace: list[BrowserAgentAttempt] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a VALID browser action.

    Handles:

    - no tool
    - wrong tool
    - wrong arguments
    - schema problems
    - wrong semantic values
    - wrong browser refs

    Rejected decisions are traced.
    """

    feedback: str | None = None

    # =====================================================
    # AI DECISION RETRY LOOP
    # =====================================================

    for attempt_number in range(
        1,
        max_attempts + 1,
    ):
        current_goal = goal

        # -------------------------------------------------
        # Previous action was rejected
        # -------------------------------------------------

        if feedback is not None:
            current_goal = (
                f"{goal}\n\n"
                "Your previous browser action was rejected.\n"
                f"Reason: {feedback}\n\n"
                "Generate a corrected browser action.\n"
                "Follow the correction exactly.\n"
                "Use the current accessibility "
                "snapshot carefully.\n"
                "Do not repeat the invalid tool, "
                "target, or argument."
            )

        # -------------------------------------------------
        # Ask LLM
        # -------------------------------------------------

        tool_name, raw_arguments = choose_browser_action(
            goal=current_goal,
            snapshot=snapshot,
            tools=tools,
        )

        raw_arguments = dict(raw_arguments or {})

        print(
            f"\nAttempt {attempt_number} tool:",
            tool_name,
        )

        print(
            f"Attempt {attempt_number} raw arguments:",
            raw_arguments,
        )

        # =================================================
        # NO TOOL SELECTED
        # =================================================

        if tool_name is None:
            feedback = "A browser tool must be selected for this subgoal."

            if attempt_trace is not None:
                attempt_trace.append(
                    BrowserAgentAttempt(
                        step_number=step_number,
                        attempt_number=(attempt_number),
                        goal=goal,
                        tool_name=None,
                        arguments={},
                        accepted=False,
                        rejection_reason=feedback,
                    )
                )

            print(
                f"Attempt {attempt_number} rejected:",
                feedback,
            )

            continue

        # From here Pylance knows:
        #
        # tool_name = str

        normalized_arguments: dict[str, Any] | None = None

        try:
            # =================================================
            # NORMALIZATION
            # =================================================

            normalized_arguments = normalize_browser_arguments(
                tool_name=tool_name,
                arguments=raw_arguments,
                tools=tools,
            )

            print(
                f"Attempt {attempt_number} normalized:",
                normalized_arguments,
            )

            # =================================================
            # SEMANTIC + ELEMENT VALIDATION
            # =================================================

            validate_browser_action(
                goal=goal,
                tool_name=tool_name,
                arguments=(normalized_arguments),
                snapshot=snapshot,
            )

            # =================================================
            # ACCEPTED DECISION
            # =================================================

            if attempt_trace is not None:
                attempt_trace.append(
                    BrowserAgentAttempt(
                        step_number=step_number,
                        attempt_number=(attempt_number),
                        goal=goal,
                        tool_name=tool_name,
                        arguments=dict(normalized_arguments),
                        accepted=True,
                        rejection_reason=None,
                    )
                )

            print(f"Attempt {attempt_number} accepted")

            return (
                tool_name,
                normalized_arguments,
            )

        except ValueError as exc:
            feedback = str(exc)

            # -------------------------------------------------
            # Preserve best available arguments
            # -------------------------------------------------

            if normalized_arguments is None:
                traced_arguments = dict(raw_arguments)

            else:
                traced_arguments = dict(normalized_arguments)

            # -------------------------------------------------
            # Rejected attempt trace
            # -------------------------------------------------

            if attempt_trace is not None:
                attempt_trace.append(
                    BrowserAgentAttempt(
                        step_number=step_number,
                        attempt_number=(attempt_number),
                        goal=goal,
                        tool_name=tool_name,
                        arguments=(traced_arguments),
                        accepted=False,
                        rejection_reason=feedback,
                    )
                )

            print(
                f"\nAttempt {attempt_number} rejected:",
                feedback,
            )

    # =====================================================
    # ALL AI DECISIONS FAILED
    # =====================================================

    raise ValueError(
        f"Unable to generate a valid browser action after {max_attempts} attempts"
    )


# =========================================================
# TOOL DISCOVERY
# =========================================================


async def discover_browser_agent_tools(
    client,
) -> list[dict[str, Any]]:
    """
    Discover actual Playwright MCP schemas.

    We do not hardcode runtime schema.
    """

    tools_result = await client.list_tools()

    allowed_tools = {
        "browser_type",
        "browser_click",
    }

    tools: list[dict[str, Any]] = []

    for tool in tools_result.tools:
        if tool.name not in allowed_tools:
            continue

        tools.append({
            "type": "function",
            "function": {
                "name": (tool.name),
                "description": (tool.description or ""),
                "parameters": (tool.input_schema),
            },
        })

    if not tools:
        raise RuntimeError("Required Playwright MCP browser tools were not found")

    return tools


# =========================================================
# AUTONOMOUS BROWSER AGENT LOOP
# =========================================================


async def run_browser_agent(
    client,
    goal: str,
    max_steps: int = 5,
    max_execution_retries: int = 1,
) -> BrowserAgentRunResult:
    """
    Autonomous browser-agent loop.

    Session 15 adds:

    1. rejected AI decision tracing
    2. accepted AI decision tracing
    3. MCP execution tracing
    4. explicit MCP failure retry
    5. successful-step tracking
    6. final task-completion verification
    """

    # =====================================================
    # DISCOVER TOOLS
    # =====================================================

    tools = await discover_browser_agent_tools(client)

    # =====================================================
    # TRACE STORAGE
    # =====================================================

    executed_steps: list[BrowserAgentStep] = []

    all_attempts: list[BrowserAgentAttempt] = []

    all_execution_attempts: list[BrowserAgentExecutionAttempt] = []

    # =====================================================
    # AUTONOMOUS LOOP
    # =====================================================

    for step_number in range(
        1,
        max_steps + 1,
    ):
        print("\n==============================")

        print(f"AUTONOMOUS STEP {step_number}")

        print("==============================")

        # =================================================
        # FRESH BROWSER SNAPSHOT
        # =================================================

        snapshot_result = await client.call_tool(
            "browser_snapshot",
            {},
        )

        if snapshot_result.is_error:
            error_text = get_text_content(snapshot_result)

            raise RuntimeError(f"Browser snapshot failed: {error_text}")

        snapshot = get_text_content(snapshot_result)

        # =================================================
        # NEXT SUBGOAL
        # =================================================

        next_goal = determine_next_subgoal(
            goal,
            snapshot,
        )

        # =================================================
        # STOP CONDITION
        # =================================================

        if next_goal is None:
            print("\nGoal completed.")

            return BrowserAgentRunResult(
                completed=True,
                steps=executed_steps,
                final_snapshot=snapshot,
                attempts=all_attempts,
                execution_attempts=(all_execution_attempts),
            )

        print(
            "\nNext subgoal:",
            next_goal,
        )

        # =================================================
        # AI DECISION
        #
        # Ollama call is synchronous.
        #
        # Keep it outside MCP async event loop.
        # =================================================

        tool_name, arguments = await asyncio.to_thread(
            choose_valid_browser_action,
            goal=next_goal,
            snapshot=snapshot,
            tools=tools,
            max_attempts=3,
            step_number=step_number,
            # Directly trace all decisions
            attempt_trace=all_attempts,
        )

        print(
            "\nExecuting:",
            tool_name,
            arguments,
        )

        # =================================================
        # MCP EXECUTION RETRY
        # =================================================

        successful_result = None

        total_execution_attempts = max_execution_retries + 1

        for execution_attempt_number in range(
            1,
            total_execution_attempts + 1,
        ):
            execution_result = await client.call_tool(
                tool_name,
                arguments,
            )

            # =============================================
            # MCP SUCCESS
            # =============================================

            if not execution_result.is_error:
                all_execution_attempts.append(
                    BrowserAgentExecutionAttempt(
                        step_number=step_number,
                        execution_attempt_number=(execution_attempt_number),
                        goal=next_goal,
                        tool_name=tool_name,
                        arguments=dict(arguments),
                        succeeded=True,
                        error=None,
                        execution_result=(execution_result),
                    )
                )

                successful_result = execution_result

                break

            # =============================================
            # MCP EXPLICIT FAILURE
            # =============================================

            error_text = get_text_content(execution_result)

            all_execution_attempts.append(
                BrowserAgentExecutionAttempt(
                    step_number=step_number,
                    execution_attempt_number=(execution_attempt_number),
                    goal=next_goal,
                    tool_name=tool_name,
                    arguments=dict(arguments),
                    succeeded=False,
                    error=error_text,
                    execution_result=(execution_result),
                )
            )

            print(
                "\nMCP execution attempt",
                execution_attempt_number,
                "failed:",
                error_text,
            )

            # =============================================
            # NO RETRY LEFT
            # =============================================

            if execution_attempt_number >= total_execution_attempts:
                raise RuntimeError(
                    "Browser execution failed "
                    f"after "
                    f"{total_execution_attempts} "
                    "attempt(s): "
                    f"{error_text}"
                )

            print("Retrying explicit MCP execution failure...")

        # =================================================
        # DEFENSIVE CHECK
        # =================================================

        if successful_result is None:
            raise RuntimeError("Browser execution did not produce a successful result")

        # =================================================
        # STORE SUCCESSFUL BROWSER STEP
        # =================================================

        executed_steps.append(
            BrowserAgentStep(
                step_number=step_number,
                goal=next_goal,
                tool_name=tool_name,
                arguments=dict(arguments),
                execution_result=(successful_result),
            )
        )

    # =====================================================
    # MAX STEPS USED
    #
    # Re-read browser state because task might have
    # completed exactly on the last allowed action.
    # =====================================================

    final_snapshot_result = await client.call_tool(
        "browser_snapshot",
        {},
    )

    if final_snapshot_result.is_error:
        error_text = get_text_content(final_snapshot_result)

        raise RuntimeError(f"Final browser snapshot failed: {error_text}")

    final_snapshot = get_text_content(final_snapshot_result)

    # =====================================================
    # COMPLETED EXACTLY ON LAST STEP
    # =====================================================

    if is_goal_complete(
        goal,
        final_snapshot,
    ):
        return BrowserAgentRunResult(
            completed=True,
            steps=executed_steps,
            final_snapshot=(final_snapshot),
            attempts=all_attempts,
            execution_attempts=(all_execution_attempts),
        )

    # =====================================================
    # LOOP SAFETY
    # =====================================================

    raise RuntimeError(f"Browser agent exceeded maximum step count: {max_steps}")
