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
    Find the checkbox that belongs to one specific todo.

    Supports Playwright refs such as:

        e21
        f1e21
        f2e105
        other future non-space ref formats

    We first try to stay inside the matching listitem block.
    If that structure is unavailable, we use a small
    proximity fallback.
    """

    lines = snapshot.splitlines()

    todo_name_lower = todo_name.casefold()

    # =====================================================
    # Find lines containing the requested todo text
    # =====================================================

    todo_indexes: list[int] = []

    for index, line in enumerate(lines):
        if todo_name_lower in line.casefold():
            todo_indexes.append(index)

    if not todo_indexes:
        return None

    # =====================================================
    # Try each matching occurrence
    # =====================================================

    for todo_index in todo_indexes:
        # -------------------------------------------------
        # Find nearest parent listitem
        # -------------------------------------------------

        listitem_index: int | None = None

        for index in range(
            todo_index,
            -1,
            -1,
        ):
            line = lines[index]

            if "listitem" in line.lower():
                listitem_index = index
                break

        # -------------------------------------------------
        # If listitem exists, search only inside that block
        # -------------------------------------------------

        if listitem_index is not None:
            listitem_line = lines[listitem_index]

            base_indent = len(listitem_line) - len(listitem_line.lstrip())

            block_end = len(lines)

            for index in range(
                listitem_index + 1,
                len(lines),
            ):
                candidate = lines[index]

                if not candidate.strip():
                    continue

                candidate_indent = len(candidate) - len(candidate.lstrip())

                # Another sibling/top-level node means
                # this listitem block has finished.
                if candidate_indent <= base_indent:
                    block_end = index
                    break

            block = lines[listitem_index:block_end]

            for line in block:
                lower_line = line.casefold()

                if (
                    "checkbox" in lower_line
                    and "mark all as complete" not in lower_line
                ):
                    return line

        # =================================================
        # Fallback
        #
        # Some MCP snapshot versions may not preserve the
        # exact listitem hierarchy we expect.
        #
        # Search close to the todo text.
        # =================================================

        search_start = max(
            0,
            todo_index - 12,
        )

        search_end = min(
            len(lines),
            todo_index + 6,
        )

        candidates: list[tuple[int, str]] = []

        for index in range(
            search_start,
            search_end,
        ):
            line = lines[index]

            lower_line = line.casefold()

            if "checkbox" not in lower_line:
                continue

            if "mark all as complete" in lower_line:
                continue

            distance = abs(index - todo_index)

            candidates.append((
                distance,
                line,
            ))

        if candidates:
            candidates.sort(key=lambda item: item[0])

            return candidates[0][1]

    return None


def find_todo_checkbox_ref(
    todo_name: str,
    snapshot: str,
) -> str | None:
    """
    Return the dynamic Playwright accessibility ref
    for the requested todo checkbox.

    DO NOT assume refs always look like:

        e21

    Playwright MCP may return:

        e21
        f1e21
        f2e104

    Therefore capture any valid non-space value inside
    [ref=...].
    """

    checkbox_line = find_todo_checkbox_line(
        todo_name=todo_name,
        snapshot=snapshot,
    )

    if checkbox_line is None:
        return None

    match = re.search(
        r"\[ref=([^\]\s]+)\]",
        checkbox_line,
    )

    if match is None:
        return None

    return match.group(1)


def find_todo_input_line(
    snapshot: str,
) -> str | None:
    """
    Find TodoMVC's add-new-todo textbox line
    from the Playwright accessibility snapshot.

    Supports refs such as:
        e8
        f1e8
        f2e8
    """

    lines = snapshot.splitlines()

    # Prefer the known TodoMVC textbox.
    preferred_keywords = (
        "what needs to be done",
        "new todo",
    )

    for line in lines:
        lower_line = line.casefold()

        if "textbox" not in lower_line:
            continue

        if "[ref=" not in lower_line:
            continue

        if any(keyword in lower_line for keyword in preferred_keywords):
            return line

    # Fallback:
    # TodoMVC normally has one editable textbox.
    for line in lines:
        lower_line = line.casefold()

        if "textbox" in lower_line and "[ref=" in lower_line:
            return line

    return None


def find_todo_input_ref(
    snapshot: str,
) -> str | None:
    """
    Return exact dynamic Playwright ref for the
    TodoMVC new-todo textbox.

    Examples:
        e8
        f1e8
        f2e8
    """

    textbox_line = find_todo_input_line(
        snapshot=snapshot,
    )

    if textbox_line is None:
        return None

    match = re.search(
        r"\[ref=([^\]\s]+)\]",
        textbox_line,
    )

    if match is None:
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
    Validate an AI-selected browser action before
    allowing it to reach Playwright MCP.

    This validator separates:

    1. Tool selection
    2. Argument correctness
    3. Element grounding
    4. Business/goal correctness

    Supported subgoals:

        Add <todo> to the todo list

        Mark <todo> as complete

    Important:
    We do NOT silently correct the AI action here.

    If the AI chooses the wrong target/ref/text/tool,
    this function raises ValueError.

    choose_valid_browser_action() can then give the
    error back to the LLM and allow it to re-plan.
    """

    # =====================================================
    # BASIC VALIDATION
    # =====================================================

    if not goal.strip():
        raise ValueError("Browser action goal cannot be empty")

    if not tool_name.strip():
        raise ValueError("Browser tool name cannot be empty")

    if not isinstance(
        arguments,
        dict,
    ):
        raise ValueError("Browser tool arguments must be a dictionary")

    if not snapshot.strip():
        raise ValueError("Browser snapshot cannot be empty")

    # =====================================================
    # ADD TODO GOAL
    #
    # Example:
    #
    # Add Buy milk to the todo list
    # =====================================================

    add_todo = extract_todo_from_add_goal(goal)

    if add_todo is not None:
        # -------------------------------------------------
        # 1. Correct tool
        # -------------------------------------------------

        if tool_name != "browser_type":
            raise ValueError("Adding a todo requires browser_type")

        # -------------------------------------------------
        # 2. Exact todo text
        #
        # Agent must not:
        #
        # Buy milk-123  -> Buy milk
        #
        # or rewrite/summarize the task.
        # -------------------------------------------------

        actual_text = arguments.get("text")

        if actual_text != add_todo:
            raise ValueError(f"Type only '{add_todo}' into the todo textbox")

        # -------------------------------------------------
        # 3. Must submit
        #
        # browser_type(..., submit=True)
        #
        # Otherwise todo may only be typed but not added.
        # -------------------------------------------------

        if arguments.get("submit") is not True:
            raise ValueError("browser_type must use submit=True when adding a todo")

        # -------------------------------------------------
        # 4. Determine exact textbox ref from CURRENT
        #    accessibility snapshot.
        #
        # Never hardcode:
        #
        # e8
        # f1e8
        # f2e8
        # -------------------------------------------------

        expected_target = find_todo_input_ref(
            snapshot=snapshot,
        )

        if expected_target is None:
            raise ValueError(
                "Could not find the new-todo textbox in the current browser snapshot"
            )

        actual_target = arguments.get("target")

        print("\nTodo textbox grounding check:")

        print(
            "Todo:",
            add_todo,
        )

        print(
            "Expected textbox target:",
            expected_target,
        )

        print(
            "AI selected target:",
            actual_target,
        )

        # -------------------------------------------------
        # 5. Exact element grounding
        #
        # Examples we must reject:
        #
        # expected f1e8
        # AI       f1e11
        #
        # expected f2e8
        # AI       generic[f2e7]
        # -------------------------------------------------

        if actual_target != expected_target:
            raise ValueError(
                "For adding todo "
                f"'{add_todo}', the correct "
                "textbox target is "
                f"{expected_target}, "
                "but you selected "
                f"{actual_target}. "
                "Use browser_type with target "
                f"{expected_target}."
            )

        # -------------------------------------------------
        # ADD action is valid
        # -------------------------------------------------

        return

    # =====================================================
    # MARK TODO COMPLETE GOAL
    #
    # Example:
    #
    # Mark Buy milk as complete
    # =====================================================

    completion_todo = extract_todo_from_completion_goal(goal)

    if completion_todo is not None:
        # -------------------------------------------------
        # 1. Correct tool
        # -------------------------------------------------

        if tool_name != "browser_click":
            raise ValueError("Marking a todo complete requires browser_click")

        # -------------------------------------------------
        # 2. Left click only
        #
        # button may be omitted by MCP/LLM because left
        # click is normally the default.
        # -------------------------------------------------

        button = arguments.get(
            "button",
            "left",
        )

        if button != "left":
            raise ValueError("Todo checkbox must use button='left'")
        # -------------------------------------------------
        # 3. Reject double click
        # -------------------------------------------------

        double_click = arguments.get(
            "doubleClick",
            False,
        )

        if double_click is True:
            raise ValueError("Todo checkbox should use a single click, not doubleClick")

        # -------------------------------------------------
        # 4. Find checkbox belonging specifically
        #    to requested todo.
        #
        # This deliberately avoids:
        #
        # Mark all as complete
        # All filter
        # Active filter
        # Completed filter
        # todo text/generic node
        # -------------------------------------------------

        expected_target = find_todo_checkbox_ref(
            todo_name=(completion_todo),
            snapshot=snapshot,
        )

        if expected_target is None:
            raise ValueError(f"Could not find checkbox for todo '{completion_todo}'")

        actual_target = arguments.get("target")

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
            actual_target,
        )

        # -------------------------------------------------
        # 5. Exact checkbox grounding
        # -------------------------------------------------

        if actual_target != expected_target:
            raise ValueError(
                f"For todo '{completion_todo}', "
                "the correct checkbox target is "
                f"{expected_target}, "
                "but you selected "
                f"{actual_target}. "
                "Use browser_click with target "
                f"{expected_target}."
            )

        # -------------------------------------------------
        # COMPLETE action is valid
        # -------------------------------------------------

        return

    # =====================================================
    # UNSUPPORTED GOAL
    #
    # Important:
    # Don't guess what the user/agent meant.
    # =====================================================

    raise ValueError(f"Unsupported browser subgoal: {goal}")


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
