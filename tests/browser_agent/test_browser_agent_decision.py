import pytest
from mcp import Client

from src.browser_agent.browser_agent import (
    choose_browser_action,
    extract_todo_from_full_goal,
    normalize_browser_arguments,
    run_browser_agent,
)


def test_agent_selects_browser_type():

    snapshot = """
    - heading "todos" [ref=e7]
    - textbox "What needs to be done?" [ref=e8]
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_type",
                "description": "Type text into an element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "text": {
                            "type": "string",
                        },
                        "submit": {
                            "type": "boolean",
                        },
                    },
                    "required": [
                        "target",
                        "text",
                    ],
                },
            },
        }
    ]

    # -----------------------------------------
    # RAW LLM decision
    # -----------------------------------------

    tool_name, raw_arguments = choose_browser_action(
        goal=("Add Buy milk to the todo list"),
        snapshot=snapshot,
        tools=tools,
    )

    assert tool_name is not None
    assert tool_name == "browser_type"

    # -----------------------------------------
    # Normalize raw LLM arguments
    #
    # "true" -> True
    # -----------------------------------------

    arguments = normalize_browser_arguments(
        tool_name=tool_name,
        arguments=raw_arguments,
        tools=tools,
    )

    print(
        "\nSelected tool:",
        tool_name,
    )

    print(
        "Raw arguments:",
        raw_arguments,
    )

    print(
        "Normalized arguments:",
        arguments,
    )

    assert arguments["target"] == "e8"

    assert arguments["text"] == "Buy milk"

    assert arguments["submit"] is True


# =========================================================
# PARSER TESTS
# =========================================================


def test_full_goal_parser():

    assert (
        extract_todo_from_full_goal("Add Buy milk and mark it complete") == "Buy milk"
    )

    assert (
        extract_todo_from_full_goal("Add Buy milk and mark it as complete")
        == "Buy milk"
    )

    assert (
        extract_todo_from_full_goal(
            "Add Buy milk to the todo list and mark it complete"
        )
        == "Buy milk"
    )

    assert (
        extract_todo_from_full_goal(
            "Add Buy milk to the todo list and mark it as complete"
        )
        == "Buy milk"
    )


# =========================================================
# AUTONOMOUS PLAYWRIGHT MCP TEST
# =========================================================


@pytest.mark.anyio
async def test_agent_uses_real_playwright_snapshot():

    async with Client("http://localhost:8931/mcp") as client:
        # =================================
        # STEP 1
        # Open real application
        # =================================

        navigate_result = await client.call_tool(
            "browser_navigate",
            {
                "url": ("https://demo.playwright.dev/todomvc"),
            },
        )

        assert navigate_result.is_error is False

        # =================================
        # ONE HIGH-LEVEL GOAL
        # =================================

        goal = "Add Buy milk and mark it complete"

        # Sanity check:
        # goal parser must work
        # before autonomous execution.

        assert extract_todo_from_full_goal(goal) == "Buy milk"

        # =================================
        # AUTONOMOUS AGENT LOOP
        # =================================

        result = await run_browser_agent(
            client=client,
            goal=goal,
            max_steps=5,
        )

        print(
            "\nAgent completed:",
            result.completed,
        )

        print("\nExecuted steps:")

        for step in result.steps:
            print(
                step.step_number,
                step.goal,
                step.tool_name,
                step.arguments,
            )

        print(
            "\nFinal snapshot:\n",
            result.final_snapshot,
        )

        # =================================
        # FINAL TASK COMPLETION
        # =================================

        assert result.completed is True

        assert "Buy milk" in result.final_snapshot

        assert "[checked]" in result.final_snapshot

        # =================================
        # TRAJECTORY VALIDATION
        # =================================

        assert len(result.steps) == 2

        assert result.steps[0].tool_name == "browser_type"

        assert result.steps[1].tool_name == "browser_click"

        # =================================
        # STEP ORDER
        # =================================

        assert result.steps[0].goal == ("Add Buy milk to the todo list")

        assert result.steps[1].goal == ("Mark Buy milk as complete")


def test_agent_stops_when_goal_already_complete():

    snapshot = """
    - list:
        - listitem:
            - checkbox "Toggle Todo" [checked] [ref=e21]
            - generic [ref=e22]: Buy milk
    """

    from src.browser_agent.browser_agent import (
        determine_next_subgoal,
        is_goal_complete,
    )

    goal = "Add Buy milk and mark it complete"

    next_goal = determine_next_subgoal(
        goal=goal,
        snapshot=snapshot,
    )

    assert next_goal is None

    assert (
        is_goal_complete(
            goal=goal,
            snapshot=snapshot,
        )
        is True
    )


@pytest.mark.anyio
async def test_agent_stops_at_max_steps():

    async with Client("http://localhost:8931/mcp") as client:
        await client.call_tool(
            "browser_navigate",
            {
                "url": ("https://demo.playwright.dev/todomvc"),
            },
        )

        goal = "Add Buy milk and mark it complete"

        with pytest.raises(
            RuntimeError,
            match="maximum step count",
        ):
            await run_browser_agent(
                client=client,
                goal=goal,
                # Intentionally too small
                max_steps=1,
            )


def test_agent_rejects_unsupported_goal():

    snapshot = """
    - heading "todos" [ref=e7]
    - textbox "What needs to be done?" [ref=e8]
    """

    from src.browser_agent.browser_agent import (
        determine_next_subgoal,
    )

    goal = "Do something with Buy milk"

    with pytest.raises(
        ValueError,
        match="Unsupported autonomous goal",
    ):
        determine_next_subgoal(
            goal=goal,
            snapshot=snapshot,
        )


def test_validator_rejects_wrong_todo_target():

    from src.browser_agent.browser_agent import (
        validate_browser_action,
    )

    snapshot = """
    - checkbox "Toggle Todo" [ref=e21]
    - generic [ref=e22]: Buy milk
    - link "Completed" [ref=e32]
    """

    goal = "Mark Buy milk as complete"

    wrong_arguments = {
        "target": "e32",
        "button": "left",
    }

    with pytest.raises(
        ValueError,
        match="correct checkbox target",
    ):
        validate_browser_action(
            goal=goal,
            tool_name="browser_click",
            arguments=wrong_arguments,
            snapshot=snapshot,
        )


def test_validator_rejects_wrong_tool_for_completion():

    from src.browser_agent.browser_agent import (
        validate_browser_action,
    )

    snapshot = """
    - checkbox "Toggle Todo" [ref=e21]
    - generic [ref=e22]: Buy milk
    """

    goal = "Mark Buy milk as complete"

    wrong_arguments = {
        "target": "e21",
        "text": "Buy milk",
    }

    with pytest.raises(
        ValueError,
        match="browser_click",
    ):
        validate_browser_action(
            goal=goal,
            tool_name="browser_type",
            arguments=wrong_arguments,
            snapshot=snapshot,
        )


def test_validator_rejects_submit_false_for_add_todo():

    from src.browser_agent.browser_agent import (
        validate_browser_action,
    )

    snapshot = """
    - textbox "What needs to be done?" [ref=e8]
    """

    goal = "Add Buy milk to the todo list"

    wrong_arguments = {
        "target": "e8",
        "text": "Buy milk",
        "submit": False,
    }

    with pytest.raises(
        ValueError,
        match="submit=True",
    ):
        validate_browser_action(
            goal=goal,
            tool_name="browser_type",
            arguments=wrong_arguments,
            snapshot=snapshot,
        )


def test_normalizer_converts_boolean_string():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_type",
                "description": "Type text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "text": {
                            "type": "string",
                        },
                        "submit": {
                            "type": "boolean",
                        },
                    },
                    "required": [
                        "target",
                        "text",
                    ],
                },
            },
        }
    ]

    raw_arguments = {
        "target": "e8",
        "text": "Buy milk",
        "submit": "false",
    }

    normalized = normalize_browser_arguments(
        tool_name="browser_type",
        arguments=raw_arguments,
        tools=tools,
    )

    assert normalized["submit"] is False


def test_normalizer_removes_optional_none_argument():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": "Click an element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "button": {
                            "type": "string",
                        },
                        "modifiers": {
                            "type": "array",
                        },
                    },
                    "required": [
                        "target",
                    ],
                },
            },
        }
    ]

    raw_arguments = {
        "target": "e21",
        "button": "left",
        "modifiers": None,
    }

    normalized = normalize_browser_arguments(
        tool_name="browser_click",
        arguments=raw_arguments,
        tools=tools,
    )

    assert normalized == {
        "target": "e21",
        "button": "left",
    }

    assert "modifiers" not in normalized


def test_normalizer_rejects_required_none_argument():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": "Click an element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "button": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "target",
                    ],
                },
            },
        }
    ]

    raw_arguments = {
        "target": None,
        "button": "left",
    }

    with pytest.raises(
        ValueError,
        match="target is required",
    ):
        normalize_browser_arguments(
            tool_name="browser_click",
            arguments=raw_arguments,
            tools=tools,
        )


def test_normalizer_rejects_unknown_argument():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": "Click an element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "button": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "target",
                    ],
                },
            },
        }
    ]

    raw_arguments = {
        "target": "e21",
        "button": "left",
        "foo": "bar",
    }

    with pytest.raises(
        ValueError,
        match="Unknown argument 'foo'",
    ):
        normalize_browser_arguments(
            tool_name="browser_click",
            arguments=raw_arguments,
            tools=tools,
        )


def test_normalizer_rejects_invalid_boolean_string():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_type",
                "description": "Type text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "text": {"type": "string"},
                        "submit": {"type": "boolean"},
                    },
                    "required": [
                        "target",
                        "text",
                    ],
                },
            },
        }
    ]

    with pytest.raises(
        ValueError,
        match="submit must be boolean",
    ):
        normalize_browser_arguments(
            tool_name="browser_type",
            arguments={
                "target": "e8",
                "text": "Buy milk",
                "submit": "yes",
            },
            tools=tools,
        )


def test_normalizer_rejects_unknown_tool():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": "Click",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                    },
                    "required": ["target"],
                },
            },
        }
    ]

    with pytest.raises(
        ValueError,
        match="Unknown browser tool",
    ):
        normalize_browser_arguments(
            tool_name="browser_magic",
            arguments={
                "target": "e21",
            },
            tools=tools,
        )


def test_validator_rejects_wrong_text_for_add():

    from src.browser_agent.browser_agent import (
        validate_browser_action,
    )

    snapshot = """
    - textbox "What needs to be done?" [ref=e8]
    """

    with pytest.raises(
        ValueError,
        match="Type only 'Buy milk'",
    ):
        validate_browser_action(
            goal="Add Buy milk to the todo list",
            tool_name="browser_type",
            arguments={
                "target": "e8",
                "text": "Add Buy milk to the todo list",
                "submit": True,
            },
            snapshot=snapshot,
        )


def test_validator_rejects_middle_click():

    from src.browser_agent.browser_agent import (
        validate_browser_action,
    )

    snapshot = """
    - checkbox "Toggle Todo" [ref=e21]
    - generic [ref=e22]: Buy milk
    """

    with pytest.raises(
        ValueError,
        match="button='left'",
    ):
        validate_browser_action(
            goal="Mark Buy milk as complete",
            tool_name="browser_click",
            arguments={
                "target": "e21",
                "button": "middle",
            },
            snapshot=snapshot,
        )


def test_normalizer_converts_array_string():

    from src.browser_agent.browser_agent import (
        normalize_browser_arguments,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": ("Click an element"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                        },
                        "button": {
                            "type": "string",
                        },
                        "doubleClick": {
                            "type": "boolean",
                        },
                        "modifiers": {
                            "type": "array",
                        },
                    },
                    "required": [
                        "target",
                    ],
                },
            },
        }
    ]

    raw_arguments = {
        "target": "e21",
        "button": "left",
        "doubleClick": "false",
        "modifiers": "[]",
    }

    result = normalize_browser_arguments(
        tool_name="browser_click",
        arguments=raw_arguments,
        tools=tools,
    )

    assert result == {
        "target": "e21",
        "button": "left",
        "doubleClick": False,
        "modifiers": [],
    }
