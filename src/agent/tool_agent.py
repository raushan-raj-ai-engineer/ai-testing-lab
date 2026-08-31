from dataclasses import dataclass
from typing import Any, Literal

from mcp import Client
from mcp_types import TextContent
from ollama import chat
from pydantic import BaseModel

MODEL = "llama3.2"


@dataclass
class AgentDecision:
    tool_name: str | None
    arguments: dict[str, Any]


class ToolRoute(BaseModel):
    tool_name: Literal[
        "none",
        "calculate_pass_rate",
        "release_decision",
    ]

    reason: str


@dataclass
class AgentRunResult:
    tool_name: str | None
    arguments: dict[str, Any]
    tool_result: dict[str, Any] | None
    final_answer: str


def route_tool(
    user_request: str,
) -> ToolRoute:
    """
    Decide whether the user request requires
    an MCP tool before exposing tools to the model.
    """

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a QA agent router. "
                    "Decide whether the user's request "
                    "requires one of the available QA tools. "
                    "Choose 'calculate_pass_rate' ONLY when "
                    "the user wants an actual pass-rate "
                    "calculation and provides both passed "
                    "and total test counts. "
                    "Choose 'release_decision' ONLY when "
                    "the user wants an actual release or "
                    "deployment decision and provides both "
                    "the pass rate and critical failure count. "
                    "Choose 'none' for explanations, definitions, "
                    "conceptual questions, examples, general "
                    "conversation, or when required numeric "
                    "information is missing. "
                    "Never invent missing information."
                ),
            },
            {
                "role": "user",
                "content": user_request,
            },
        ],
        format=ToolRoute.model_json_schema(),
        options={
            "temperature": 0,
        },
    )

    content = response.message.content

    if content is None:
        raise ValueError("Tool router returned no content.")

    return ToolRoute.model_validate_json(content)


async def get_ollama_tools(
    client: Client,
) -> list[dict[str, Any]]:
    """
    Discover MCP tools and convert them into
    Ollama-compatible tool schemas.
    """

    result = await client.list_tools()

    tools: list[dict[str, Any]] = []

    for tool in result.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (tool.description or ""),
                "parameters": tool.input_schema,
            },
        })

    return tools


def normalize_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:

    tool_definition = next(
        (tool for tool in tools if tool["function"]["name"] == tool_name),
        None,
    )

    if tool_definition is None:
        raise ValueError(f"Unknown tool selected: {tool_name}")

    parameters = tool_definition["function"]["parameters"]

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

    # ---------------------------------------------
    # Reject unknown arguments
    # ---------------------------------------------

    for name in arguments:
        if name not in properties:
            raise ValueError(f"Unknown argument '{name}' for tool '{tool_name}'.")

    # ---------------------------------------------
    # Normalize values
    # ---------------------------------------------

    for name, value in arguments.items():
        property_schema = properties[name]

        expected_type = property_schema.get("type")

        if expected_type == "integer":
            try:
                normalized_value = int(value)

            except (TypeError, ValueError) as exc:
                raise ValueError(f"Argument '{name}' must be an integer.") from exc

        elif expected_type == "number":
            try:
                normalized_value = float(value)

            except (TypeError, ValueError) as exc:
                raise ValueError(f"Argument '{name}' must be a number.") from exc

        elif expected_type == "boolean":
            if isinstance(value, bool):
                normalized_value = value

            elif isinstance(value, str):
                lowered = value.lower()

                if lowered == "true":
                    normalized_value = True

                elif lowered == "false":
                    normalized_value = False

                else:
                    raise ValueError(f"Argument '{name}' must be a boolean.")

            else:
                raise ValueError(f"Argument '{name}' must be a boolean.")

        else:
            normalized_value = value

        # -----------------------------------------
        # Numeric constraints from MCP schema
        # -----------------------------------------

        if expected_type in {
            "integer",
            "number",
        }:
            minimum = property_schema.get("minimum")

            maximum = property_schema.get("maximum")

            exclusive_minimum = property_schema.get("exclusiveMinimum")

            exclusive_maximum = property_schema.get("exclusiveMaximum")

            if minimum is not None and normalized_value < minimum:
                raise ValueError(f"Argument '{name}' must be >= {minimum}.")

            if maximum is not None and normalized_value > maximum:
                raise ValueError(f"Argument '{name}' must be <= {maximum}.")

            if exclusive_minimum is not None and normalized_value <= exclusive_minimum:
                raise ValueError(f"Argument '{name}' must be > {exclusive_minimum}.")

            if exclusive_maximum is not None and normalized_value >= exclusive_maximum:
                raise ValueError(f"Argument '{name}' must be < {exclusive_maximum}.")

        normalized[name] = normalized_value

    # ---------------------------------------------
    # Required arguments
    # ---------------------------------------------

    missing = required - set(normalized.keys())

    if missing:
        missing_names = ", ".join(sorted(missing))

        raise ValueError(
            f"Missing required arguments for tool '{tool_name}': {missing_names}"
        )

    return normalized


def choose_tool(
    user_request: str,
    tools: list[dict[str, Any]],
) -> AgentDecision:
    """
    Route the request first.

    Only expose an MCP tool to the model when
    the router determines that a tool is needed.
    """

    route = route_tool(user_request)

    # --------------------------------------------------
    # No tool required
    # --------------------------------------------------

    if route.tool_name == "none":
        return AgentDecision(
            tool_name=None,
            arguments={},
        )

    # --------------------------------------------------
    # Find only the selected MCP tool
    # --------------------------------------------------

    selected_tools = [
        tool for tool in tools if (tool["function"]["name"] == route.tool_name)
    ]

    if not selected_tools:
        raise ValueError(f"Router selected unknown tool: {route.tool_name}")

    # --------------------------------------------------
    # Generate arguments for selected tool
    # --------------------------------------------------

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a QA automation agent. "
                    "The correct tool has already been selected. "
                    "Call the supplied tool using only information "
                    "explicitly provided by the user. "
                    "Do not invent, assume, default, or fabricate "
                    "tool arguments."
                ),
            },
            {
                "role": "user",
                "content": user_request,
            },
        ],
        tools=selected_tools,
        options={
            "temperature": 0,
        },
    )

    tool_calls = response.message.tool_calls

    if not tool_calls:
        return AgentDecision(
            tool_name=None,
            arguments={},
        )

    tool_call = tool_calls[0]

    tool_name = tool_call.function.name

    raw_arguments = dict(tool_call.function.arguments)

    normalized_arguments = normalize_arguments(
        tool_name=tool_name,
        arguments=raw_arguments,
        tools=tools,
    )

    return AgentDecision(
        tool_name=tool_name,
        arguments=normalized_arguments,
    )


def get_tool_result_text(
    result: Any,
) -> str:
    """
    Extract model-readable text from an MCP tool result.
    """

    text_parts: list[str] = []

    for block in result.content:
        if isinstance(
            block,
            TextContent,
        ):
            text_parts.append(block.text)

    return "\n".join(text_parts)


async def run_agent(
    user_request: str,
    client: Client,
) -> AgentRunResult:
    """
    Run the complete QA agent flow:

    user
      -> route
      -> tool selection
      -> argument validation
      -> MCP execution
      -> final LLM answer
    """

    tools = await get_ollama_tools(client)

    # ---------------------------------------------
    # Step 1: Decide whether a tool is required
    # ---------------------------------------------

    route = route_tool(user_request)

    # ---------------------------------------------
    # Step 2: No-tool path
    # ---------------------------------------------

    if route.tool_name == "none":
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a QA automation "
                        "assistant. Answer the user's "
                        "question clearly and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": user_request,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        final_answer = response.message.content

        if final_answer is None:
            raise ValueError("Agent returned no final answer.")

        return AgentRunResult(
            tool_name=None,
            arguments={},
            tool_result=None,
            final_answer=final_answer,
        )

    # ---------------------------------------------
    # Step 3: Expose only routed tool
    # ---------------------------------------------

    selected_tools = [
        tool for tool in tools if (tool["function"]["name"] == route.tool_name)
    ]

    if not selected_tools:
        raise ValueError(f"Router selected unknown tool: {route.tool_name}")

    # ---------------------------------------------
    # Step 4: Ask model to generate tool call
    # ---------------------------------------------

    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "You are a QA automation agent. "
                "Use the supplied tool using only "
                "information explicitly provided "
                "by the user. "
                "Never invent tool arguments."
            ),
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    response = chat(
        model=MODEL,
        messages=messages,
        tools=selected_tools,
        options={
            "temperature": 0,
        },
    )

    tool_calls = response.message.tool_calls

    if not tool_calls:
        raise ValueError("Agent was routed to a tool but generated no tool call.")

    # ---------------------------------------------
    # Step 5: Use first selected tool call
    # ---------------------------------------------

    tool_call = tool_calls[0]

    tool_name = tool_call.function.name

    raw_arguments = dict(tool_call.function.arguments)

    arguments = normalize_arguments(
        tool_name=tool_name,
        arguments=raw_arguments,
        tools=tools,
    )

    # ---------------------------------------------
    # Step 6: Execute actual MCP tool
    # ---------------------------------------------

    mcp_result = await client.call_tool(
        tool_name,
        arguments,
    )

    if mcp_result.is_error:
        error_text = get_tool_result_text(mcp_result)

        raise ValueError(f"MCP tool '{tool_name}' failed: {error_text}")

    tool_result = mcp_result.structured_content

    tool_text = get_tool_result_text(mcp_result)

    # ---------------------------------------------
    # Step 7:
    # Send tool call + result back to LLM
    # ---------------------------------------------

    messages.append(response.message)

    messages.append({
        "role": "tool",
        "tool_name": tool_name,
        "content": tool_text,
    })

    # ---------------------------------------------
    # Step 8: Final response
    # ---------------------------------------------

    final_response = chat(
        model=MODEL,
        messages=messages,
        tools=selected_tools,
        options={
            "temperature": 0,
        },
    )

    final_answer = final_response.message.content

    if final_answer is None:
        raise ValueError("Agent returned no final answer.")

    return AgentRunResult(
        tool_name=tool_name,
        arguments=arguments,
        tool_result=tool_result,
        final_answer=final_answer,
    )
