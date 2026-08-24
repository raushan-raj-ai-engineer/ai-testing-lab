from typing import Literal

from ollama import chat
from pydantic import BaseModel


class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    needs_human: bool
    reason: str


def classify_ticket(ticket: str) -> TicketClassification:
    system_prompt = """
    You are a customer-support ticket classifier.

    Follow these business rules:

    1. Duplicate payment or duplicate charge:
       category = billing
       priority = high
       needs_human = true

    2. Password/login problem:
       category = account

    3. Application crash or software error:
       category = technical

    Return only data matching the provided schema.
    """

    response = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ticket},
        ],
        format=TicketClassification.model_json_schema(),
        options={"temperature": 0},
    )
    content = response.message.content

    if content is None:
        raise ValueError("LLM returned empty response content")

    return TicketClassification.model_validate_json(content)
