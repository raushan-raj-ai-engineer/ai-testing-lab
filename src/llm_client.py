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

Follow these business rules exactly.

1. Duplicate payment, duplicate transaction, charged twice,
   or billed twice:
   category = billing
   priority = high
   needs_human = true

2. Password or login problem:
   category = account
   priority = medium
   needs_human = true

3. Application crash or software error:
   category = technical
   priority = high
   needs_human = true

4. If the customer says the problem is resolved,
   everything is working, thanks for the help,
   or no further support is required:
   category = other
   priority = low
   needs_human = false

5. Otherwise:
   category = other
   priority = low
   needs_human = false

Reason rules:
- Write exactly one short sentence.
- Clearly state the core issue described by the customer.
- Do not return generic warnings or generic explanations.
- Do not invent information.
- Different wording is allowed, but preserve the meaning.

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
