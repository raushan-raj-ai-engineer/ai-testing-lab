from typing import TypedDict


class KnowledgeDocument(TypedDict):
    id: str
    text: str


DOCUMENTS: list[KnowledgeDocument] = [
    {
        "id": "REFUND_POLICY",
        "text": ("Customers can request a refund within 30 days of purchase."),
    },
    {
        "id": "PASSWORD_POLICY",
        "text": (
            "Users can reset their password using "
            "the Forgot Password link on the login page."
        ),
    },
    {
        "id": "SHIPPING_POLICY",
        "text": ("Standard shipping normally takes 3 to 5 business days."),
    },
]
