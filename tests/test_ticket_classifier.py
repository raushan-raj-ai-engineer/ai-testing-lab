from src.llm_client import classify_ticket


def test_duplicate_charge_ticket():

    ticket = """
    My credit card was charged twice for the same order.
    I need the extra charge refunded immediately.
    """

    result = classify_ticket(ticket)

    assert result.category == "billing"
    assert result.priority == "high"
    assert result.needs_human is True


def test_output_schema():

    result = classify_ticket("I forgot my password and cannot login.")

    assert result.category in {"billing", "technical", "account", "other"}

    assert result.priority in {"low", "medium", "high"}

    assert isinstance(result.needs_human, bool)

    assert result.reason


def test_login_problem_classified_as_account():

    result = classify_ticket("I reset my password but still cannot login.")

    assert result.category == "account"


def test_duplicate_charge_classification_is_consistent():

    ticket = """
    My card was charged twice for one purchase.
    """

    results = []

    for _ in range(5):
        result = classify_ticket(ticket)
        results.append(result.category)

    assert all(category == "billing" for category in results)
