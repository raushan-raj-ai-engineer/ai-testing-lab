# Character Count — Important Program


def character_count(text: str) -> dict:
    """
    Count the number of characters in a given text.

    Args:
        text (str): The input text to count characters from.

    Returns:
        dict: A dictionary containing the character count.
    """
    character_counts = {}
    for char in text:
        if char in character_counts:
            character_counts[char] = character_counts[char] + 1
        else:
            character_counts[char] = 1
    return {"character_counts": character_counts}


count_result = character_count("Hello World")
print(
    count_result
)  # Output: {'character_counts': {'H': 1, 'e': 1, 'l': 3, 'o': 2, ',': 1, ' ': 1, 'W': 1, 'r': 1, 'd': 1, '!': 1}}
