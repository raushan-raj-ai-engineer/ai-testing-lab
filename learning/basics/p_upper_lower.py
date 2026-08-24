# Find Upper and Lower Case Count


def count_upper_lower(text: str) -> dict:
    upper_count = 0
    lower_count = 0
    for char in text:
        if char.isupper():
            upper_count += 1
        elif char.islower():
            lower_count += 1
    return {"upper_count": upper_count, "lower_count": lower_count}


text_input = "Hello World"
result = count_upper_lower(text_input)
print(result)  # Output: {'upper_count': 2, 'lower_count': 8}
