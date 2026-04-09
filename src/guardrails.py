import re

INJECTION_PATTERNS = [
    r"ignore all previous instructions",
    r"forget all previous instructions",
    r"system prompt",
    r"developer mode",
    r"jailbreak",
    r"bypass",
]


def check_input_safety(text: str):
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS):
        return False, "Blocked: prompt injection detected."
    if len(text) > 800:
        return False, "Blocked: message too long."
    return True, "OK"


def check_output_safety(text: str):
    if re.search(r"\b\d{10}\b", text):
        return "Blocked: sensitive information detected."
    return text
