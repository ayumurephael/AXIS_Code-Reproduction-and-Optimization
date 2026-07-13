import re


def score_from_text(text):
    # Accept both plain "Score: 4" and appendix markdown "**Score:** 4".
    hits = re.findall(r"(?:\*\*)?Score(?:\*\*)?\s*:\s*(?:\*\*)?\s*\[?([1-5])", text or "", flags=re.I)
    return int(hits[-1]) if hits else None
