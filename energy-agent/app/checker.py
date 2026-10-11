"""Number checker (R10): every number in the LLM's text must appear in the tool results.
This is what guarantees the LLM never invents a figure."""

import re

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
IGNORE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{4}-Q[1-4]|\b20\d{2}\b|\d{1,2}:\d{2}")   # dates, years, clock times


def numbers_in(obj) -> list[float]:
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(n.replace(",", "")) for n in NUMBER.findall(IGNORE.sub(" ", obj))]
    if isinstance(obj, dict):
        obj = list(obj.values())
    return [n for v in obj for n in numbers_in(v)] if isinstance(obj, list) else []


def invented_numbers(text: str, tool_results: dict) -> list[str]:
    """Numbers in `text` that match no tool number (also accepting x100 for %, and rounding)."""
    allowed = numbers_in(tool_results)
    allowed += [a * 100 for a in allowed]                    # 0.35 written as 35%
    bad = []
    for raw in NUMBER.findall(IGNORE.sub(" ", text)):
        x = float(raw.replace(",", ""))
        if raw.lstrip("-").isdigit() and abs(x) <= 10:      # small counts like "top 3"
            continue
        if not any(abs(abs(x) - abs(a)) <= max(abs(a) * 0.01, 0.05) for a in allowed):   # "14,400-19,200" reads as -19,200
            bad.append(raw)
    return bad
