"""
evaluator.py
------------
Zero-token programmatic validators for each domain.
Uses Python builtins only — no LLM calls.
All validators return (is_valid: bool, cleaned_output: str).
"""
import ast
import json
import re
import sympy
from pydantic import BaseModel, ValidationError


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schema for NER output
# ──────────────────────────────────────────────────────────────────────────────
class NEROutput(BaseModel):
    person:   list[str]
    org:      list[str]
    location: list[str]
    date:     list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _strip_markdown(text: str, lang: str = "") -> str:
    """Remove ```lang ... ``` fences from model output."""
    fence = f"```{lang}"
    if fence in text:
        text = text.split(fence, 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        elif len(parts) == 2:
            text = parts[1]
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Domain validators
# ──────────────────────────────────────────────────────────────────────────────
def validate_ner(response: str) -> tuple[bool, str]:
    """Validate NER JSON schema: must have person, org, location, date keys."""
    cleaned = _strip_markdown(response, "json")
    try:
        data = json.loads(cleaned)
        NEROutput(**data)
        return True, cleaned
    except (json.JSONDecodeError, ValidationError, Exception):
        return False, response


def validate_python_syntax(response: str) -> tuple[bool, str]:
    """Validate that the response contains syntactically correct Python."""
    cleaned = _strip_markdown(response, "python")
    if not cleaned:
        return False, response
    try:
        ast.parse(cleaned)
        return True, cleaned
    except SyntaxError:
        return False, response


def validate_sentiment(response: str) -> tuple[bool, str]:
    """Validate that the response contains a known sentiment label."""
    lower = response.lower().strip()
    for label in ("positive", "negative", "neutral"):
        if label in lower:
            return True, label  # Return the clean label only
    return False, response


def validate_math(prompt: str, response: str) -> tuple[bool, str]:
    """
    For simple math, attempt to compute the answer directly with sympy.
    Returns (True, correct_answer) if solvable programmatically.
    Otherwise (False, response) to fall through to LLM.
    """
    answer = _try_direct_math(prompt)
    if answer is not None:
        return True, answer

    # If programmatic fails, check if response contains a number at least
    has_number = bool(re.search(r"\d", response))
    return has_number, response


def validate_summarization(response: str) -> tuple[bool, str]:
    """
    Summarization is hard to validate programmatically.
    Just check it's non-empty and has a reasonable length.
    """
    cleaned = response.strip()
    if len(cleaned.split()) >= 10:   # At least 10 words
        return True, cleaned
    return False, response


def validate_factual(response: str) -> tuple[bool, str]:
    """Factual answers are hard to validate — just check non-empty."""
    cleaned = response.strip()
    return len(cleaned) > 0, cleaned


def validate_logic(response: str) -> tuple[bool, str]:
    """Logic answers — check non-empty and not just a 'yes'/'no' fragment."""
    cleaned = response.strip()
    return len(cleaned.split()) >= 3, cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Direct Math Solver (zero LLM tokens for arithmetic/algebra)
# ──────────────────────────────────────────────────────────────────────────────
def _try_direct_math(prompt: str) -> str | None:
    """
    Attempt to solve math questions directly without any LLM.
    Returns the answer string if solvable, else None.
    """
    # Strategy 1: Pure arithmetic expression in the prompt
    arith = re.search(r"[\d\s\+\-\*\/\(\)\.\^%]+", prompt)
    if arith:
        expr = arith.group().strip().replace("^", "**")
        try:
            result = sympy.sympify(expr).evalf()
            return str(result)
        except Exception:
            pass

    # Strategy 2: Simple percentage — "X% of Y"
    pct = re.search(r"(\d+\.?\d*)\s*%\s+of\s+(\d+\.?\d*)", prompt, re.IGNORECASE)
    if pct:
        try:
            pct_val = float(pct.group(1))
            base    = float(pct.group(2))
            return str(pct_val / 100 * base)
        except Exception:
            pass

    # Strategy 3: Linear equation "ax + b = c" → solve for x
    eq = re.search(r"(\d*)\s*x\s*\+\s*(\d+)\s*=\s*(\d+)", prompt, re.IGNORECASE)
    if eq:
        try:
            a = float(eq.group(1) or 1)
            b = float(eq.group(2))
            c = float(eq.group(3))
            return str((c - b) / a)
        except Exception:
            pass

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ──────────────────────────────────────────────────────────────────────────────
def validate(domain: str, prompt: str, response: str) -> tuple[bool, str]:
    """Central dispatcher. Returns (is_valid, cleaned_response)."""
    dispatch = {
        "ner":           lambda: validate_ner(response),
        "debugging":     lambda: validate_python_syntax(response),
        "codegen":       lambda: validate_python_syntax(response),
        "sentiment":     lambda: validate_sentiment(response),
        "math":          lambda: validate_math(prompt, response),
        "summarization": lambda: validate_summarization(response),
        "factual":       lambda: validate_factual(response),
        "logic":         lambda: validate_logic(response),
    }
    fn = dispatch.get(domain, lambda: (True, response))
    return fn()


def try_solve_locally(domain: str, prompt: str) -> str | None:
    """
    For some domains, we can answer WITHOUT any model at all.
    Returns the answer string, or None if a model is needed.
    """
    if domain == "math":
        return _try_direct_math(prompt)
    return None
