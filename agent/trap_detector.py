"""
trap_detector.py
----------------
Zero-new-dependency semantic trap detector.

This uses the existing scikit-learn TF-IDF stack to catch prompt variants of
logic traps that exact regexes miss. It is intentionally small and static so it
adds no model weights and no Docker size.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TRAP_EXAMPLES = [
    "A knight always tells the truth and a knave always lies. Determine who is who.",
    "One person always tells the truth and another person always lies. What question should you ask?",
    "People are seated in a row with left of right of between and adjacent constraints.",
    "Arrange five people in a line given clues about who sits next to whom.",
    "If and only if logic puzzle with unless, all, some, no, and must be true.",
    "Which of the following is most likely except not true based on the passage?",
    "A lies on certain days and tells truth on other days. What day is it?",
    "Two doors, one guard lies, one guard tells the truth, find the safe door.",
    "All A are B, some B are C, determine whether conclusion necessarily follows.",
    "A is taller than B, B is not shorter than C, infer ordering from constraints.",
]

REGEX_TRAPS = [
    r"\b(knight|knave)\b",
    r"\balways\s+(?:lies?|tells?\s+the\s+truth)\b",
    r"\btruth[- ]?teller\b|\bliar\b",
    r"\b(left|right)\s+of\b|\bbetween\b|\badjacent\b|\bnext\s+to\b",
    r"\bseated\b|\bsitting\b|\barranged\b|\bin\s+a\s+(?:row|line)\b",
    r"\bif\s+and\s+only\s+if\b|\bunless\b|\bmust\s+be\s+true\b",
    r"\ball\b.*\bsome\b|\bsome\b.*\bno\b|\bno\b.*\ball\b",
    r"\bwhich\s+of\s+the\s+following\b|\bmost\s+likely\b|\bexcept\b",
]


class TrapDetector:
    def __init__(self, threshold: float = 0.33):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )
        self._matrix = self.vectorizer.fit_transform(TRAP_EXAMPLES)

    def score(self, prompt: str) -> float:
        if not prompt.strip():
            return 0.0
        vec = self.vectorizer.transform([prompt])
        sims = cosine_similarity(vec, self._matrix)[0]
        return float(np.max(sims)) if len(sims) else 0.0

    def is_trap(self, prompt: str) -> tuple[bool, str, float]:
        lowered = prompt.lower()
        if self._looks_like_code_task(lowered):
            return False, "code-task", 0.0

        for pattern in REGEX_TRAPS:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True, "regex", 1.0

        score = self.score(prompt)
        return score >= self.threshold, "tfidf", score

    def _looks_like_code_task(self, lowered: str) -> bool:
        return bool(
            re.search(
                r"\b(function|python|code|implement|debug|bug|syntaxerror|traceback)\b|"
                r"\bdef\s+\w+\s*\(|\bclass\s+\w+",
                lowered,
            )
        )
