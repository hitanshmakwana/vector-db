"""Tokenisation for the BM25 index.

Deliberately minimal, per ARCHITECTURE.md §4 and LEARNING.md layer 3:
lowercase, strip punctuation, split on whitespace. No stemming, no stopword
removal. Modern practice for BM25 baselines is close to "just lowercase and
split", and every extra step is a knob that would need tuning and evaluating.

Both are natural v2 additions and both are configurable in production engines,
which is exactly why they are not hardcoded here.
"""

from __future__ import annotations

import re

__all__ = ["tokenize", "TOKEN_RE"]

#: A token is a maximal run of word characters. Unicode-aware by default in
#: Python 3, so accented text and non-Latin scripts survive tokenisation.
#: Underscore is excluded (``\w`` includes it) so ``snake_case`` splits.
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split.

    >>> tokenize("The quick, brown FOX -- jumps!")
    ['the', 'quick', 'brown', 'fox', 'jumps']
    >>> tokenize("")
    []
    """
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())
