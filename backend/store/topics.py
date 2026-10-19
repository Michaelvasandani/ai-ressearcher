"""Topic keys and similarity for the findings store.

A topic key is the normalized content words of a sub-question: lowercased,
stop words dropped, lightly stemmed, deduplicated and sorted. Two questions
match when the Jaccard similarity of their token sets clears a threshold.

The same functions score candidates for the in-memory and the Postgres store
(Postgres only narrows candidates with a `tokens && $1` array-overlap query on
a GIN index), so both behave identically.
"""

from __future__ import annotations

import re

DEFAULT_TTL_DAYS = 30.0  # findings older than this are researched again
DEFAULT_MIN_SIMILARITY = 0.5  # Jaccard over topic tokens

_WORD = re.compile(r"[a-z0-9]+")

STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her
    here hers how i if in into is it its itself just me more most my no nor not now of off on once only or other
    our out over own same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why will with would you
    your yours versus vs per compared whether there's effect effects
    """.split()
)


def _stem(word: str) -> str:
    """A deliberately small stemmer: plural and common verb endings only."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def topic_tokens(text: str) -> list[str]:
    """Sorted, unique, stemmed content words."""
    words = _WORD.findall(text.lower())
    return sorted({_stem(w) for w in words if w not in STOP_WORDS and len(w) > 1})


def topic_key(text: str) -> str:
    return " ".join(topic_tokens(text))


def similarity(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
