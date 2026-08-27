"""Token-level Jaccard similarity (whitespace tokenization, no embeddings).

    jaccard(a, b) = |tokens(a) ∩ tokens(b)| / |tokens(a) ∪ tokens(b)|

Used by Stage 7 for both the contamination check (against source benchmark
questions) and the intra-bank deduplication.
"""
from __future__ import annotations


def tokenize(text: str) -> frozenset[str]:
    """Whitespace-tokenize and lowercase; returns a set of tokens."""
    if not text:
        return frozenset()
    return frozenset(str(text).lower().split())


def jaccard(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    """Jaccard similarity between two token sets (0.0 when either is empty)."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def max_jaccard_against(
    tokens: frozenset[str],
    corpus: list[tuple[str, frozenset[str]]],
    threshold: float,
) -> tuple[float, str | None]:
    """Return the highest Jaccard similarity of *tokens* against a corpus.

    *corpus* is a list of (label, token_set) pairs. A cheap length-ratio
    prefilter skips pairs whose Jaccard mathematically cannot exceed
    *threshold* (J ≤ min(|A|,|B|) / max(|A|,|B|)).

    Returns (best_similarity, label_of_best_or_None).
    """
    if not tokens:
        return 0.0, None
    best = 0.0
    best_label: str | None = None
    for label, other in corpus:
        if not other:
            continue
        shorter, longer = sorted((len(tokens), len(other)))
        if shorter <= threshold * longer:
            continue  # Jaccard can never exceed threshold here
        sim = jaccard(tokens, other)
        if sim > best:
            best = sim
            best_label = label
    return best, best_label
