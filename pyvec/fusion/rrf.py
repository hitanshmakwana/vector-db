"""Reciprocal Rank Fusion.

    RRF_score(d) = sum over rankers of  1 / (k + rank_of_d_in_that_ranker)

Ranks are 1-based. Documents missing from a ranker's list simply contribute
nothing from it. ``k = 60`` is the constant from Cormack et al. (2009) and the
default in Elasticsearch, OpenSearch and Vespa.

Why this and not a weighted score sum (ADR-003): BM25 scores are unbounded,
cosine is bounded to ``[-1, 1]``, and dot product scales with vector norms.
Normalising across them is a research problem in its own right — and any
normalisation you pick is a function of the score *distribution*, so it shifts
under you as the corpus changes. RRF throws the scores away and keeps only the
ordering, which is the one thing both retrievers agree on the meaning of.

What it costs: there is no knob to say "trust the dense side twice as much".
``k`` is the only tuning parameter, and it controls how sharply top ranks are
favoured over the tail — smaller ``k`` weights rank 1 much more heavily than
rank 10, larger ``k`` flattens the curve.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, TypeVar

__all__ = ["DEFAULT_RRF_K", "FusedResult", "reciprocal_rank_fusion", "rrf"]

#: The canonical constant. Robust across datasets; do not tune it without data.
DEFAULT_RRF_K = 60

T = TypeVar("T")


@dataclass(slots=True)
class FusedResult:
    """One fused document, with provenance.

    The per-ranker ranks are carried through because RRF returns no comparable
    similarity score, so they are the only way to explain *why* a document
    ranked where it did. API_SPEC surfaces them as ``dense_rank`` /
    ``sparse_rank`` "for debuggability".
    """

    id: T  # type: ignore[valid-type]
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def num_rankers(self) -> int:
        return len(self.ranks)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[T]] | Iterable[Sequence[T]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse ranked id lists into one ranking.

    Args:
        rankings: either a mapping of ranker name -> ranked ids (best first), or
            a plain iterable of ranked lists, in which case rankers are named
            ``ranker_0``, ``ranker_1``, ... Each list must already be sorted by
            that retriever's own notion of relevance; RRF never looks at scores.
        k: the RRF constant.
        top_k: truncate the output. ``None`` returns everything either ranker
            found.

    Returns:
        :class:`FusedResult` objects sorted by descending fused score. Ties break
        on the number of rankers that retrieved the document (agreement is
        evidence), then on the best single rank achieved, then on the id — so the
        output is fully deterministic, which matters for reproducible evaluation.
    """
    if k < 0:
        raise ValueError(f"RRF k must be non-negative, got {k}")

    named: dict[str, Sequence[T]]
    if isinstance(rankings, Mapping):
        named = dict(rankings)
    else:
        named = {f"ranker_{i}": r for i, r in enumerate(rankings)}

    scores: dict[T, float] = defaultdict(float)
    ranks: dict[T, dict[str, int]] = defaultdict(dict)

    for name, ranking in named.items():
        seen: set[T] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            # A ranker listing the same id twice would double-count it.
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
            ranks[doc_id][name] = rank

    fused = [
        FusedResult(id=doc_id, score=score, ranks=ranks[doc_id])
        for doc_id, score in scores.items()
    ]
    fused.sort(
        key=lambda r: (
            -r.score,
            -r.num_rankers,
            min(r.ranks.values()),
            str(r.id),
        )
    )
    return fused if top_k is None else fused[:top_k]


def rrf(
    rankings: list[list[T]], k: int = DEFAULT_RRF_K, top_k: int = 10
) -> list[T]:
    """Ids only — the compact form from ARCHITECTURE.md §5."""
    return [r.id for r in reciprocal_rank_fusion(rankings, k=k, top_k=top_k)]
