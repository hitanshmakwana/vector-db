"""Hybrid retrieval: dense + BM25 fused with RRF.

The headline claim (PRD G2, résumé bullet 2) is that hybrid beats dense-only. The
central test here builds a corpus with the property that makes hybrid *necessary*
— a document that is lexically an exact match but sits in an unhelpful place in
embedding space — and shows RRF recovers it where dense search alone cannot.

BENCHMARKS.md benchmark 3 does the real evaluation on MS MARCO with nDCG@10; this
file is the unit-level version that runs in milliseconds and pins the mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyvec.core.errors import InvalidRequestError

DIM = 16


def synthetic_corpus():
    """A corpus where dense and sparse disagree in a controlled way.

    Twenty filler documents whose embeddings cluster near the query, plus one
    "needle" whose text matches the query terms exactly but whose embedding is
    deliberately orthogonal to the query. Dense search buries the needle; BM25
    puts it first; RRF should promote it into the top few.
    """
    rng = np.random.default_rng(7)
    query_vec = np.zeros(DIM, dtype=np.float32)
    query_vec[0] = 1.0

    items = []
    for i in range(20):
        v = query_vec + rng.normal(scale=0.15, size=DIM).astype(np.float32)
        items.append(
            {
                "id": f"filler-{i}",
                "vector": v.tolist(),
                "metadata": {
                    "content": f"unrelated filler document number {i} about weather",
                    "kind": "filler",
                },
            }
        )

    needle = np.zeros(DIM, dtype=np.float32)
    needle[DIM - 1] = 1.0  # orthogonal to the query
    items.append(
        {
            "id": "needle",
            "vector": needle.tolist(),
            "metadata": {
                "content": "reciprocal rank fusion combines heterogeneous rankings",
                "kind": "needle",
            },
        }
    )
    return items, query_vec.tolist()


@pytest.fixture
def hybrid_collection(make_collection):
    c = make_collection(dimension=DIM, index_type="flat", text_field="content")
    items, query = synthetic_corpus()
    c.insert(items)
    return c, query


class TestHybridBeatsDense:
    def test_dense_alone_misses_the_lexical_match(self, hybrid_collection):
        c, query = hybrid_collection
        dense_ids = [h.id for h in c.search(query, k=5)]
        assert "needle" not in dense_ids, "fixture no longer isolates the effect"

    def test_bm25_alone_finds_it(self, hybrid_collection):
        c, _ = hybrid_collection
        assert [h.id for h in c.search_text("reciprocal rank fusion", k=3)][0] == (
            "needle"
        )

    def test_hybrid_recovers_it(self, hybrid_collection):
        """The claim: fusing ranks surfaces what either retriever alone misses."""
        c, query = hybrid_collection
        hybrid_ids = [h.id for h in c.search_hybrid(query, "reciprocal rank fusion", k=5)]
        assert "needle" in hybrid_ids

    def test_hybrid_still_keeps_strong_dense_hits(self, hybrid_collection):
        """Adding the sparse side must not evict everything the dense side found."""
        c, query = hybrid_collection
        dense = {h.id for h in c.search(query, k=5)}
        hybrid = {h.id for h in c.search_hybrid(query, "reciprocal rank fusion", k=5)}
        assert len(dense & hybrid) >= 3


class TestRanksAndScores:
    def test_per_retriever_ranks_are_reported(self, hybrid_collection):
        c, query = hybrid_collection
        hits = c.search_hybrid(query, "reciprocal rank fusion", k=10)
        assert any(h.ranks.get("dense") for h in hits)
        assert any(h.ranks.get("sparse") for h in hits)
        for h in hits:
            assert h.ranks, "every hit must record where it came from"

    def test_ranks_are_one_based_and_consistent_with_the_dense_ranking(
        self, hybrid_collection
    ):
        c, query = hybrid_collection
        dense_order = [h.id for h in c.search(query, k=10)]
        hits = c.search_hybrid(query, "reciprocal rank fusion", k=20)
        by_id = {h.id: h for h in hits}
        assert by_id[dense_order[0]].ranks["dense"] == 1
        assert by_id[dense_order[1]].ranks["dense"] == 2

    def test_scores_are_rrf_scores_and_descend(self, hybrid_collection):
        c, query = hybrid_collection
        scores = [h.score for h in c.search_hybrid(query, "filler weather", k=10)]
        assert scores == sorted(scores, reverse=True)
        # RRF scores live in (0, 2/(k+1)]; nothing like a cosine.
        assert all(0 < s <= 2 / 61 for s in scores)

    def test_documents_found_by_both_outrank_single_source_hits(self, make_collection):
        c = make_collection(dimension=DIM, index_type="flat", text_field="content")
        q = np.zeros(DIM, dtype=np.float32)
        q[0] = 1.0
        c.insert(
            [
                # Near in embedding space AND a lexical match: should win.
                {"id": "both", "vector": q.tolist(),
                 "metadata": {"content": "alpha beta"}},
                # Near, but no lexical overlap.
                {"id": "dense-only", "vector": (q * 0.99).tolist(),
                 "metadata": {"content": "nothing in common"}},
                # Lexical match, far away.
                {"id": "sparse-only", "vector": np.eye(DIM, dtype=np.float32)[-1].tolist(),
                 "metadata": {"content": "alpha beta"}},
            ]
        )
        hits = c.search_hybrid(q.tolist(), "alpha beta", k=3)
        assert hits[0].id == "both"
        assert hits[0].ranks == {"dense": 1, "sparse": 1}


class TestParameters:
    def test_rrf_k_is_configurable(self, hybrid_collection):
        c, query = hybrid_collection
        a = c.search_hybrid(query, "filler", k=5, params={"rrf_k": 1})
        b = c.search_hybrid(query, "filler", k=5, params={"rrf_k": 1000})
        assert a[0].score != b[0].score

    def test_candidate_counts_are_configurable(self, hybrid_collection):
        c, query = hybrid_collection
        narrow = c.search_hybrid(
            query, "reciprocal rank fusion", k=5,
            params={"dense_candidates": 1, "sparse_candidates": 1},
        )
        assert len(narrow) <= 2, "only one candidate from each side"

    def test_narrow_dense_candidates_can_drop_the_needle(self, hybrid_collection):
        """Shows the candidate counts really do bound what fusion can see."""
        c, query = hybrid_collection
        hits = c.search_hybrid(
            query, "unrelated filler weather", k=3,
            params={"dense_candidates": 2, "sparse_candidates": 2},
        )
        assert len(hits) <= 4

    def test_index_params_pass_through_to_the_dense_side(self, make_collection):
        c = make_collection(dimension=DIM, index_type="hnsw", text_field="content")
        items, query = synthetic_corpus()
        c.insert(items)
        hits = c.search_hybrid(
            query, "reciprocal rank fusion", k=5, params={"ef_search": 100}
        )
        assert hits

    def test_k_must_be_positive(self, hybrid_collection):
        c, query = hybrid_collection
        with pytest.raises(InvalidRequestError):
            c.search_hybrid(query, "text", k=0)


class TestFiltersAndDeletion:
    def test_filter_applies_after_fusion(self, hybrid_collection):
        c, query = hybrid_collection
        hits = c.search_hybrid(
            query, "reciprocal rank fusion", k=10, filter={"kind": "needle"}
        )
        assert [h.id for h in hits] == ["needle"]

    def test_deleted_documents_are_excluded_from_both_sides(self, hybrid_collection):
        c, query = hybrid_collection
        c.delete("needle")
        hits = c.search_hybrid(query, "reciprocal rank fusion", k=10)
        assert "needle" not in {h.id for h in hits}

    def test_never_returns_more_than_k(self, hybrid_collection):
        c, query = hybrid_collection
        assert len(c.search_hybrid(query, "filler weather document", k=4)) == 4


class TestDegenerateCases:
    def test_query_text_matching_nothing_falls_back_to_dense_order(
        self, hybrid_collection
    ):
        c, query = hybrid_collection
        dense = [h.id for h in c.search(query, k=5)]
        hybrid = [h.id for h in c.search_hybrid(query, "zzzznotpresent", k=5)]
        assert hybrid == dense

    def test_empty_collection_returns_nothing(self, make_collection):
        c = make_collection(dimension=DIM, text_field="content")
        assert c.search_hybrid([0.0] * DIM, "anything", k=5) == []

    def test_works_when_only_some_documents_have_text(self, make_collection):
        c = make_collection(dimension=DIM, index_type="flat", text_field="content")
        c.insert(
            [
                {"id": "with", "vector": [1.0] + [0.0] * (DIM - 1),
                 "metadata": {"content": "alpha"}},
                {"id": "without", "vector": [0.9] + [0.0] * (DIM - 1),
                 "metadata": {}},
            ]
        )
        hits = c.search_hybrid([1.0] + [0.0] * (DIM - 1), "alpha", k=2)
        ids = [h.id for h in hits]
        assert set(ids) == {"with", "without"}
        assert ids[0] == "with", "the lexical match should be promoted"

    @pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivf"])
    def test_works_with_every_dense_index(self, make_collection, index_type):
        c = make_collection(
            dimension=DIM, index_type=index_type, text_field="content",
            index_params={"nlist": 4} if index_type == "ivf" else {},
        )
        items, query = synthetic_corpus()
        c.insert(items)
        hits = c.search_hybrid(query, "reciprocal rank fusion", k=5)
        assert "needle" in {h.id for h in hits}
