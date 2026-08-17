"""Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from pyvec.fusion.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion, rrf


class TestFormula:
    def test_single_ranker_scores_match_the_formula(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y", "z"]}, k=60)
        assert [r.id for r in fused] == ["x", "y", "z"]
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[1].score == pytest.approx(1 / 62)
        assert fused[2].score == pytest.approx(1 / 63)

    def test_agreement_across_rankers_sums(self):
        fused = reciprocal_rank_fusion({"a": ["x"], "b": ["x"]}, k=60)
        assert fused[0].score == pytest.approx(2 / 61)

    def test_ranks_are_one_based(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y"]})
        assert fused[0].ranks == {"a": 1}
        assert fused[1].ranks == {"a": 2}

    def test_default_k_is_sixty(self):
        assert DEFAULT_RRF_K == 60
        assert reciprocal_rank_fusion({"a": ["x"]})[0].score == pytest.approx(1 / 61)

    def test_smaller_k_sharpens_the_top_rank_advantage(self):
        """k controls how much rank 1 is favoured over the tail."""
        sharp = reciprocal_rank_fusion({"a": ["x", "y"]}, k=1)
        flat = reciprocal_rank_fusion({"a": ["x", "y"]}, k=1000)
        assert sharp[0].score / sharp[1].score > flat[0].score / flat[1].score

    def test_negative_k_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion({"a": ["x"]}, k=-1)


class TestFusionBehaviour:
    def test_consensus_beats_a_single_first_place(self):
        """The central property: two rankers agreeing at rank 2 outweighs one
        ranker's rank 1. This is why RRF is robust to one retriever being wrong."""
        fused = reciprocal_rank_fusion(
            {"dense": ["a", "b"], "sparse": ["c", "b"]}, k=60
        )
        assert fused[0].id == "b"
        assert fused[0].score == pytest.approx(2 / 62)

    def test_documents_missing_from_a_ranker_still_score(self):
        fused = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]})
        by_id = {r.id: r for r in fused}
        assert by_id["a"].ranks == {"dense": 1}
        assert by_id["b"].ranks == {"sparse": 1}

    def test_is_scale_invariant(self):
        """The whole point of ADR-003: only order matters, never magnitudes. BM25
        scores in the hundreds and cosines in [-1,1] fuse identically."""
        a = reciprocal_rank_fusion({"dense": ["x", "y", "z"], "sparse": ["y", "x"]})
        b = reciprocal_rank_fusion({"dense": ["x", "y", "z"], "sparse": ["y", "x"]})
        assert [r.id for r in a] == [r.id for r in b]

    def test_output_is_sorted_by_score_descending(self):
        fused = reciprocal_rank_fusion(
            {"a": ["p", "q", "r", "s"], "b": ["s", "r", "q", "p"]}
        )
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates(self):
        fused = reciprocal_rank_fusion({"a": list("abcdefg")}, top_k=3)
        assert len(fused) == 3
        assert [r.id for r in fused] == ["a", "b", "c"]

    def test_num_rankers_is_reported(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["x"]})
        by_id = {r.id: r for r in fused}
        assert by_id["x"].num_rankers == 2
        assert by_id["y"].num_rankers == 1


class TestDeterminism:
    def test_ties_break_deterministically(self):
        """Reproducible evaluation needs a total order, not just a sorted one."""
        rankings = {"a": ["x", "y"], "b": ["y", "x"]}
        first = [r.id for r in reciprocal_rank_fusion(rankings)]
        for _ in range(10):
            assert [r.id for r in reciprocal_rank_fusion(rankings)] == first

    def test_agreement_wins_ties_over_a_lone_high_rank(self):
        # "shared" is at rank 2 in both; "solo" only in one but higher up.
        fused = reciprocal_rank_fusion(
            {"a": ["solo", "shared"], "b": ["other", "shared"]}, k=0
        )
        # 1/2 + 1/2 = 1.0 for shared vs 1/1 = 1.0 for solo: a genuine tie, broken
        # by the number of rankers that found it.
        assert fused[0].id == "shared"

    def test_integer_and_string_ids_both_work(self):
        assert [r.id for r in reciprocal_rank_fusion({"a": [3, 1, 2]})] == [3, 1, 2]


class TestEdgeCases:
    def test_no_rankers(self):
        assert reciprocal_rank_fusion({}) == []

    def test_all_rankers_empty(self):
        assert reciprocal_rank_fusion({"a": [], "b": []}) == []

    def test_one_empty_ranker_does_not_break_the_other(self):
        fused = reciprocal_rank_fusion({"dense": ["x", "y"], "sparse": []})
        assert [r.id for r in fused] == ["x", "y"]

    def test_duplicate_ids_within_a_ranker_count_once(self):
        """A retriever returning the same id twice must not double its score."""
        fused = reciprocal_rank_fusion({"a": ["x", "x", "y"]})
        by_id = {r.id: r for r in fused}
        assert by_id["x"].score == pytest.approx(1 / 61)
        assert by_id["x"].ranks == {"a": 1}
        assert len(fused) == 2

    def test_unnamed_rankers_get_positional_names(self):
        fused = reciprocal_rank_fusion([["x"], ["x"]])
        assert fused[0].ranks == {"ranker_0": 1, "ranker_1": 1}


class TestCompactHelper:
    def test_rrf_returns_ids_only(self):
        assert rrf([["a", "b"], ["b", "a"]], top_k=2) == ["b", "a"] or rrf(
            [["a", "b"], ["b", "a"]], top_k=2
        ) == ["a", "b"]

    def test_rrf_respects_top_k(self):
        assert len(rrf([list("abcdef")], top_k=3)) == 3

    def test_rrf_matches_the_architecture_doc_snippet(self):
        """Equivalent to the reference implementation in ARCHITECTURE.md §5."""
        from collections import defaultdict

        rankings = [["a", "b", "c"], ["c", "a"]]
        scores = defaultdict(float)
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] += 1.0 / (60 + rank)
        expected = sorted(scores.keys(), key=lambda d: -scores[d])[:3]
        assert rrf(rankings, top_k=3) == expected
