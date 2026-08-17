"""BM25 scoring and the inverted index.

The important tests here compute BM25 by hand from the formula in LEARNING.md and
compare. A scorer that is *nearly* right — wrong IDF variant, missing length
normalisation — still returns plausible rankings on small corpora, so ranking
assertions alone would not catch it.
"""

from __future__ import annotations

import math

import numpy as np

import pytest

from pyvec.core.tokenize import tokenize
from pyvec.indexes.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index
from tests.conftest import TEXT_CORPUS


@pytest.fixture
def index():
    idx = BM25Index()
    for i, text in enumerate(TEXT_CORPUS):
        idx.add(i, text)
    return idx


class TestTokenizer:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("The quick, brown FOX -- jumps!") == [
            "the", "quick", "brown", "fox", "jumps",
        ]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize("!!!...") == []

    def test_splits_snake_case_on_underscores(self):
        assert tokenize("snake_case_name") == ["snake", "case", "name"]

    def test_keeps_digits(self):
        assert tokenize("covid 19 and gpt4") == ["covid", "19", "and", "gpt4"]

    def test_handles_unicode(self):
        assert tokenize("café Ünicode naïve") == ["café", "ünicode", "naïve"]

    def test_collapses_repeated_whitespace(self):
        assert tokenize("a\t b\n\nc") == ["a", "b", "c"]


class TestFormula:
    def test_idf_matches_the_documented_formula(self, index):
        n = index.num_docs
        for term in ("the", "fox", "quick", "consectetur"):
            df = sum(1 for _ in index.postings.get(term, []))
            expected = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            assert index.idf(term) == pytest.approx(expected)

    def test_idf_is_never_negative(self):
        """The Lucene ``+1`` inside the log matters: without it, a term in more
        than half the corpus gets negative weight and matching it *penalises* a
        document."""
        idx = BM25Index()
        for i in range(10):
            idx.add(i, "common term everywhere")
        assert idx.idf("common") >= 0.0

    def test_rare_terms_outweigh_common_ones(self, index):
        assert index.idf("consectetur") > index.idf("the")

    def test_unknown_term_has_zero_idf(self, index):
        assert index.idf("zzzznotpresent") == 0.0

    def test_score_matches_hand_computed_bm25(self):
        """Full arithmetic on a three-document corpus, done by hand."""
        idx = BM25Index(k1=DEFAULT_K1, b=DEFAULT_B)
        idx.add(0, "cat dog")  # len 2
        idx.add(1, "cat cat bird")  # len 3
        idx.add(2, "fish")  # len 1
        avgdl = 6 / 3

        n, df = 3, 2  # "cat" appears in docs 0 and 1
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        def bm25(tf, doclen):
            norm = DEFAULT_K1 * (1 - DEFAULT_B + DEFAULT_B * doclen / avgdl)
            return idf * (tf * (DEFAULT_K1 + 1)) / (tf + norm)

        results = dict(idx.search("cat", k=10))
        assert results[0] == pytest.approx(bm25(tf=1, doclen=2))
        assert results[1] == pytest.approx(bm25(tf=2, doclen=3))
        assert 2 not in results

    def test_saturation_means_diminishing_returns(self):
        """The k1 term: going 1 -> 2 occurrences must help more than 9 -> 10."""
        idx = BM25Index()
        idx.add(0, "x " * 1)
        idx.add(1, "x " * 2)
        idx.add(2, "x " * 9)
        idx.add(3, "x " * 10)
        scores = dict(idx.search("x", k=10))
        first_gain = scores[1] - scores[0]
        later_gain = scores[3] - scores[2]
        assert first_gain > later_gain > 0

    def test_length_normalisation_penalises_long_documents(self):
        """The b term: equal term counts, different lengths -> shorter wins."""
        idx = BM25Index()
        idx.add(0, "target word")
        idx.add(1, "target " + "filler " * 50)
        scores = dict(idx.search("target", k=10))
        assert scores[0] > scores[1]

    def test_b_zero_disables_length_normalisation(self):
        idx = BM25Index(b=0.0)
        idx.add(0, "target word")
        idx.add(1, "target " + "filler " * 50)
        scores = dict(idx.search("target", k=10))
        assert scores[0] == pytest.approx(scores[1])

    def test_explain_totals_match_search(self, index):
        breakdown = index.explain("quick fox", 0)
        score = dict(index.search("quick fox", k=10))[0]
        assert breakdown["score"] == pytest.approx(score)
        assert [t["term"] for t in breakdown["terms"]] == ["quick", "fox"]

    def test_repeated_query_terms_count_twice(self, index):
        once = dict(index.search("quick", k=10))[0]
        twice = dict(index.search("quick quick", k=10))[0]
        assert twice == pytest.approx(2 * once)


class TestRanking:
    def test_documents_matching_all_terms_rank_above_partial_matches(self, index):
        ranked = [i for i, _ in index.search("quick brown fox", k=8)]
        # Docs 0 and 1 both contain all of quick/brown/fox and are both 9 tokens
        # long; doc 6 contains only "fox". Full matches must come first.
        assert set(ranked[:2]) == {0, 1}
        assert ranked.index(6) > 1
        # Between the two full matches, doc 1 says "quick" three times to doc 0's
        # once, at equal length — so tf saturation still favours doc 1.
        assert ranked[0] == 1

    def test_term_frequency_breaks_the_tie(self, index):
        """Doc 1 says "quick" twice; doc 0 once. On "quick" alone, 1 wins."""
        ranked = [i for i, _ in index.search("quick", k=8)]
        assert ranked[0] == 1

    def test_returns_only_documents_containing_a_query_term(self, index):
        ranked = [i for i, _ in index.search("consectetur", k=8)]
        assert ranked == [2]

    def test_no_match_returns_empty(self, index):
        assert index.search("zzzznotpresent", k=5) == []

    def test_empty_query_returns_empty(self, index):
        assert index.search("", k=5) == []
        assert index.search("!!!", k=5) == []

    def test_respects_k(self, index):
        assert len(index.search("the", k=2)) == 2

    def test_scores_are_descending(self, index):
        scores = [s for _, s in index.search("the quick dog", k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_ties_break_deterministically(self):
        idx = BM25Index()
        for i in range(5):
            idx.add(i, "identical text here")
        a = idx.search("identical", k=5)
        b = idx.search("identical", k=5)
        assert a == b
        assert [i for i, _ in a] == [0, 1, 2, 3, 4]


class TestVectorisedScoringEquivalence:
    """The array-based scorer must be a pure optimisation.

    `search` was rewritten from a Python accumulator loop to NumPy array ops for a
    ~7x speedup. The reference loop is kept here, transcribed straight from the
    formula, and the two are required to agree *exactly* — not approximately.
    Scores are accumulated in float64 on both paths specifically so this can be an
    equality assertion; a tolerance would let a genuine drift in the arithmetic
    hide behind it.
    """

    @staticmethod
    def reference_search(index: BM25Index, text: str, k: int = 10):
        """BM25 the obvious way: one Python loop over each posting list."""
        from collections import defaultdict

        terms = tokenize(text)
        avgdl = index.avg_doc_len
        if not terms or avgdl <= 0:
            return []
        k1, b = index.k1, index.b
        norms: dict[int, float] = {}
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            plist = index.postings.get(term)
            if not plist:
                continue
            idf = index.idf(term)
            if idf == 0.0:
                continue
            for doc_id, tf in plist:
                if doc_id in index._deleted:
                    continue
                norm = norms.get(doc_id)
                if norm is None:
                    norm = k1 * (1.0 - b + b * (index.doc_lens[doc_id] / avgdl))
                    norms[doc_id] = norm
                scores[doc_id] += idf * (tf * (k1 + 1.0)) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(d, float(s)) for d, s in ranked[:k]]

    def test_matches_the_reference_loop_on_the_fixture(self, index):
        for query in ("quick", "quick fox", "the lazy dog", "dense retrieval",
                      "vectors", "the", "quick quick fox", "nonexistent"):
            assert index.search(query, k=8) == self.reference_search(
                index, query, k=8
            ), f"divergence on {query!r}"

    def test_matches_the_reference_loop_on_a_zipfian_corpus(self):
        """The realistic hard case: head terms whose posting lists span the corpus,
        and many documents tying on identical scores."""
        rng = np.random.default_rng(0)
        vocab = [f"term{i:04d}" for i in range(400)]

        def sample(count):
            idx = np.clip(rng.zipf(1.3, size=count) - 1, 0, len(vocab) - 1)
            return " ".join(vocab[i] for i in idx)

        index = BM25Index()
        for doc_id in range(1500):
            index.add(doc_id, sample(20))

        qrng = np.random.default_rng(1)
        for _ in range(40):
            terms = np.clip(qrng.zipf(1.3, size=3) - 1, 0, len(vocab) - 1)
            query = " ".join(vocab[i] for i in terms)
            assert index.search(query, k=10) == self.reference_search(
                index, query, k=10
            ), f"divergence on {query!r}"

    def test_matches_the_reference_loop_with_tombstones(self, index):
        index.remove([0, 3])
        for query in ("quick fox", "the", "dog"):
            assert index.search(query, k=8) == self.reference_search(index, query, k=8)

    def test_the_numpy_view_is_invalidated_on_mutation(self, index):
        """A stale array cache would silently serve pre-mutation results."""
        before = index.search("quick", k=5)
        index.add(999, "quick quick quick quick brand new document")
        after = index.search("quick", k=5)
        assert after != before
        assert after[0][0] == 999, "the new document should dominate on tf"
        assert after == self.reference_search(index, "quick", k=5)

    def test_the_numpy_view_is_invalidated_on_purge(self, index):
        top = index.search("quick", k=1)[0][0]
        index.remove([top], purge=True)
        assert index.search("quick", k=5) == self.reference_search(index, "quick", k=5)
        assert top not in {d for d, _ in index.search("quick", k=5)}


class TestBookkeeping:
    def test_counts_and_lengths(self, index):
        assert index.num_docs == len(TEXT_CORPUS)
        assert index.doc_lens[0] == len(tokenize(TEXT_CORPUS[0]))
        assert index.avg_doc_len == pytest.approx(
            sum(len(tokenize(t)) for t in TEXT_CORPUS) / len(TEXT_CORPUS)
        )

    def test_vocabulary_contains_every_distinct_token(self, index):
        expected = {t for text in TEXT_CORPUS for t in tokenize(text)}
        assert index.vocabulary_size == len(expected)

    def test_empty_document_is_indexed_but_matches_nothing(self):
        idx = BM25Index()
        idx.add(0, "")
        idx.add(1, "real content")
        assert idx.num_docs == 2
        assert idx.doc_lens[0] == 0
        assert [i for i, _ in idx.search("real", k=5)] == [1]

    def test_readding_an_id_replaces_its_postings(self):
        """Otherwise the old and new text both contribute and tf double-counts."""
        idx = BM25Index()
        idx.add(0, "original text")
        idx.add(0, "replacement words")
        assert idx.num_docs == 1
        assert idx.search("original", k=5) == []
        assert [i for i, _ in idx.search("replacement", k=5)] == [0]
        assert idx.doc_lens[0] == 2

    def test_add_batch_matches_repeated_add(self):
        a = BM25Index()
        a.add_batch(enumerate(TEXT_CORPUS))
        b = BM25Index()
        for i, t in enumerate(TEXT_CORPUS):
            b.add(i, t)
        assert a.postings == b.postings
        assert a.search("quick fox", k=5) == b.search("quick fox", k=5)

    def test_stats_shape(self, index):
        stats = index.stats()
        assert stats["type"] == "bm25"
        assert stats["num_docs"] == len(TEXT_CORPUS)
        assert stats["k1"] == DEFAULT_K1 and stats["b"] == DEFAULT_B


class TestDeletion:
    def test_tombstoned_docs_do_not_appear(self, index):
        index.remove([0])
        assert 0 not in {i for i, _ in index.search("quick fox", k=8)}

    def test_deletion_changes_idf(self, index):
        """N and df both shift, so cached IDF has to be invalidated."""
        before = index.idf("fox")
        index.remove([0, 6])
        assert index.idf("fox") != before

    def test_num_docs_excludes_tombstones(self, index):
        index.remove([0, 1])
        assert index.num_docs == len(TEXT_CORPUS) - 2

    def test_avg_doc_len_excludes_tombstones(self):
        idx = BM25Index()
        idx.add(0, "a b c d")
        idx.add(1, "e f")
        assert idx.avg_doc_len == pytest.approx(3.0)
        idx.remove([0])
        assert idx.avg_doc_len == pytest.approx(2.0)

    def test_purge_reclaims_posting_space(self, index):
        before = sum(len(p) for p in index.postings.values())
        index.remove([0], purge=True)
        assert sum(len(p) for p in index.postings.values()) < before
        assert 0 not in index.doc_lens

    def test_optimize_purges_all_tombstones(self, index):
        index.remove([0, 1])
        report = index.optimize()
        assert report["purged_docs"] == 2
        assert index.num_docs == len(TEXT_CORPUS) - 2
        assert all(
            0 not in {d for d, _ in p} for p in index.postings.values()
        )

    def test_purging_every_doc_leaves_an_empty_index(self):
        idx = BM25Index()
        idx.add(0, "only document")
        idx.remove([0], purge=True)
        assert idx.num_docs == 0
        assert idx.postings == {}
        assert idx.search("only", k=5) == []

    def test_exclude_argument_is_honoured(self, index):
        got = {i for i, _ in index.search("the", k=8, exclude={0, 3})}
        assert got.isdisjoint({0, 3})


class TestPersistence:
    def test_round_trip(self, tmp_path, index):
        before = index.search("quick brown fox", k=5)
        index.save(tmp_path / "bm25.json")

        restored = BM25Index()
        restored.load(tmp_path / "bm25.json")

        assert restored.postings == index.postings
        assert restored.doc_lens == index.doc_lens
        assert restored.num_docs == index.num_docs
        assert restored.avg_doc_len == pytest.approx(index.avg_doc_len)
        assert restored.search("quick brown fox", k=5) == before

    def test_tombstones_and_params_survive(self, tmp_path):
        idx = BM25Index(k1=1.2, b=0.5)
        for i, t in enumerate(TEXT_CORPUS):
            idx.add(i, t)
        idx.remove([1])
        idx.save(tmp_path / "bm25.json")

        restored = BM25Index()
        restored.load(tmp_path / "bm25.json")
        assert restored.k1 == 1.2 and restored.b == 0.5
        assert restored.num_docs == len(TEXT_CORPUS) - 1
        assert 1 not in {i for i, _ in restored.search("quick", k=8)}

    def test_loading_a_missing_file_is_a_no_op(self, tmp_path):
        idx = BM25Index()
        idx.load(tmp_path / "absent.json")
        assert idx.num_docs == 0
