"""FlatIndex — the correctness oracle.

Everything else in the project is measured against this index, so its own tests
compare it against hand-computed answers and a completely independent NumPy
brute force. PRD NF5.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyvec.core.types import ArrayVectorSource, Metric
from pyvec.indexes.flat import FlatIndex


def brute_force(q, x, metric, k):
    """Independent oracle written the slow, obvious way."""
    if metric is Metric.L2:
        d = [float(np.sum((q - row) ** 2)) for row in x]
    elif metric is Metric.DOT:
        d = [float(-np.dot(q, row)) for row in x]
    else:
        d = [
            float(
                1.0
                - np.dot(q, row)
                / (np.linalg.norm(q) * np.linalg.norm(row) + 1e-12)
            )
            for row in x
        ]
    return sorted(range(len(x)), key=lambda i: (d[i], i))[:k]


class TestExactness:
    @pytest.mark.parametrize("metric", list(Metric))
    def test_matches_independent_brute_force(self, metric, vectors, queries):
        x = vectors
        if metric is Metric.COSINE:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        index = FlatIndex(x.shape[1], metric, ArrayVectorSource(x))
        index.add(range(len(x)))
        for q in queries[:10]:
            if metric is Metric.COSINE:
                q = q / np.linalg.norm(q)
            got = [i for i, _ in index.search(q, 10)]
            expected = brute_force(q, x, metric, 10)
            assert got == expected

    def test_finds_itself_first(self, vectors, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(len(vectors)))
        for i in (0, 17, 500, 999):
            top, dist = index.search(vectors[i], 1)[0]
            assert top == i
            assert dist == pytest.approx(0.0, abs=1e-4)

    def test_hand_computed_example(self):
        x = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32
        )
        index = FlatIndex(2, Metric.L2, ArrayVectorSource(x))
        index.add([0, 1, 2, 3])
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), 4)
        assert [i for i, _ in results] == [0, 2, 1, 3]
        # squared L2 from (1,0): 0, 0.04+0.36=0.4, 2, 4
        assert [pytest.approx(d, abs=1e-5) for _, d in results] == [0.0, 0.4, 2.0, 4.0]

    def test_distances_are_sorted_ascending(self, vectors, source, queries):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(len(vectors)))
        d = [dist for _, dist in index.search(queries[0], 20)]
        assert d == sorted(d)


class TestBlocking:
    def test_block_size_does_not_change_results(self, vectors, source, queries):
        """Chunked scanning must be invisible from the outside."""
        whole = FlatIndex(32, Metric.L2, source, block=10_000)
        chunked = FlatIndex(32, Metric.L2, source, block=64)
        for index in (whole, chunked):
            index.add(range(len(vectors)))
        for q in queries[:5]:
            a = whole.search(q, 15)
            b = chunked.search(q, 15)
            assert [i for i, _ in a] == [i for i, _ in b]


class TestEdgeCases:
    def test_empty_index_returns_nothing(self, source):
        assert FlatIndex(32, Metric.L2, source).search(np.zeros(32), 5) == []

    def test_k_larger_than_collection_returns_everything(self, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(5))
        assert len(index.search(np.zeros(32), 100)) == 5

    def test_k_zero_returns_nothing(self, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(10))
        assert index.search(np.zeros(32), 0) == []

    def test_single_vector(self, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add([3])
        assert [i for i, _ in index.search(np.zeros(32), 5)] == [3]

    def test_ignores_unknown_search_params(self, source):
        """The API forwards one params dict to whichever index is configured."""
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(10))
        assert len(index.search(np.zeros(32), 3, ef_search=64, nprobe=8)) == 3


class TestDeletion:
    def test_deleted_ids_are_excluded(self, vectors, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(100))
        index.remove([5, 6, 7])
        got = {i for i, _ in index.search(vectors[5], 10)}
        assert got.isdisjoint({5, 6, 7})

    def test_len_accounts_for_deletions(self, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(50))
        index.remove([1, 2])
        assert len(index) == 48

    def test_exclude_argument_is_honoured(self, vectors, source):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(100))
        got = {i for i, _ in index.search(vectors[0], 5, exclude={0, 1})}
        assert got.isdisjoint({0, 1})


class TestPersistence:
    def test_round_trip(self, tmp_path, vectors, source, queries):
        index = FlatIndex(32, Metric.L2, source)
        index.add(range(200))
        index.remove([3, 4])
        before = index.search(queries[0], 10)

        index.save(tmp_path / "flat.idx")
        restored = FlatIndex(32, Metric.L2, source)
        restored.load(tmp_path / "flat.idx")

        assert len(restored) == len(index)
        assert restored.search(queries[0], 10) == before

    def test_loading_a_missing_file_is_a_no_op(self, tmp_path, source):
        index = FlatIndex(32, Metric.L2, source)
        index.load(tmp_path / "nope.idx")
        assert len(index) == 0
