"""IVF-Flat and its k-means quantiser."""

from __future__ import annotations

import numpy as np
import pytest

from pyvec.core.kmeans import assign, kmeans
from pyvec.core.types import ArrayVectorSource, Metric
from pyvec.indexes.flat import FlatIndex
from pyvec.indexes.ivf import IVFFlatIndex
from tests.conftest import recall_at_k


@pytest.fixture
def clustered(rng):
    """Well-separated blobs.

    IVF is a partitioning method, so it is only meaningfully exercised on data
    that *has* partitions. LEARNING.md layer 2: "IVF loves clustered data. It
    hurts on uniformly distributed data." Testing recall only on Gaussian noise
    would understate the index and hide regressions in centroid quality.
    """
    centres = rng.normal(size=(8, 32)).astype(np.float32) * 6.0
    blobs = [c + rng.normal(size=(150, 32)).astype(np.float32) for c in centres]
    return np.ascontiguousarray(np.vstack(blobs), dtype=np.float32)


class TestKMeans:
    def test_recovers_known_clusters(self, rng):
        centres = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]], np.float32)
        x = np.vstack(
            [c + rng.normal(scale=0.3, size=(100, 2)).astype(np.float32) for c in centres]
        )
        result = kmeans(x, 3, metric=Metric.L2, seed=1)
        # Each true centre should have a learned centroid essentially on top of it.
        for centre in centres:
            best = min(float(np.linalg.norm(centre - c)) for c in result.centroids)
            assert best < 0.5, f"no centroid near {centre}"

    def test_produces_the_requested_number_of_centroids(self, clustered):
        assert kmeans(clustered, 16, seed=1).centroids.shape == (16, 32)

    def test_no_empty_clusters(self, clustered):
        """Empty clusters would silently reduce nlist and waste probes."""
        result = kmeans(clustered, 32, metric=Metric.L2, seed=1)
        counts = np.bincount(result.assignments, minlength=32)
        assert np.all(counts > 0), f"{int(np.sum(counts == 0))} empty clusters"

    def test_inertia_decreases_with_more_centroids(self, clustered):
        coarse = kmeans(clustered, 4, seed=1).inertia
        fine = kmeans(clustered, 32, seed=1).inertia
        assert fine < coarse

    def test_is_seed_deterministic(self, clustered):
        a = kmeans(clustered, 8, seed=7)
        b = kmeans(clustered, 8, seed=7)
        np.testing.assert_array_equal(a.centroids, b.centroids)
        np.testing.assert_array_equal(a.assignments, b.assignments)

    def test_different_seeds_give_different_centroids(self, clustered):
        a = kmeans(clustered, 8, seed=1)
        b = kmeans(clustered, 8, seed=2)
        assert not np.allclose(a.centroids, b.centroids)

    def test_k_is_clamped_to_the_dataset_size(self):
        x = np.arange(12, dtype=np.float32).reshape(4, 3)
        assert kmeans(x, 100, seed=1).centroids.shape[0] == 4

    def test_cosine_centroids_are_unit_norm(self, clustered):
        """Spherical k-means: a mean of unit vectors is not a unit vector, and
        the cosine ordering distance assumes unit norms on both sides."""
        x = clustered / np.linalg.norm(clustered, axis=1, keepdims=True)
        result = kmeans(x, 8, metric=Metric.COSINE, seed=1)
        np.testing.assert_allclose(
            np.linalg.norm(result.centroids, axis=1), 1.0, rtol=1e-5
        )

    def test_sampling_caps_the_training_set(self, rng):
        x = rng.normal(size=(5000, 8)).astype(np.float32)
        result = kmeans(x, 8, sample=500, seed=1)
        assert result.assignments.shape[0] == 500
        assert result.centroids.shape == (8, 8)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            kmeans(np.zeros((0, 4), dtype=np.float32), 4)

    def test_assign_is_blocking_invariant(self, clustered):
        centroids = kmeans(clustered, 8, seed=1).centroids
        a, _ = assign(clustered, centroids, Metric.L2, block=10_000)
        b, _ = assign(clustered, centroids, Metric.L2, block=7)
        np.testing.assert_array_equal(a, b)


class TestRecall:
    def test_meets_the_90_percent_bar_on_clustered_data(self, clustered):
        """PRD NF1 for IVF: >=90% recall@10 with nprobe=16 on 256 centroids.

        Scaled to the fixture size (1200 vectors), so nlist=32 with nprobe=8
        keeps the same 1:4 probe ratio the PRD target uses.
        """
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=32, nprobe=8, seed=1)
        index.add(range(len(clustered)))
        index.train()
        flat = FlatIndex(32, Metric.L2, src)
        flat.add(range(len(clustered)))

        scores = [
            recall_at_k(flat.search(q, 10), index.search(q, 10), 10)
            for q in clustered[::40]
        ]
        mean = float(np.mean(scores))
        assert mean >= 0.90, f"recall@10 = {mean:.4f}"

    def test_recall_increases_with_nprobe(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=32, seed=1)
        index.add(range(len(clustered)))
        index.train()
        flat = FlatIndex(32, Metric.L2, src)
        flat.add(range(len(clustered)))

        means = []
        for nprobe in (1, 2, 8, 32):
            means.append(
                float(
                    np.mean(
                        [
                            recall_at_k(
                                flat.search(q, 10), index.search(q, 10, nprobe=nprobe), 10
                            )
                            for q in clustered[::60]
                        ]
                    )
                )
            )
        assert means == sorted(means), f"recall not monotonic in nprobe: {means}"

    def test_nprobe_equal_to_nlist_is_exact(self, clustered):
        """Probing every bucket degenerates to a full scan, so it must be exact."""
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=16, seed=1)
        index.add(range(len(clustered)))
        index.train()
        flat = FlatIndex(32, Metric.L2, src)
        flat.add(range(len(clustered)))
        for q in clustered[::200]:
            got = [i for i, _ in index.search(q, 10, nprobe=16)]
            assert got == [i for i, _ in flat.search(q, 10)]

    @pytest.mark.parametrize("metric", list(Metric))
    def test_every_metric_works(self, metric, clustered):
        x = clustered
        if metric is Metric.COSINE:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        src = ArrayVectorSource(x)
        index = IVFFlatIndex(32, metric, src, nlist=16, nprobe=16, seed=1)
        index.add(range(len(x)))
        index.train()
        flat = FlatIndex(32, metric, src)
        flat.add(range(len(x)))
        for q in x[::300]:
            assert [i for i, _ in index.search(q, 5)] == [
                i for i, _ in flat.search(q, 5)
            ]


class TestPostingLists:
    def test_every_vector_lands_in_exactly_one_bucket(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=16, seed=1)
        index.add(range(len(clustered)))
        index.train()
        assigned = [i for bucket in index.postings.values() for i in bucket]
        assert sorted(assigned) == list(range(len(clustered)))

    def test_buckets_are_roughly_balanced_on_clustered_data(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=8, seed=1)
        index.add(range(len(clustered)))
        index.train()
        sizes = sorted(len(v) for v in index.postings.values())
        assert sizes[0] > 0
        # A single bucket owning most of the data means k-means++ init failed.
        assert sizes[-1] < len(clustered) * 0.5, sizes


class TestUntrainedBehaviour:
    def test_search_before_training_is_exact(self, vectors, source):
        """Below the training threshold IVF degenerates to a full scan. Slow, but
        it must never silently return nothing."""
        index = IVFFlatIndex(32, Metric.L2, source, nlist=256)
        index.add(range(50))
        assert not index.is_trained
        flat = FlatIndex(32, Metric.L2, source)
        flat.add(range(50))
        got = [i for i, _ in index.search(vectors[0], 5)]
        assert got == [i for i, _ in flat.search(vectors[0], 5)]

    def test_trains_automatically_once_there_is_enough_data(self, vectors, source):
        index = IVFFlatIndex(32, Metric.L2, source, nlist=4, seed=1)
        index.add(range(len(vectors)))
        assert index.is_trained
        assert len(index.postings) > 0

    def test_pending_vectors_are_still_searchable(self, vectors, source):
        """Vectors added between trainings live in no bucket; excluding them from
        the scan would make them invisible."""
        index = IVFFlatIndex(32, Metric.L2, source, nlist=4, seed=1)
        index.add(range(300))
        index.train()
        index._pending = [999]  # simulate an unassigned arrival
        got = {i for i, _ in index.search(vectors[999], 5)}
        assert 999 in got


class TestIncrementalInserts:
    def test_online_assignment_keeps_new_vectors_findable(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(
            32, Metric.L2, src, nlist=16, nprobe=16, seed=1,
            retrain_growth_factor=None,
        )
        index.add(range(600))
        index.train()
        index.add(range(600, len(clustered)))
        for i in (700, 900, 1100):
            got = {j for j, _ in index.search(clustered[i], 5)}
            assert i in got

    def test_growth_triggers_an_automatic_retrain(self, clustered):
        """Guards against serving centroids learned from the first few hundred
        vectors of a collection that later grew by orders of magnitude."""
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(
            32, Metric.L2, src, nlist=8, seed=1, retrain_growth_factor=2.0
        )
        index.add(range(300))
        index.train()
        first = index.centroids.copy()
        index.add(range(300, len(clustered)))
        assert index._trained_size > 300
        assert not np.allclose(first, index.centroids), "should have retrained"

    def test_retrain_can_be_disabled(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(
            32, Metric.L2, src, nlist=8, seed=1, retrain_growth_factor=None
        )
        index.add(range(200))
        index.train()
        first = index.centroids.copy()
        index.add(range(200, len(clustered)))
        np.testing.assert_array_equal(first, index.centroids)


class TestDeletion:
    def test_deleted_ids_are_excluded(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=16, nprobe=16, seed=1)
        index.add(range(len(clustered)))
        index.train()
        index.remove([0, 1, 2])
        got = {i for i, _ in index.search(clustered[0], 10)}
        assert got.isdisjoint({0, 1, 2})

    def test_optimize_purges_tombstones_and_retrains(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=16, seed=1)
        index.add(range(len(clustered)))
        index.train()
        index.remove(range(0, 100))
        before = len(index)
        report = index.optimize()
        assert report["num_vectors"] == before
        assert len(index) == before
        assigned = {i for bucket in index.postings.values() for i in bucket}
        assert assigned.isdisjoint(range(100))


class TestEdgeCases:
    def test_empty_index_returns_nothing(self, source):
        assert IVFFlatIndex(32, Metric.L2, source).search(np.zeros(32), 5) == []

    def test_k_zero_returns_nothing(self, source):
        index = IVFFlatIndex(32, Metric.L2, source, nlist=4, seed=1)
        index.add(range(100))
        assert index.search(np.zeros(32), 0) == []

    def test_nprobe_is_clamped_to_nlist(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=8, seed=1)
        index.add(range(len(clustered)))
        index.train()
        assert len(index.search(clustered[0], 5, nprobe=10_000)) == 5

    def test_results_are_sorted(self, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=8, nprobe=4, seed=1)
        index.add(range(len(clustered)))
        index.train()
        d = [dist for _, dist in index.search(clustered[0], 20)]
        assert d == sorted(d)

    def test_params_are_reported_for_the_api(self, source):
        index = IVFFlatIndex(32, Metric.L2, source, nlist=64, nprobe=4)
        assert index.params == {"nlist": 64, "nprobe": 4}


class TestPersistence:
    def test_round_trip(self, tmp_path, clustered):
        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=16, nprobe=4, seed=1)
        index.add(range(len(clustered)))
        index.train()
        index.remove([5, 6])
        before = index.search(clustered[0], 10)

        index.save(tmp_path / "ivf.idx")
        restored = IVFFlatIndex(32, Metric.L2, src)
        restored.load(tmp_path / "ivf.idx")

        assert restored.nlist == 16
        assert restored.nprobe == 4
        assert restored.is_trained
        np.testing.assert_array_equal(restored.centroids, index.centroids)
        assert restored.postings == index.postings
        assert len(restored) == len(index)
        assert restored.search(clustered[0], 10) == before

    def test_untrained_round_trip(self, tmp_path, vectors, source):
        index = IVFFlatIndex(32, Metric.L2, source, nlist=256)
        index.add(range(20))
        index.save(tmp_path / "ivf.idx")
        restored = IVFFlatIndex(32, Metric.L2, source)
        restored.load(tmp_path / "ivf.idx")
        assert not restored.is_trained
        assert len(restored) == 20
        assert len(restored.search(vectors[0], 5)) == 5

    def test_loading_a_missing_file_is_a_no_op(self, tmp_path, source):
        index = IVFFlatIndex(32, Metric.L2, source)
        index.load(tmp_path / "absent.idx")
        assert not index.is_trained

    def test_missing_centroid_file_is_reported(self, tmp_path, clustered):
        from pyvec.core.errors import CorruptDataError

        src = ArrayVectorSource(clustered)
        index = IVFFlatIndex(32, Metric.L2, src, nlist=8, seed=1)
        index.add(range(len(clustered)))
        index.train()
        path = tmp_path / "ivf.idx"
        index.save(path)
        path.with_suffix(".centroids.npy").unlink()
        with pytest.raises(CorruptDataError, match="trained but"):
            IVFFlatIndex(32, Metric.L2, src).load(path)
