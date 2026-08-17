"""HNSW correctness.

The tests are organised around the three traps from LEARNING.md, because those
are the failure modes that produce an index which *looks* fine:

* recall stuck near 50%  -> naive neighbour selection instead of the heuristic
* recall stuck near 80%  -> broken level distribution
* crash on first insert  -> entry point handling

Plus the structural invariants (degree bounds, edge symmetry, reachability) that
a recall number alone would not catch: an index can hit 95% recall while quietly
leaking unreachable nodes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyvec.core.types import ArrayVectorSource, Metric
from pyvec.indexes.flat import FlatIndex
from pyvec.indexes.hnsw import HNSWIndex
from tests.conftest import recall_at_k

# Smaller ef_construction than the 200 default keeps the suite quick; recall
# targets below are set against this cheaper build, so they are conservative.
BUILD = {"M": 16, "ef_construction": 100, "seed": 42}


@pytest.fixture
def built(vectors, source):
    index = HNSWIndex(32, Metric.L2, source, **BUILD)
    index.add(range(len(vectors)))
    return index


@pytest.fixture
def oracle(vectors, source):
    index = FlatIndex(32, Metric.L2, source)
    index.add(range(len(vectors)))
    return index


class TestRecall:
    """The headline quality claim: PRD NF1 and PROJECT_PLAN week 2."""

    def test_recall_at_10_meets_the_90_percent_bar(self, built, oracle, queries):
        scores = [
            recall_at_k(oracle.search(q, 10), built.search(q, 10, ef_search=64), 10)
            for q in queries
        ]
        mean = float(np.mean(scores))
        # PROJECT_PLAN week 2 test: ">=90% vs FlatIndex on random vectors".
        assert mean >= 0.90, f"recall@10 = {mean:.4f}; check neighbour selection"

    def test_recall_increases_with_ef_search(self, built, oracle, queries):
        """The knob users trade latency for recall with must actually work."""
        means = []
        for ef in (8, 16, 64, 256):
            means.append(
                float(
                    np.mean(
                        [
                            recall_at_k(
                                oracle.search(q, 10),
                                built.search(q, 10, ef_search=ef),
                                10,
                            )
                            for q in queries[:20]
                        ]
                    )
                )
            )
        assert means == sorted(means), f"recall not monotonic in ef_search: {means}"
        assert means[-1] > means[0]

    def test_large_ef_reaches_near_perfect_recall(self, built, oracle, queries):
        """A wide enough beam should find essentially everything."""
        scores = [
            recall_at_k(oracle.search(q, 10), built.search(q, 10, ef_search=400), 10)
            for q in queries[:20]
        ]
        assert float(np.mean(scores)) >= 0.99

    def test_finds_exact_match_for_an_indexed_vector(self, built, vectors):
        """A stored vector must be its own nearest neighbour."""
        hits = 0
        for i in range(0, 1000, 97):
            top = built.search(vectors[i], 1, ef_search=64)
            if top and top[0][0] == i:
                hits += 1
        assert hits >= 10  # every probe should succeed; allow one pathological miss

    @pytest.mark.parametrize("metric", list(Metric))
    def test_recall_holds_for_every_metric(self, metric, vectors, queries):
        x = vectors
        q_all = queries
        if metric is Metric.COSINE:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            q_all = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        src = ArrayVectorSource(x)
        index = HNSWIndex(32, metric, src, **BUILD)
        index.add(range(len(x)))
        flat = FlatIndex(32, metric, src)
        flat.add(range(len(x)))
        scores = [
            recall_at_k(flat.search(q, 10), index.search(q, 10, ef_search=128), 10)
            for q in q_all[:20]
        ]
        assert float(np.mean(scores)) >= 0.90


class TestLevelDistribution:
    """Trap #2. A wrong distribution caps recall around 80%."""

    def test_histogram_decays_geometrically_by_M(self, built):
        hist = built.level_histogram()
        assert hist[0] == 1000, "layer 0 must contain every vector"
        assert len(hist) >= 2
        for lower, upper in zip(hist, hist[1:]):
            assert upper < lower, f"layers not thinning: {hist}"
        # Each layer keeps ~1/M of the one below. Allow a wide band: this is a
        # random process and 1000 samples is a small tail.
        ratio = hist[1] / hist[0]
        assert 1 / (built.M * 4) < ratio < 4 / built.M, f"ratio {ratio:.4f}"

    def test_mean_level_matches_theory(self, built):
        """``E[level] = 1/(M-1)``, and ``P(level >= 1) = 1/M``.

        Worth stating carefully, because LEARNING.md's "mean level ~ 1/ln(M)"
        describes the *continuous* draw ``-ln(U) * mL`` before flooring. The
        stored level is the floor of that, and for ``X ~ Exp(ln M)``::

            E[floor(X)] = sum_{k>=1} P(X >= k) = sum_{k>=1} M^-k = 1/(M - 1)

        At M=16 that is 0.067, not 0.361 — a 5x difference. Asserting the
        continuous mean would fail against a perfectly correct implementation,
        which is exactly the kind of thing that sends you hunting for a bug in
        working code.
        """
        levels = np.array(list(built.node_levels.values()))
        n = len(levels)
        # Binomial-ish standard error on 1000 samples is around 0.01; allow 4x.
        assert float(levels.mean()) == pytest.approx(
            1.0 / (built.M - 1), abs=4 * math.sqrt(1 / built.M / n) + 0.01
        )
        assert float(np.mean(levels >= 1)) == pytest.approx(
            1.0 / built.M, abs=4 * math.sqrt((1 / built.M) / n)
        )

    def test_level_distribution_is_exponential_not_uniform(self, source):
        """A large sample pins the geometric tail: P(level >= k) = M^-k."""
        index = HNSWIndex(32, Metric.L2, source, M=16, seed=99)
        draws = np.array([index._random_level() for _ in range(200_000)])
        for k in (1, 2):
            assert float(np.mean(draws >= k)) == pytest.approx(
                16.0**-k, rel=0.15
            ), f"P(level >= {k}) is off; level assignment is not exponential"

    def test_level_zero_is_the_common_case(self, built):
        levels = list(built.node_levels.values())
        fraction = sum(1 for lvl in levels if lvl == 0) / len(levels)
        assert 0.85 < fraction < 0.99

    def test_random_level_is_never_negative(self, source):
        index = HNSWIndex(32, Metric.L2, source, M=16, seed=7)
        assert all(index._random_level() >= 0 for _ in range(5000))

    def test_level_assignment_is_seed_deterministic(self, source):
        a = HNSWIndex(32, Metric.L2, source, M=16, seed=123)
        b = HNSWIndex(32, Metric.L2, source, M=16, seed=123)
        assert [a._random_level() for _ in range(100)] == [
            b._random_level() for _ in range(100)
        ]

    def test_m_of_one_is_rejected(self, source):
        """mL = 1/ln(1) divides by zero; fail loudly at construction."""
        with pytest.raises(ValueError, match="M must be >= 2"):
            HNSWIndex(32, Metric.L2, source, M=1)


class TestGraphInvariants:
    def test_no_structural_violations(self, built):
        assert built.validate() == []

    def test_degree_bounds_are_respected(self, built):
        for layer_no, layer in enumerate(built.layers):
            cap = built.M0 if layer_no == 0 else built.M
            worst = max((len(c) for c in layer.values()), default=0)
            assert worst <= cap, f"layer {layer_no}: degree {worst} > {cap}"

    def test_edges_are_symmetric_enough_to_be_navigable(self, built):
        """HNSW does not guarantee full symmetry — pruning can drop a back-edge —
        but the overwhelming majority must be mutual or the graph fragments."""
        layer = built.layers[0]
        total = mutual = 0
        for node, conns in layer.items():
            for n in conns:
                total += 1
                if node in layer.get(n, ()):
                    mutual += 1
        assert mutual / total > 0.75, f"only {mutual / total:.2%} of edges mutual"

    def test_layer_zero_graph_is_connected(self, built):
        """Every node reachable from the entry point. An unreachable node can
        never be returned, no matter how large ef_search gets."""
        layer = built.layers[0]
        seen = {built.entry_point}
        stack = [built.entry_point]
        while stack:
            for n in layer.get(stack.pop(), ()):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        assert len(seen) == len(layer), f"{len(layer) - len(seen)} unreachable nodes"

    def test_nodes_appear_on_every_layer_up_to_their_level(self, built):
        for node, level in built.node_levels.items():
            for lc in range(level + 1):
                assert node in built.layers[lc]
            if level + 1 < len(built.layers):
                assert node not in built.layers[level + 1]

    def test_entry_point_sits_on_the_top_layer(self, built):
        assert built.node_levels[built.entry_point] == built.max_level
        assert built.max_level == len(built.layers) - 1


class TestNeighbourSelection:
    """Trap #1: the heuristic (algorithm 4), not naive top-M."""

    def test_heuristic_prefers_diversity_over_pure_proximity(self, source):
        # Three candidates clustered together plus one far away in another
        # direction. Naive top-M with m=2 takes the two closest (the cluster);
        # the heuristic keeps one from the cluster and reaches for the outlier.
        x = np.array(
            [
                [0.0, 0.0],  # the query point itself, node 0
                [1.0, 0.0],  # cluster
                [1.05, 0.02],  # cluster, nearly on top of node 1
                [0.0, 1.4],  # different direction, further away
            ],
            dtype=np.float32,
        )
        index = HNSWIndex(2, Metric.L2, ArrayVectorSource(x), M=16, seed=1)
        q = x[0]
        from pyvec.core.distance import distance

        cands = [
            (float(distance(Metric.L2, q, x[i : i + 1])[0]), i) for i in (1, 2, 3)
        ]
        chosen = index._select_neighbours(cands, 2, keep_pruned=False)
        assert 1 in chosen, "should keep the nearest candidate"
        assert 3 in chosen, "should keep the diverse direction, not the twin"
        assert 2 not in chosen

    def test_returns_everything_when_under_budget(self, source):
        index = HNSWIndex(32, Metric.L2, source, M=16)
        cands = [(0.5, 1), (0.7, 2)]
        assert sorted(index._select_neighbours(cands, 8)) == [1, 2]

    def test_never_exceeds_the_budget(self, vectors, source):
        index = HNSWIndex(32, Metric.L2, source, M=16)
        cands = [(float(i), i) for i in range(50)]
        assert len(index._select_neighbours(cands, 5)) == 5

    def test_keep_pruned_backfills_to_the_budget(self, source):
        """keepPrunedConnections stops nodes ending up under-connected."""
        x = np.array([[0.0, 0.0]] + [[1.0, 0.01 * i] for i in range(10)], np.float32)
        index = HNSWIndex(2, Metric.L2, ArrayVectorSource(x), M=16)
        from pyvec.core.distance import distance

        q = x[0]
        cands = [
            (float(distance(Metric.L2, q, x[i : i + 1])[0]), i) for i in range(1, 11)
        ]
        assert len(index._select_neighbours(cands, 4, keep_pruned=True)) == 4
        assert len(index._select_neighbours(cands, 4, keep_pruned=False)) <= 4


class TestEntryPoint:
    """Trap #3."""

    def test_first_insert_becomes_the_entry_point(self, source):
        index = HNSWIndex(32, Metric.L2, source)
        assert index.entry_point is None
        index.add([0])
        assert index.entry_point == 0
        assert index.max_level >= 0
        assert index.validate() == []

    def test_search_on_an_empty_index_returns_nothing(self, source):
        assert HNSWIndex(32, Metric.L2, source).search(np.zeros(32), 5) == []

    def test_single_element_index(self, vectors, source):
        index = HNSWIndex(32, Metric.L2, source)
        index.add([7])
        assert [i for i, _ in index.search(vectors[7], 5)] == [7]

    def test_entry_point_moves_up_when_a_taller_node_arrives(self, source):
        index = HNSWIndex(32, Metric.L2, source, M=16, seed=42)
        index.add(range(300))
        assert index.node_levels[index.entry_point] == index.max_level
        assert index.max_level > 0, "300 nodes should produce more than one layer"

    def test_deleting_the_entry_point_keeps_search_working(self, vectors, source):
        """Tombstoning must not orphan the graph: the entry point stays as a
        routing waypoint even when it is no longer a valid result (ADR-010)."""
        index = HNSWIndex(32, Metric.L2, source, **BUILD)
        index.add(range(300))
        ep = index.entry_point
        index.remove([ep])
        results = index.search(vectors[ep], 10, ef_search=64)
        assert len(results) == 10
        assert ep not in {i for i, _ in results}


class TestDeletion:
    def test_deleted_ids_never_appear(self, vectors, source):
        index = HNSWIndex(32, Metric.L2, source, **BUILD)
        index.add(range(500))
        victims = set(range(0, 500, 5))
        index.remove(victims)
        for q in vectors[:10]:
            assert victims.isdisjoint({i for i, _ in index.search(q, 20, ef_search=64)})

    def test_still_returns_k_results_despite_heavy_deletion(self, vectors, source):
        """The beam has to widen when tombstones crowd out live candidates."""
        index = HNSWIndex(32, Metric.L2, source, **BUILD)
        index.add(range(500))
        index.remove(range(0, 400))  # 80% tombstoned
        results = index.search(vectors[0], 10, ef_search=16)
        assert len(results) == 10

    def test_len_reflects_deletions(self, source):
        index = HNSWIndex(32, Metric.L2, source, **BUILD)
        index.add(range(100))
        index.remove([1, 2, 3])
        assert len(index) == 97


class TestSearchContract:
    def test_results_are_sorted_by_distance(self, built, queries):
        d = [dist for _, dist in built.search(queries[0], 20, ef_search=64)]
        assert d == sorted(d)

    def test_ef_search_is_raised_to_at_least_k(self, built, queries):
        """A beam narrower than k cannot return k results."""
        assert len(built.search(queries[0], 25, ef_search=1)) == 25

    def test_k_zero_returns_nothing(self, built, queries):
        assert built.search(queries[0], 0) == []

    def test_never_returns_more_than_k(self, built, queries):
        assert len(built.search(queries[0], 7, ef_search=200)) == 7

    def test_default_ef_search_is_used_when_unspecified(self, built, queries):
        assert len(built.search(queries[0], 10)) == 10


class TestIncrementalBuild:
    def test_adding_in_batches_matches_adding_at_once(self, vectors, source, queries):
        one_shot = HNSWIndex(32, Metric.L2, source, **BUILD)
        one_shot.add(range(400))
        batched = HNSWIndex(32, Metric.L2, source, **BUILD)
        for start in range(0, 400, 50):
            batched.add(range(start, start + 50))
        # Identical seed and identical insertion order means identical graphs.
        assert one_shot.level_histogram() == batched.level_histogram()
        assert one_shot.search(queries[0], 10) == batched.search(queries[0], 10)

    def test_reinserting_an_existing_id_is_a_no_op(self, source):
        index = HNSWIndex(32, Metric.L2, source, **BUILD)
        index.add(range(50))
        histogram = index.level_histogram()
        index.add([10, 20])
        assert index.level_histogram() == histogram
        assert index.validate() == []

    def test_add_validates_vector_count(self, vectors, source):
        index = HNSWIndex(32, Metric.L2, source)
        with pytest.raises(ValueError, match="ids but"):
            index.add([0, 1, 2], vectors[:2])


class TestPersistence:
    def test_round_trip_preserves_graph_and_results(
        self, tmp_path, built, source, queries
    ):
        before = [built.search(q, 10, ef_search=64) for q in queries[:10]]
        built.save(tmp_path / "hnsw.idx")

        restored = HNSWIndex(32, Metric.L2, source)
        restored.load(tmp_path / "hnsw.idx")

        assert restored.M == built.M
        assert restored.M0 == built.M0
        assert restored.ef_construction == built.ef_construction
        assert restored.entry_point == built.entry_point
        assert restored.max_level == built.max_level
        assert restored.level_histogram() == built.level_histogram()
        assert restored.node_levels == built.node_levels
        assert restored.layers == built.layers
        assert restored.validate() == []
        after = [restored.search(q, 10, ef_search=64) for q in queries[:10]]
        assert after == before

    def test_tombstones_survive_a_round_trip(self, tmp_path, built, source, vectors):
        built.remove([1, 2, 3])
        built.save(tmp_path / "hnsw.idx")
        restored = HNSWIndex(32, Metric.L2, source)
        restored.load(tmp_path / "hnsw.idx")
        assert len(restored) == len(built)
        got = {i for i, _ in restored.search(vectors[1], 10, ef_search=64)}
        assert got.isdisjoint({1, 2, 3})

    def test_inserts_after_reload_stay_deterministic(self, tmp_path, vectors, source):
        """The level RNG is replayed on load, so a restart does not change the
        sequence of levels the next inserts get."""
        a = HNSWIndex(32, Metric.L2, source, **BUILD)
        a.add(range(200))
        a.save(tmp_path / "a.idx")
        a.add(range(200, 260))

        b = HNSWIndex(32, Metric.L2, source, **BUILD)
        b.load(tmp_path / "a.idx")
        b.add(range(200, 260))

        assert b.node_levels == a.node_levels
        assert b.level_histogram() == a.level_histogram()

    def test_l2_norm_cache_is_rebuilt_on_load(self, tmp_path, built, source, queries):
        built.save(tmp_path / "hnsw.idx")
        restored = HNSWIndex(32, Metric.L2, source)
        restored.load(tmp_path / "hnsw.idx")
        assert restored._sqnorms is not None
        assert restored.search(queries[0], 10, ef_search=64) == built.search(
            queries[0], 10, ef_search=64
        )

    def test_loading_a_missing_file_is_a_no_op(self, tmp_path, source):
        index = HNSWIndex(32, Metric.L2, source)
        index.load(tmp_path / "absent.idx")
        assert index.entry_point is None

    def test_corrupt_magic_is_rejected(self, tmp_path, source):
        from pyvec.core.errors import CorruptDataError

        path = tmp_path / "bad.idx"
        path.write_bytes(b"NOTHNSW!" + b"\x00" * 128)
        with pytest.raises(CorruptDataError, match="not an HNSW index"):
            HNSWIndex(32, Metric.L2, source).load(path)

    def test_truncated_header_is_rejected(self, tmp_path, source):
        from pyvec.core.errors import CorruptDataError

        path = tmp_path / "short.idx"
        path.write_bytes(b"PYVECHNS")
        with pytest.raises(CorruptDataError, match="truncated"):
            HNSWIndex(32, Metric.L2, source).load(path)


class TestParameterEffects:
    def test_larger_M_improves_recall(self, vectors, source, queries):
        """More edges per node means more paths to the true neighbours."""
        recalls = {}
        flat = FlatIndex(32, Metric.L2, source)
        flat.add(range(len(vectors)))
        for m in (4, 24):
            index = HNSWIndex(32, Metric.L2, source, M=m, ef_construction=100, seed=42)
            index.add(range(len(vectors)))
            recalls[m] = float(
                np.mean(
                    [
                        recall_at_k(flat.search(q, 10), index.search(q, 10, ef_search=32), 10)
                        for q in queries[:20]
                    ]
                )
            )
        assert recalls[24] > recalls[4], recalls

    def test_layer_zero_cap_defaults_to_double_M(self, source):
        assert HNSWIndex(32, Metric.L2, source, M=16).M0 == 32

    def test_params_are_reported_for_the_api(self, built):
        assert built.params == {
            "M": 16,
            "ef_construction": 100,
            "ef_search": 64,
        }


@pytest.mark.slow
class TestScale:
    """PROJECT_PLAN week 2: ">=90% recall vs FlatIndex on random 10k x 64 vectors".

    Random Gaussian vectors are the **hardest** case for a graph index, and it is
    worth being precise about why, because the number here looks lower than the
    project's headline target:

    * Isotropic Gaussian data in 64 dimensions has high *intrinsic*
      dimensionality and no cluster structure. Distances between a query and its
      100th neighbour are barely larger than to its 1st, so the greedy walk has
      very little gradient to follow.
    * Real datasets are not like this. SIFT descriptors and text embeddings lie
      near much lower-dimensional manifolds, which is exactly the structure
      HNSW's long edges exploit.

    Measured here at M=16, ef_construction=200: recall@10 is ~0.87 at
    ``ef_search=64`` and ~0.96 at ``ef_search=128``. PRD NF1's >=95% at
    ``ef_search=64`` is specified against **SIFT-1M**, not against random noise;
    ``benchmarks/sift_1m.py`` is what tests that claim.

    Every parameter was verified to move recall in the direction theory predicts
    (M=16 -> 32 takes ef=64 recall from 0.87 to 0.98; ef_construction saturates
    past 200), which is what distinguishes "hard data" from "broken index".
    """

    N = 10_000
    DIM = 64

    @pytest.fixture(scope="class")
    def scale_setup(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(self.N, self.DIM)).astype(np.float32)
        # 100 queries rather than 50: at 10 results each, 50 queries put the
        # standard error on the recall estimate near +-0.02, which is the same
        # size as the differences being asserted.
        q = rng.normal(size=(100, self.DIM)).astype(np.float32)
        src = ArrayVectorSource(x)
        index = HNSWIndex(
            self.DIM, Metric.L2, src, M=16, ef_construction=200, seed=42
        )
        index.add(range(len(x)))
        flat = FlatIndex(self.DIM, Metric.L2, src)
        flat.add(range(len(x)))
        truth = [flat.search(qq, 10) for qq in q]
        return index, q, truth

    def _recall(self, index, queries, truth, ef):
        return float(
            np.mean(
                [
                    recall_at_k(t, index.search(qq, 10, ef_search=ef), 10)
                    for qq, t in zip(queries, truth)
                ]
            )
        )

    def test_meets_the_90_percent_target(self, scale_setup):
        index, queries, truth = scale_setup
        mean = self._recall(index, queries, truth, ef=128)
        assert mean >= 0.90, f"recall@10 = {mean:.4f} at ef_search=128"

    def test_ef_64_lands_where_this_data_allows(self, scale_setup):
        """Regression guard on the hard case. Pinned loosely on purpose: the point
        is to catch a collapse to ~0.5 (naive neighbour selection) or ~0.8
        (broken level distribution), not to freeze a specific value."""
        index, queries, truth = scale_setup
        mean = self._recall(index, queries, truth, ef=64)
        assert 0.82 <= mean, f"recall@10 = {mean:.4f} at ef_search=64"

    def test_recall_is_monotonic_in_ef_at_scale(self, scale_setup):
        index, queries, truth = scale_setup
        means = [self._recall(index, queries, truth, ef) for ef in (16, 64, 128, 256)]
        assert means == sorted(means), f"not monotonic: {means}"
        assert means[-1] >= 0.98, f"a wide beam should find nearly everything: {means}"

    def test_graph_is_structurally_sound_at_scale(self, scale_setup):
        index, _, _ = scale_setup
        assert index.validate() == []
        assert len(index.layers[0]) == self.N
        # Reachability: an unreachable node can never be returned at any ef.
        layer = index.layers[0]
        seen = {index.entry_point}
        stack = [index.entry_point]
        while stack:
            for n in layer.get(stack.pop(), ()):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        assert len(seen) == self.N, f"{self.N - len(seen)} unreachable nodes"
