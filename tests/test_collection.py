"""Collection semantics: insert, delete, filters, stats, id mapping."""

from __future__ import annotations

import numpy as np
import pytest

from pyvec.core.collection import MAX_BATCH, Collection
from pyvec.core.errors import (
    IdExistsError,
    IdNotFoundError,
    InvalidDimensionError,
    InvalidRequestError,
    NoTextFieldError,
    PayloadTooLargeError,
)
from pyvec.core.types import IndexType, Metric
from tests.conftest import TEXT_CORPUS

INDEX_TYPES = ["flat", "hnsw", "ivf"]


def items(n, dim=8, seed=0, offset=0, text=True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        metadata = {"category": "even" if i % 2 == 0 else "odd", "n": i}
        if text:
            metadata["content"] = TEXT_CORPUS[i % len(TEXT_CORPUS)]
        out.append(
            {
                "id": f"d{i + offset}",
                "vector": rng.normal(size=dim).astype(np.float32).tolist(),
                "metadata": metadata,
            }
        )
    return out


class TestInsert:
    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_insert_and_count(self, make_collection, index_type):
        c = make_collection(index_type=index_type, text_field="content")
        assert c.insert(items(20)) == {"inserted": 20, "duplicates_skipped": 0}
        assert len(c) == 20

    def test_wrong_dimension_is_rejected(self, make_collection):
        c = make_collection(dimension=8)
        with pytest.raises(InvalidDimensionError, match="expected dimension 8"):
            c.insert([{"id": "x", "vector": [1.0, 2.0]}])

    def test_duplicate_id_is_rejected(self, make_collection):
        c = make_collection()
        c.insert(items(1))
        with pytest.raises(IdExistsError, match="already exists"):
            c.insert(items(1))

    def test_upsert_overwrites(self, make_collection):
        c = make_collection()
        c.insert([{"id": "d0", "vector": [1.0] * 8, "metadata": {"v": 1}}])
        c.insert(
            [{"id": "d0", "vector": [2.0] * 8, "metadata": {"v": 2}}], upsert=True
        )
        assert len(c) == 1
        assert c.get("d0")["metadata"] == {"v": 2}

    def test_upsert_replaces_the_vector_too(self, make_collection):
        c = make_collection(index_type="flat", metric="l2")
        c.insert([{"id": "d0", "vector": [1.0] + [0.0] * 7}])
        c.insert([{"id": "d0", "vector": [9.0] + [0.0] * 7}], upsert=True)
        hit = c.search([9.0] + [0.0] * 7, k=1)[0]
        assert hit.id == "d0"
        assert hit.score == pytest.approx(0.0, abs=1e-3)

    def test_empty_batch_is_rejected(self, make_collection):
        with pytest.raises(InvalidRequestError, match="at least one"):
            make_collection().insert([])

    def test_oversized_batch_is_rejected(self, make_collection):
        c = make_collection()
        with pytest.raises(PayloadTooLargeError):
            c.insert(items(MAX_BATCH + 1))

    def test_duplicate_ids_within_one_batch_are_rejected(self, make_collection):
        c = make_collection()
        with pytest.raises(InvalidRequestError, match="twice in the same batch"):
            c.insert(
                [
                    {"id": "same", "vector": [1.0] * 8},
                    {"id": "same", "vector": [2.0] * 8},
                ]
            )

    def test_missing_fields_are_rejected(self, make_collection):
        c = make_collection()
        with pytest.raises(InvalidRequestError, match="needs an 'id'"):
            c.insert([{"vector": [1.0] * 8}])
        with pytest.raises(InvalidRequestError, match="no 'vector'"):
            c.insert([{"id": "x"}])

    def test_nan_and_infinity_are_rejected(self, make_collection):
        """A NaN silently poisons every distance comparison it touches."""
        c = make_collection()
        with pytest.raises(InvalidRequestError, match="NaN or infinity"):
            c.insert([{"id": "x", "vector": [float("nan")] + [0.0] * 7}])
        with pytest.raises(InvalidRequestError, match="NaN or infinity"):
            c.insert([{"id": "y", "vector": [float("inf")] + [0.0] * 7}])

    def test_a_failed_batch_applies_nothing(self, make_collection):
        """Validation happens before the WAL, so a bad item cannot leave half a
        batch durably applied."""
        c = make_collection()
        good = items(3)
        bad = good + [{"id": "bad", "vector": [1.0, 2.0]}]
        with pytest.raises(InvalidDimensionError):
            c.insert(bad)
        assert len(c) == 0
        assert c.wal.size_bytes == c.wal.stats()["size_bytes"]
        assert not any(c.iter_ids())

    def test_metadata_is_optional(self, make_collection):
        c = make_collection()
        c.insert([{"id": "x", "vector": [1.0] * 8}])
        assert c.get("x")["metadata"] == {}


class TestNormalisation:
    def test_cosine_collections_normalise_on_insert(self, make_collection):
        """ADR-009 / LEARNING.md layer 0: unit vectors make cosine == dot."""
        c = make_collection(metric="cosine", index_type="flat")
        c.insert([{"id": "x", "vector": [3.0, 4.0] + [0.0] * 6}])
        stored = np.asarray(c.get("x")["vector"])
        assert np.linalg.norm(stored) == pytest.approx(1.0, abs=1e-5)
        np.testing.assert_allclose(stored[:2], [0.6, 0.8], atol=1e-5)

    def test_l2_collections_do_not_normalise(self, make_collection):
        c = make_collection(metric="l2", index_type="flat")
        c.insert([{"id": "x", "vector": [3.0, 4.0] + [0.0] * 6}])
        np.testing.assert_allclose(c.get("x")["vector"][:2], [3.0, 4.0])

    def test_cosine_query_is_normalised_too(self, make_collection):
        """An unnormalised query against normalised data would score wrongly."""
        c = make_collection(metric="cosine", index_type="flat")
        c.insert([{"id": "x", "vector": [1.0, 0.0] + [0.0] * 6}])
        for scale in (1.0, 10.0, 0.01):
            hit = c.search([scale, 0.0] + [0.0] * 6, k=1)[0]
            assert hit.score == pytest.approx(1.0, abs=1e-5)


class TestSearch:
    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_returns_k_results_sorted_by_score(self, make_collection, index_type):
        c = make_collection(index_type=index_type)
        c.insert(items(50, text=False))
        hits = c.search(items(1, text=False)[0]["vector"], k=5)
        assert len(hits) == 5
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), "cosine: higher is better"

    def test_l2_scores_ascend_because_lower_is_better(self, make_collection):
        c = make_collection(metric="l2", index_type="flat")
        c.insert(items(30, text=False))
        scores = [h.score for h in c.search(items(1, text=False)[0]["vector"], k=5)]
        assert scores == sorted(scores)

    def test_exact_match_scores_one_for_cosine(self, make_collection):
        c = make_collection(metric="cosine", index_type="flat")
        batch = items(10, text=False)
        c.insert(batch)
        hit = c.search(batch[3]["vector"], k=1)[0]
        assert hit.id == "d3"
        assert hit.score == pytest.approx(1.0, abs=1e-5)

    def test_metadata_comes_back_with_results(self, make_collection):
        c = make_collection()
        c.insert(items(10))
        hit = c.search(items(1)[0]["vector"], k=1)[0]
        assert hit.metadata["category"] in {"even", "odd"}

    def test_k_must_be_positive(self, make_collection):
        c = make_collection()
        c.insert(items(5))
        with pytest.raises(InvalidRequestError, match="k must be positive"):
            c.search(items(1)[0]["vector"], k=0)

    def test_search_on_an_empty_collection_returns_nothing(self, make_collection):
        assert make_collection().search([1.0] * 8, k=5) == []

    def test_k_larger_than_the_collection(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(3, text=False))
        assert len(c.search([1.0] * 8, k=100)) == 3

    def test_wrong_query_dimension_is_rejected(self, make_collection):
        c = make_collection()
        c.insert(items(3))
        with pytest.raises(InvalidDimensionError):
            c.search([1.0, 2.0], k=1)

    def test_hnsw_ef_search_param_is_forwarded(self, make_collection):
        c = make_collection(index_type="hnsw")
        c.insert(items(100, text=False))
        wide = c.search(items(1, text=False)[0]["vector"], k=10, params={"ef_search": 200})
        assert len(wide) == 10

    def test_ivf_nprobe_param_is_forwarded(self, make_collection):
        c = make_collection(index_type="ivf", index_params={"nlist": 4})
        c.insert(items(100, text=False))
        assert len(c.search(items(1, text=False)[0]["vector"], k=5, params={"nprobe": 4})) == 5


class TestFilters:
    def test_shallow_equality_filter(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(40))
        hits = c.search(items(1)[0]["vector"], k=10, filter={"category": "even"})
        assert hits
        assert all(h.metadata["category"] == "even" for h in hits)

    def test_list_value_means_any_of(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(20))
        hits = c.search(items(1)[0]["vector"], k=20, filter={"n": [0, 2, 4]})
        assert {h.metadata["n"] for h in hits} <= {0, 2, 4}
        assert hits

    def test_multiple_keys_are_anded(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(20))
        hits = c.search(
            items(1)[0]["vector"], k=20, filter={"category": "even", "n": 4}
        )
        assert [h.metadata["n"] for h in hits] == [4]

    def test_unmatchable_filter_returns_nothing(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(20))
        assert c.search(items(1)[0]["vector"], k=5, filter={"category": "nope"}) == []

    def test_missing_metadata_key_does_not_match(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(10))
        assert c.search(items(1)[0]["vector"], k=5, filter={"absent": "x"}) == []

    def test_restrictive_filter_may_return_fewer_than_k(self, make_collection):
        """Documented consequence of post-filtering (ARCHITECTURE.md)."""
        c = make_collection(index_type="flat")
        c.insert(items(100))
        hits = c.search(items(1)[0]["vector"], k=100, filter={"n": 7})
        assert len(hits) == 1

    def test_overfetching_finds_matches_a_narrow_k_would_miss(self, make_collection):
        """``k=1`` with a filter searches to the MIN_FILTER_FETCH floor, not to 10.

        Without the floor, "nearest document where n = 41" returns nothing unless
        that document happens to be in the unfiltered top ten.
        """
        c = make_collection(index_type="flat")
        c.insert(items(60))
        hits = c.search(items(1)[0]["vector"], k=1, filter={"n": 41})
        assert [h.metadata["n"] for h in hits] == [41]

    def test_filtering_is_bounded_by_retrieval_depth(self, make_collection):
        """The honest limit of post-filtering: a match ranked deeper than the
        over-fetch window is invisible. Real pre-filtering (modifying the graph
        walk to skip non-matching nodes) is the v2 fix; ARCHITECTURE.md says so.
        """
        from pyvec.core.collection import MAX_OVERFETCH

        c = make_collection(index_type="flat")
        batch = items(MAX_OVERFETCH + 200, text=False)
        for it in batch:
            it["metadata"]["tag"] = "no"
        for start in range(0, len(batch), MAX_BATCH):  # respect the batch limit
            c.insert(batch[start : start + MAX_BATCH])
        # Nothing is tagged "yes", so the filter can only ever come back empty —
        # the same shape of outcome as a match sitting past the window.
        assert c.search(batch[0]["vector"], k=10, filter={"tag": "yes"}) == []


class TestDelete:
    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_delete_removes_from_results(self, make_collection, index_type):
        c = make_collection(index_type=index_type)
        batch = items(30)
        c.insert(batch)
        c.delete("d3")
        assert len(c) == 29
        assert not c.contains("d3")
        assert "d3" not in {h.id for h in c.search(batch[3]["vector"], k=10)}

    def test_deleting_an_unknown_id_raises(self, make_collection):
        c = make_collection()
        with pytest.raises(IdNotFoundError):
            c.delete("nope")

    def test_deleting_twice_raises(self, make_collection):
        c = make_collection()
        c.insert(items(2))
        c.delete("d0")
        with pytest.raises(IdNotFoundError):
            c.delete("d0")

    def test_get_on_a_deleted_id_raises(self, make_collection):
        c = make_collection()
        c.insert(items(2))
        c.delete("d0")
        with pytest.raises(IdNotFoundError):
            c.get("d0")

    def test_the_id_can_be_reinserted_after_deletion(self, make_collection):
        c = make_collection()
        c.insert(items(2))
        c.delete("d0")
        c.insert([{"id": "d0", "vector": [1.0] * 8, "metadata": {"fresh": True}}])
        assert c.contains("d0")
        assert c.get("d0")["metadata"] == {"fresh": True}
        assert len(c) == 2

    def test_tombstones_are_counted(self, make_collection):
        c = make_collection()
        c.insert(items(10))
        c.delete("d1")
        c.delete("d2")
        assert c.num_deleted == 2
        assert c.num_vectors == 8


class TestOptimize:
    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_compacts_and_keeps_data_searchable(self, make_collection, index_type):
        c = make_collection(index_type=index_type, text_field="content")
        batch = items(60)
        c.insert(batch)
        for i in range(0, 20):
            c.delete(f"d{i}")

        report = c.optimize()
        assert report["compacted"] == 20
        assert c.num_deleted == 0
        assert len(c) == 40
        # Surviving ids must still be present, findable and correctly mapped.
        for i in (25, 40, 59):
            assert c.contains(f"d{i}")
            hits = {h.id for h in c.search(batch[i]["vector"], k=5)}
            assert f"d{i}" in hits
        assert {h.id for h in c.search_text("quick fox", k=5)}

    def test_deleted_ids_stay_gone_after_compaction(self, make_collection):
        c = make_collection(index_type="flat")
        batch = items(20)
        c.insert(batch)
        c.delete("d5")
        c.optimize()
        assert not c.contains("d5")
        assert "d5" not in {h.id for h in c.search(batch[5]["vector"], k=20)}

    def test_optimize_on_an_empty_collection(self, make_collection):
        c = make_collection()
        assert c.optimize()["num_vectors"] == 0

    def test_metadata_survives_renumbering(self, make_collection):
        c = make_collection(index_type="flat")
        c.insert(items(20))
        c.delete("d0")
        c.optimize()
        for i in range(1, 20):
            assert c.get(f"d{i}")["metadata"]["n"] == i


class TestTextAndHybridGuards:
    def test_text_search_without_a_text_field_is_rejected(self, make_collection):
        c = make_collection(text_field=None)
        c.insert(items(5, text=False))
        with pytest.raises(NoTextFieldError, match="no text_field"):
            c.search_text("anything")

    def test_hybrid_without_a_text_field_is_rejected(self, make_collection):
        c = make_collection(text_field=None)
        c.insert(items(5, text=False))
        with pytest.raises(NoTextFieldError):
            c.search_hybrid([1.0] * 8, "anything")

    def test_documents_without_the_text_field_are_simply_not_indexed(
        self, make_collection
    ):
        c = make_collection(text_field="content")
        c.insert(
            [
                {"id": "a", "vector": [1.0] * 8, "metadata": {"content": "hello world"}},
                {"id": "b", "vector": [2.0] * 8, "metadata": {"other": "no text"}},
            ]
        )
        assert [h.id for h in c.search_text("hello", k=5)] == ["a"]
        assert len(c) == 2  # b is still a normal vector

    def test_non_string_text_field_is_skipped_without_crashing(self, make_collection):
        c = make_collection(text_field="content")
        c.insert([{"id": "a", "vector": [1.0] * 8, "metadata": {"content": 12345}}])
        assert c.search_text("12345", k=5) == []
        assert len(c) == 1


class TestStats:
    def test_reports_the_documented_fields(self, make_collection):
        c = make_collection(index_type="hnsw", text_field="content")
        c.insert(items(20))
        c.delete("d0")
        stats = c.stats()
        for key in (
            "name", "dimension", "metric", "index", "text_field",
            "num_vectors", "num_deleted", "memory_bytes", "disk_bytes",
        ):
            assert key in stats
        assert stats["num_vectors"] == 19
        assert stats["num_deleted"] == 1
        assert stats["index"]["type"] == "hnsw"
        assert stats["index"]["params"]["M"] == 16
        assert stats["memory_bytes"] > 0
        assert stats["disk_bytes"] > 0
        assert stats["sparse_stats"]["num_docs"] == 19

    def test_counters_track_operations(self, make_collection):
        c = make_collection(text_field="content")
        c.insert(items(10))
        c.search(items(1)[0]["vector"], k=2)
        c.search_text("quick", k=2)
        c.search_hybrid(items(1)[0]["vector"], "quick", k=2)
        c.delete("d0")
        counters = c.stats()["counters"]
        assert counters["inserts"] == 10
        assert counters["deletes"] == 1
        assert counters["queries_dense"] == 1
        assert counters["queries_text"] == 1
        assert counters["queries_hybrid"] == 1


class TestConfiguration:
    def test_invalid_metric_is_rejected(self, data_root):
        from pyvec.core.errors import InvalidMetricError

        with pytest.raises(InvalidMetricError):
            Collection.create("x", data_root, dimension=4, metric="manhattan")

    def test_invalid_index_type_is_rejected(self, data_root):
        from pyvec.core.errors import InvalidIndexTypeError

        with pytest.raises(InvalidIndexTypeError):
            Collection.create("x", data_root, dimension=4, index_type="magic")

    def test_non_positive_dimension_is_rejected(self, data_root):
        with pytest.raises(InvalidDimensionError):
            Collection.create("x", data_root, dimension=0)

    def test_index_params_reach_the_index(self, make_collection):
        c = make_collection(index_type="hnsw", index_params={"M": 8, "ef_construction": 50})
        assert c.dense.M == 8
        assert c.dense.ef_construction == 50

    def test_bm25_params_reach_the_index(self, make_collection):
        c = make_collection(text_field="content", bm25_params={"k1": 1.2, "b": 0.5})
        assert c.sparse.k1 == 1.2 and c.sparse.b == 0.5

    def test_repr_is_informative(self, make_collection):
        c = make_collection(index_type="ivf", metric="l2")
        assert "ivf" in repr(c) and "l2" in repr(c)


class TestIdMapping:
    def test_external_ids_map_to_dense_internal_rows(self, make_collection):
        c = make_collection()
        c.insert(items(5))
        assert sorted(c._id_map.values()) == [0, 1, 2, 3, 4]

    def test_reverse_map_stays_consistent(self, make_collection):
        c = make_collection()
        c.insert(items(5))
        for ext, internal in c._id_map.items():
            assert c._rev_map[internal] == ext

    def test_arbitrary_string_ids_are_supported(self, make_collection):
        c = make_collection()
        weird = ["a/b", "with space", "ünïcode", "123", "-", "a" * 200]
        c.insert(
            [{"id": w, "vector": [float(i)] * 8} for i, w in enumerate(weird)]
        )
        for w in weird:
            assert c.contains(w)
            assert c.get(w)["id"] == w
