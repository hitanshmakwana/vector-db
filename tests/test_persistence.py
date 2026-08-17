"""Durability: checkpoint, reopen, and WAL replay on a clean shutdown.

Crash paths live in test_crash_safety.py. This file covers the ordinary case —
PRD UC4, "a developer restarts the server; all previously inserted data is
intact and queryable" — plus the manifest/snapshot round trip.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyvec.core.collection import (
    MANIFEST_FILE,
    METADATA_FILE,
    Collection,
)
from pyvec.core.collection_manager import CollectionManager
from pyvec.core.errors import CorruptDataError
from tests.conftest import TEXT_CORPUS

INDEX_TYPES = ["flat", "hnsw", "ivf"]
DIM = 8


def make_items(n, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "id": f"d{i}",
            "vector": rng.normal(size=DIM).astype(np.float32).tolist(),
            "metadata": {
                "content": TEXT_CORPUS[i % len(TEXT_CORPUS)],
                "n": i,
                "category": "even" if i % 2 == 0 else "odd",
            },
        }
        for i in range(n)
    ]


class TestReopen:
    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_data_survives_close_and_reopen(self, data_root, index_type):
        items = make_items(50)
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type=index_type,
            text_field="content",
            index_params={"nlist": 4} if index_type == "ivf" else {},
        )
        c.insert(items)
        before = [h.id for h in c.search(items[0]["vector"], k=5)]
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            assert len(reopened) == 50
            assert [h.id for h in reopened.search(items[0]["vector"], k=5)] == before
            for i in (0, 17, 49):
                assert reopened.contains(f"d{i}")
                assert reopened.get(f"d{i}")["metadata"]["n"] == i
        finally:
            reopened.close()

    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_vectors_survive_bit_exact(self, data_root, index_type):
        """float32 in, identical float32 out — the mmap holds raw bytes."""
        items = make_items(20)
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type=index_type,
            index_params={"nlist": 4} if index_type == "ivf" else {},
            metric="l2",  # no normalisation, so the stored bytes are the input
        )
        c.insert(items)
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            for item in items:
                np.testing.assert_array_equal(
                    np.asarray(reopened.get(item["id"])["vector"], dtype=np.float32),
                    np.asarray(item["vector"], dtype=np.float32),
                )
        finally:
            reopened.close()

    def test_configuration_survives(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, metric="l2", index_type="hnsw",
            index_params={"M": 8, "ef_construction": 42}, text_field="content",
            bm25_params={"k1": 1.2, "b": 0.5},
        )
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            assert reopened.dimension == DIM
            assert reopened.metric.value == "l2"
            assert reopened.index_type.value == "hnsw"
            assert reopened.dense.M == 8
            assert reopened.dense.ef_construction == 42
            assert reopened.text_field == "content"
            assert reopened.sparse.k1 == 1.2 and reopened.sparse.b == 0.5
        finally:
            reopened.close()

    def test_tombstones_survive(self, data_root):
        items = make_items(20)
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(items)
        c.delete("d5")
        c.delete("d9")
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            assert len(reopened) == 18
            assert reopened.num_deleted == 2
            assert not reopened.contains("d5")
            found = {h.id for h in reopened.search(items[5]["vector"], k=20)}
            assert found.isdisjoint({"d5", "d9"})
        finally:
            reopened.close()

    def test_bm25_index_survives(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, text_field="content", index_type="flat"
        )
        c.insert(make_items(20))
        before = [h.id for h in c.search_text("quick brown fox", k=5)]
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            assert [h.id for h in reopened.search_text("quick brown fox", k=5)] == before
            assert reopened.sparse.num_docs == 20
        finally:
            reopened.close()

    def test_bm25_is_rebuilt_when_its_snapshot_is_missing(self, data_root):
        """ARCHITECTURE.md startup path: BM25 is cheap to rebuild from metadata,
        so a missing sparse file must not be fatal."""
        from pyvec.core.collection import SPARSE_INDEX_FILE

        c = Collection.create(
            "docs", data_root, dimension=DIM, text_field="content", index_type="flat"
        )
        c.insert(make_items(20))
        before = [h.id for h in c.search_text("quick brown fox", k=5)]
        c.close()
        (data_root / "docs" / SPARSE_INDEX_FILE).unlink()

        reopened = Collection.open(data_root / "docs")
        try:
            assert [h.id for h in reopened.search_text("quick brown fox", k=5)] == before
        finally:
            reopened.close()

    def test_inserts_continue_after_reopen(self, data_root):
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(make_items(10))
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            more = [
                {"id": f"x{i}", "vector": [float(i)] * DIM} for i in range(5)
            ]
            reopened.insert(more)
            assert len(reopened) == 15
            # New ids must not collide with the restored ones.
            assert len(set(reopened._id_map.values())) == 15
        finally:
            reopened.close()

    def test_repeated_reopen_is_stable(self, data_root):
        items = make_items(20)
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(items)
        c.close()
        expected = None
        for _ in range(3):
            c = Collection.open(data_root / "docs")
            got = [h.id for h in c.search(items[0]["vector"], k=5)]
            assert len(c) == 20
            if expected is None:
                expected = got
            assert got == expected
            c.close()


class TestCheckpoint:
    def test_checkpoint_truncates_the_wal(self, data_root):
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        try:
            c.insert(make_items(10))
            assert c.wal.size_bytes > 0
            c.checkpoint()
            assert c.wal.size_bytes == 0
        finally:
            c.close()

    def test_close_checkpoints(self, data_root):
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(make_items(10))
        c.close()
        assert (data_root / "docs" / "wal.log").stat().st_size == 0

    def test_manifest_records_the_id_counter(self, data_root):
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(make_items(7))
        c.close()
        manifest = json.loads((data_root / "docs" / MANIFEST_FILE).read_text())
        assert manifest["next_internal_id"] == 7
        assert manifest["num_rows"] == 7
        assert manifest["dimension"] == DIM

    def test_metadata_file_is_valid_json(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat", text_field="content"
        )
        c.insert(make_items(5))
        c.close()
        payload = json.loads((data_root / "docs" / METADATA_FILE).read_text())
        assert set(payload) == {"ids", "metadata", "deleted"}
        assert payload["ids"]["0"] == "d0"

    def test_no_temp_files_are_left_behind(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="hnsw", text_field="content"
        )
        c.insert(make_items(10))
        c.checkpoint()
        c.close()
        leftovers = list((data_root / "docs").glob("*.tmp"))
        assert leftovers == [], f"atomic writes leaked temp files: {leftovers}"


class TestSnapshot:
    def test_snapshot_copies_the_durable_files(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="hnsw", text_field="content"
        )
        try:
            c.insert(make_items(10))
            result = c.snapshot()
            from pathlib import Path

            dest = Path(result["path"])
            assert dest.is_dir()
            names = {p.name for p in dest.iterdir()}
            assert {MANIFEST_FILE, METADATA_FILE, "vectors.bin", "flags.bin"} <= names
            assert result["snapshot_id"] in str(dest)
        finally:
            c.close()

    def test_a_snapshot_directory_can_be_opened_as_a_collection(self, data_root):
        """The files are self-contained, so a snapshot is a restorable copy."""
        from pathlib import Path

        items = make_items(15)
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat", text_field="content"
        )
        c.insert(items)
        snapshot_path = Path(c.snapshot()["path"])
        before = [h.id for h in c.search(items[0]["vector"], k=5)]
        c.close()

        restored = Collection.open(snapshot_path)
        try:
            assert len(restored) == 15
            assert [h.id for h in restored.search(items[0]["vector"], k=5)] == before
        finally:
            restored.close()

    def test_snapshot_truncates_the_wal(self, data_root):
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        try:
            c.insert(make_items(5))
            c.snapshot()
            assert c.wal.size_bytes == 0
        finally:
            c.close()


class TestGrowth:
    def test_capacity_doubling_preserves_everything(self, data_root):
        """Start deliberately undersized so the store has to grow repeatedly."""
        items = make_items(300)
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat", capacity=4
        )
        c.insert(items)
        assert c.store.capacity >= 300
        c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            assert len(reopened) == 300
            for i in (0, 150, 299):
                np.testing.assert_allclose(
                    reopened.get(f"d{i}")["vector"],
                    np.asarray(items[i]["vector"], dtype=np.float32)
                    / np.linalg.norm(items[i]["vector"]),
                    atol=1e-6,
                )
        finally:
            reopened.close()


class TestManagerStartup:
    def test_loads_every_collection_on_disk(self, data_root):
        for name in ("alpha", "beta", "gamma"):
            c = Collection.create(name, data_root, dimension=DIM, index_type="flat")
            c.insert(make_items(5))
            c.close()

        manager = CollectionManager(data_root)
        try:
            assert {row["name"] for row in manager.list()} == {"alpha", "beta", "gamma"}
            assert manager.get("beta").num_vectors == 5
            assert manager.load_errors == {}
        finally:
            manager.close()

    def test_a_corrupt_collection_does_not_block_the_others(self, data_root):
        for name in ("good", "bad"):
            c = Collection.create(name, data_root, dimension=DIM, index_type="flat")
            c.insert(make_items(5))
            c.close()
        (data_root / "bad" / MANIFEST_FILE).write_text("{ not json", encoding="utf-8")

        manager = CollectionManager(data_root)
        try:
            assert [row["name"] for row in manager.list()] == ["good"]
            assert "bad" in manager.load_errors
        finally:
            manager.close()

    def test_directories_without_a_manifest_are_ignored(self, data_root):
        (data_root / "not-a-collection").mkdir()
        (data_root / "not-a-collection" / "README").write_text("hi")
        manager = CollectionManager(data_root)
        try:
            assert manager.list() == []
        finally:
            manager.close()

    def test_manager_checkpoint_all(self, data_root):
        manager = CollectionManager(data_root)
        try:
            c = manager.create("docs", dimension=DIM, index_type="flat")
            c.insert(make_items(5))
            assert manager.checkpoint_all()["docs"]["num_vectors"] == 5
            assert c.wal.size_bytes == 0
        finally:
            manager.close()

    def test_drop_deletes_the_files(self, data_root):
        manager = CollectionManager(data_root)
        try:
            manager.create("docs", dimension=DIM, index_type="flat")
            assert (data_root / "docs").exists()
            manager.drop("docs")
            assert not (data_root / "docs").exists()
            assert manager.list() == []
        finally:
            manager.close()

    def test_creating_a_duplicate_name_is_rejected(self, data_root):
        from pyvec.core.errors import CollectionExistsError

        manager = CollectionManager(data_root)
        try:
            manager.create("docs", dimension=DIM)
            with pytest.raises(CollectionExistsError):
                manager.create("docs", dimension=DIM)
        finally:
            manager.close()

    @pytest.mark.parametrize("name", ["", "  ", "..", "a/b", "a\\b", "a:b", "a*b"])
    def test_unsafe_collection_names_are_rejected(self, data_root, name):
        """Names become directory names, so path traversal has to be blocked."""
        from pyvec.core.errors import InvalidRequestError

        manager = CollectionManager(data_root)
        try:
            with pytest.raises(InvalidRequestError):
                manager.create(name, dimension=DIM)
        finally:
            manager.close()


class TestCorruptState:
    def test_missing_manifest_is_reported(self, data_root):
        (data_root / "empty").mkdir()
        with pytest.raises(CorruptDataError, match="no manifest"):
            Collection.open(data_root / "empty")

    def test_unknown_manifest_version_is_reported(self, data_root):
        c = Collection.create("docs", data_root, dimension=DIM)
        c.close()
        path = data_root / "docs" / MANIFEST_FILE
        manifest = json.loads(path.read_text())
        manifest["version"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(CorruptDataError, match="manifest version 999"):
            Collection.open(data_root / "docs")
