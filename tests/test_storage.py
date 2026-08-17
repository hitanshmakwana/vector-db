"""Vector store, WAL, and the reader-writer lock."""

from __future__ import annotations

import struct
import threading
import time
import zlib

import numpy as np
import pytest

from pyvec.core.errors import CorruptDataError, InvalidDimensionError
from pyvec.core.rwlock import RWLock
from pyvec.storage.mmap_store import FLAG_DELETED, VectorStore
from pyvec.storage.wal import WAL, FsyncPolicy, OpType


class TestVectorStore:
    def test_append_and_read_back(self, tmp_path, rng):
        with VectorStore(tmp_path / "c", dim=8, capacity=16) as store:
            v = rng.normal(size=8).astype(np.float32)
            row = store.append(v)
            assert row == 0
            np.testing.assert_array_equal(store.get(row), v)

    def test_rows_are_handed_out_monotonically(self, tmp_path, rng):
        with VectorStore(tmp_path / "c", dim=4, capacity=8) as store:
            rows = [store.append(rng.normal(size=4).astype(np.float32)) for _ in range(5)]
            assert rows == [0, 1, 2, 3, 4]
            assert store.num_rows == 5

    def test_append_batch_matches_individual_appends(self, tmp_path, rng):
        x = rng.normal(size=(10, 4)).astype(np.float32)
        with VectorStore(tmp_path / "a", dim=4, capacity=16) as batch:
            assert batch.append_batch(x) == list(range(10))
            np.testing.assert_array_equal(batch.gather(range(10)), x)

    def test_gather_returns_rows_in_the_requested_order(self, tmp_path, rng):
        x = rng.normal(size=(10, 4)).astype(np.float32)
        with VectorStore(tmp_path / "c", dim=4) as store:
            store.append_batch(x)
            np.testing.assert_array_equal(store.gather([3, 1, 7]), x[[3, 1, 7]])

    def test_dimension_is_enforced(self, tmp_path):
        with VectorStore(tmp_path / "c", dim=4) as store:
            with pytest.raises(InvalidDimensionError):
                store.append(np.zeros(5, dtype=np.float32))
            with pytest.raises(InvalidDimensionError):
                store.append_batch(np.zeros((2, 3), dtype=np.float32))

    def test_growth_doubles_capacity_and_preserves_data(self, tmp_path, rng):
        x = rng.normal(size=(40, 4)).astype(np.float32)
        with VectorStore(tmp_path / "c", dim=4, capacity=8) as store:
            for v in x:
                store.append(v)
            assert store.capacity >= 40
            assert store.capacity == 64, "should double: 8 -> 16 -> 32 -> 64"
            np.testing.assert_array_equal(store.gather(range(40)), x)

    def test_growth_preserves_tombstones(self, tmp_path, rng):
        with VectorStore(tmp_path / "c", dim=4, capacity=4) as store:
            for _ in range(4):
                store.append(rng.normal(size=4).astype(np.float32))
            store.mark_deleted(1)
            for _ in range(20):
                store.append(rng.normal(size=4).astype(np.float32))
            assert store.is_deleted(1)
            assert store.deleted_ids() == {1}

    def test_reopen_recovers_rows_and_capacity(self, tmp_path, rng):
        x = rng.normal(size=(6, 4)).astype(np.float32)
        store = VectorStore(tmp_path / "c", dim=4, capacity=8)
        store.append_batch(x)
        store.mark_deleted(2)
        store.close()

        reopened = VectorStore(tmp_path / "c", dim=4)
        reopened.num_rows = 6  # the collection restores this from its manifest
        assert reopened.capacity == 8
        np.testing.assert_array_equal(reopened.gather(range(6)), x)
        assert reopened.is_deleted(2)
        reopened.close()

    def test_wrong_dimension_on_reopen_is_detected_when_it_can_be(self, tmp_path, rng):
        """Best-effort check: the file is a bare array with no header (per
        ARCHITECTURE.md), so the only signal available is whether its size
        divides evenly by the row width.

        That catches a mismatch like 6 -> 5 but *not* 6 -> 3, where the size
        happens to stay divisible. The manifest is the authoritative record of a
        collection's dimension; this is a cheap sanity net underneath it, not a
        substitute.
        """
        store = VectorStore(tmp_path / "c", dim=6, capacity=4)
        store.append(rng.normal(size=6).astype(np.float32))
        store.close()
        with pytest.raises(CorruptDataError, match="not a multiple"):
            VectorStore(tmp_path / "c", dim=5)

    def test_a_divisible_dimension_mismatch_slips_past_the_store(self, tmp_path, rng):
        """Documents the limitation above, so the gap is a known one."""
        store = VectorStore(tmp_path / "c", dim=6, capacity=4)
        store.append(rng.normal(size=6).astype(np.float32))
        store.close()
        sneaky = VectorStore(tmp_path / "c", dim=3)
        assert sneaky.capacity == 8  # 96 bytes / 12 bytes-per-row
        sneaky.close()

    def test_tombstone_accounting(self, tmp_path, rng):
        with VectorStore(tmp_path / "c", dim=4, capacity=16) as store:
            store.append_batch(rng.normal(size=(10, 4)).astype(np.float32))
            store.mark_deleted(0)
            store.mark_deleted(5)
            assert store.num_deleted == 2
            assert store.num_live == 8
            assert store.deleted_ids() == {0, 5}
            store.mark_live(0)
            assert store.num_deleted == 1

    def test_rows_past_the_high_water_mark_read_as_deleted(self, tmp_path):
        with VectorStore(tmp_path / "c", dim=4, capacity=16) as store:
            assert store.is_deleted(9999)

    def test_write_to_an_arbitrary_row_extends_the_store(self, tmp_path, rng):
        """WAL replay writes rows by absolute index, not by appending."""
        with VectorStore(tmp_path / "c", dim=4, capacity=2) as store:
            v = rng.normal(size=4).astype(np.float32)
            store.write(20, v)
            assert store.num_rows == 21
            np.testing.assert_array_equal(store.get(20), v)

    def test_compaction_renumbers_and_drops_tombstones(self, tmp_path, rng):
        x = rng.normal(size=(10, 4)).astype(np.float32)
        with VectorStore(tmp_path / "c", dim=4, capacity=16) as store:
            store.append_batch(x)
            for row in (1, 3, 5):
                store.mark_deleted(row)
            keep = [r for r in range(10) if r not in (1, 3, 5)]
            mapping = store.compact(keep)

            assert store.num_rows == 7
            assert store.num_deleted == 0
            assert mapping == {old: new for new, old in enumerate(keep)}
            for old, new in mapping.items():
                np.testing.assert_array_equal(store.get(new), x[old])

    def test_disk_and_memory_accounting(self, tmp_path, rng):
        with VectorStore(tmp_path / "c", dim=8, capacity=32) as store:
            store.append_batch(rng.normal(size=(10, 8)).astype(np.float32))
            assert store.memory_bytes == 10 * 8 * 4
            assert store.disk_bytes >= 32 * 8 * 4

    def test_implements_the_vector_source_protocol(self, tmp_path, rng):
        from pyvec.core.types import VectorSource

        with VectorStore(tmp_path / "c", dim=4) as store:
            store.append(rng.normal(size=4).astype(np.float32))
            assert isinstance(store, VectorSource)


class TestWAL:
    def test_append_and_replay_round_trip(self, tmp_path, rng):
        vec = rng.normal(size=4).astype(np.float32)
        with WAL(tmp_path / "wal.log") as wal:
            wal.append_insert(0, "doc-1", {"a": 1}, vec)
            wal.append_delete(0, "doc-1")

        entries = list(WAL(tmp_path / "wal.log").replay())
        assert [e.op for e in entries] == [OpType.INSERT, OpType.DELETE]
        internal, ext, meta, restored = entries[0].as_insert(4)
        assert (internal, ext, meta) == (0, "doc-1", {"a": 1})
        np.testing.assert_array_equal(restored, vec)
        assert entries[1].as_json() == {"internal_id": 0, "id": "doc-1"}

    def test_vectors_survive_bit_exact(self, tmp_path, rng):
        """The payload carries raw float32 bytes, so there is no rounding."""
        vec = rng.normal(size=128).astype(np.float32)
        wal = WAL(tmp_path / "wal.log")
        wal.append_insert(3, "x", {}, vec)
        wal.close()
        _, _, _, restored = next(WAL(tmp_path / "wal.log").replay()).as_insert(128)
        np.testing.assert_array_equal(restored, vec)

    def test_entries_replay_in_write_order(self, tmp_path, rng):
        wal = WAL(tmp_path / "wal.log")
        for i in range(20):
            wal.append_insert(i, f"doc-{i}", {}, rng.normal(size=4).astype(np.float32))
        wal.close()
        ids = [e.as_insert(4)[0] for e in WAL(tmp_path / "wal.log").replay()]
        assert ids == list(range(20))

    def test_timestamps_are_monotonic(self, tmp_path, rng):
        wal = WAL(tmp_path / "wal.log")
        for i in range(5):
            wal.append_insert(i, str(i), {}, np.zeros(4, dtype=np.float32))
        wal.close()
        stamps = [e.timestamp_ns for e in WAL(tmp_path / "wal.log").replay()]
        assert stamps == sorted(stamps)

    def test_truncate_empties_the_log(self, tmp_path):
        wal = WAL(tmp_path / "wal.log")
        wal.append_create_index({"type": "hnsw"})
        assert wal.size_bytes > 0
        wal.truncate()
        assert wal.size_bytes == 0
        assert list(wal.replay()) == []
        wal.close()

    def test_appending_after_truncate_works(self, tmp_path):
        wal = WAL(tmp_path / "wal.log")
        wal.append_create_index({"a": 1})
        wal.truncate()
        wal.append_create_index({"b": 2})
        wal.close()
        assert [e.as_json() for e in WAL(tmp_path / "wal.log").replay()] == [{"b": 2}]

    def test_replaying_a_missing_file_yields_nothing(self, tmp_path):
        assert list(WAL(tmp_path / "absent.log", enabled=False).replay()) == []

    def test_disabled_wal_writes_nothing(self, tmp_path):
        wal = WAL(tmp_path / "wal.log", enabled=False)
        wal.append_create_index({"a": 1})
        wal.close()
        assert not (tmp_path / "wal.log").exists()

    @pytest.mark.parametrize("policy", list(FsyncPolicy))
    def test_every_policy_produces_a_readable_log(self, tmp_path, policy, rng):
        wal = WAL(tmp_path / f"{policy.value}.log", fsync_policy=policy)
        for i in range(5):
            wal.append_insert(i, str(i), {}, rng.normal(size=4).astype(np.float32))
        wal.sync()
        wal.close()
        assert len(list(WAL(tmp_path / f"{policy.value}.log").replay())) == 5

    def test_stats(self, tmp_path):
        wal = WAL(tmp_path / "wal.log", fsync_policy=FsyncPolicy.BATCH)
        wal.append_create_index({"a": 1})
        stats = wal.stats()
        assert stats["enabled"] and stats["fsync_policy"] == "batch"
        assert stats["entries_written"] == 1
        wal.close()


class TestWALCorruption:
    """The reason the format carries a CRC — PRD NF4."""

    def _write_entries(self, path, n=3):
        wal = WAL(path)
        for i in range(n):
            wal.append_insert(i, f"doc-{i}", {"i": i}, np.full(4, i, dtype=np.float32))
        wal.close()
        return path.stat().st_size

    def test_a_truncated_tail_entry_is_discarded(self, tmp_path):
        """Exactly what a kill -9 mid-write leaves behind."""
        path = tmp_path / "wal.log"
        size = self._write_entries(path, 3)
        with open(path, "r+b") as f:
            f.truncate(size - 7)  # chop into the middle of the last entry

        entries = list(WAL(path).replay())
        assert len(entries) == 2, "must stop at the torn entry, not guess"
        assert [e.as_insert(4)[0] for e in entries] == [0, 1]

    def test_truncate_torn_tail_removes_the_whole_partial_entry(self, tmp_path):
        """Not just the missing bytes — the entire incomplete entry goes.

        A half-written entry is unusable, so the recovered log ends at the last
        entry that verifies. Leaving the remnant in place would make the next
        append produce a log with garbage wedged in the middle of it.
        """
        path = tmp_path / "wal.log"
        size = self._write_entries(path, 3)
        with open(path, "r+b") as f:
            f.truncate(size - 7)
        damaged_size = path.stat().st_size

        wal = WAL(path)
        good_prefix = wal.valid_length()
        removed = wal.truncate_torn_tail()

        assert removed == damaged_size - good_prefix
        assert removed > 7, "the partial entry's surviving bytes go too"
        assert path.stat().st_size == good_prefix
        assert wal.valid_length() == path.stat().st_size
        assert wal.truncate_torn_tail() == 0  # idempotent
        assert len(list(wal.replay())) == 2
        wal.close()

    def test_appending_after_tail_repair_produces_a_valid_log(self, tmp_path):
        path = tmp_path / "wal.log"
        size = self._write_entries(path, 3)
        with open(path, "r+b") as f:
            f.truncate(size - 7)

        wal = WAL(path)
        wal.truncate_torn_tail()
        wal.append_insert(9, "doc-9", {}, np.full(4, 9, dtype=np.float32))
        wal.close()

        ids = [e.as_insert(4)[0] for e in WAL(path).replay()]
        assert ids == [0, 1, 9]

    def test_a_flipped_bit_stops_replay(self, tmp_path):
        """Without the CRC this entry would be replayed as garbage."""
        path = tmp_path / "wal.log"
        self._write_entries(path, 3)
        data = bytearray(path.read_bytes())
        # Corrupt a payload byte inside the second entry.
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
        assert len(list(WAL(path).replay())) < 3

    def test_a_corrupt_length_field_does_not_allocate_wildly(self, tmp_path):
        path = tmp_path / "wal.log"
        header = struct.pack("<BQI", int(OpType.INSERT), 0, 2**31)
        path.write_bytes(header + b"\x00" * 8)
        assert list(WAL(path).replay()) == []

    def test_an_unknown_op_code_stops_replay(self, tmp_path):
        path = tmp_path / "wal.log"
        payload = b"{}"
        header = struct.pack("<BQI", 99, 0, len(payload))
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        path.write_bytes(header + payload + struct.pack("<I", crc))
        assert list(WAL(path).replay()) == []

    def test_trailing_garbage_is_ignored(self, tmp_path):
        path = tmp_path / "wal.log"
        self._write_entries(path, 2)
        with open(path, "ab") as f:
            f.write(b"\x00\x01\x02")
        assert len(list(WAL(path).replay())) == 2

    def test_an_empty_file_replays_cleanly(self, tmp_path):
        path = tmp_path / "wal.log"
        path.write_bytes(b"")
        assert list(WAL(path).replay()) == []

    def test_dimension_mismatch_on_replay_is_reported(self, tmp_path):
        path = tmp_path / "wal.log"
        wal = WAL(path)
        wal.append_insert(0, "x", {}, np.zeros(8, dtype=np.float32))
        wal.close()
        with pytest.raises(CorruptDataError, match="dims"):
            next(WAL(path).replay()).as_insert(4)


class TestRWLock:
    def test_multiple_readers_run_concurrently(self):
        lock = RWLock()
        started = threading.Barrier(3, timeout=5)

        def reader():
            with lock.read():
                started.wait()  # deadlocks unless all three hold it at once

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

    def test_a_writer_excludes_readers(self):
        lock = RWLock()
        order: list[str] = []
        release = threading.Event()

        def writer():
            with lock.write():
                order.append("writer-in")
                release.wait(timeout=5)
                order.append("writer-out")

        def reader():
            with lock.read():
                order.append("reader")

        w = threading.Thread(target=writer)
        w.start()
        while not lock.write_locked:
            time.sleep(0.001)
        r = threading.Thread(target=reader)
        r.start()
        time.sleep(0.05)
        assert order == ["writer-in"], "reader entered during a write"
        release.set()
        w.join(timeout=5)
        r.join(timeout=5)
        assert order == ["writer-in", "writer-out", "reader"]

    def test_writers_are_mutually_exclusive(self):
        lock = RWLock()
        concurrent = 0
        peak = 0
        guard = threading.Lock()

        def writer():
            nonlocal concurrent, peak
            for _ in range(50):
                with lock.write():
                    with guard:
                        concurrent += 1
                        peak = max(peak, concurrent)
                    time.sleep(0.0002)
                    with guard:
                        concurrent -= 1

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert peak == 1, f"{peak} writers held the lock at once"

    def test_a_waiting_writer_blocks_new_readers(self):
        """Writer-preferring: otherwise sustained reads starve inserts forever."""
        lock = RWLock()
        lock.acquire_read()
        writer_waiting = threading.Event()
        reader_acquired = threading.Event()

        def writer():
            writer_waiting.set()
            with lock.write():
                pass

        def late_reader():
            with lock.read():
                reader_acquired.set()

        w = threading.Thread(target=writer)
        w.start()
        writer_waiting.wait(timeout=5)
        time.sleep(0.05)  # let the writer actually enqueue
        r = threading.Thread(target=late_reader)
        r.start()
        time.sleep(0.05)
        assert not reader_acquired.is_set(), "reader jumped a waiting writer"

        lock.release_read()
        w.join(timeout=5)
        r.join(timeout=5)
        assert reader_acquired.is_set()

    def test_reader_count_is_tracked(self):
        lock = RWLock()
        assert lock.readers == 0
        lock.acquire_read()
        lock.acquire_read()
        assert lock.readers == 2
        lock.release_read()
        lock.release_read()
        assert lock.readers == 0

    def test_the_lock_is_released_when_the_body_raises(self):
        lock = RWLock()
        with pytest.raises(RuntimeError):
            with lock.write():
                raise RuntimeError("boom")
        assert not lock.write_locked
        with lock.write():  # must not deadlock
            pass
