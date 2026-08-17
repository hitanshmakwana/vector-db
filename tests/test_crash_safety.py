"""Crash safety — PRD NF4: ``kill -9`` during insert must leave the collection
consistent on next open.

Three layers of test, because each catches something the others cannot:

1. **Real process kills.** A child process inserts and is killed from outside
   with no chance to run cleanup. This is the only way to test that we never
   depended on an orderly shutdown — ``atexit``, ``__del__``, buffered writes
   and context managers all fail to run.
2. **Torn and corrupt WAL tails.** Byte-level damage, applied deliberately, so
   the recovery path is exercised at exactly the boundaries a real crash
   produces.
3. **Interrupted checkpoints.** State killed *between* the snapshot files and
   the WAL truncation — the window that makes idempotent replay necessary.

The guarantee being tested is not "no data loss ever". It is: **whatever the
client was told is durable stays durable, and the collection always opens into a
consistent state.** An insert that never returned its 201 may or may not survive;
one that did must.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pyvec.core.collection import MANIFEST_FILE, Collection
from pyvec.storage.wal import WAL, FsyncPolicy, OpType

DIM = 8
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Layer 1: real process kills
# --------------------------------------------------------------------------- #

CHILD_SCRIPT = textwrap.dedent(
    """
    import sys, numpy as np
    sys.path.insert(0, {repo!r})
    from pyvec.core.collection import Collection

    root, count = {root!r}, {count}
    c = Collection.create("docs", root, dimension={dim}, metric="l2",
                          index_type="flat", text_field="content")
    rng = np.random.default_rng(0)
    for i in range(count):
        c.insert([{{"id": f"d{{i}}",
                    "vector": rng.normal(size={dim}).astype("float32").tolist(),
                    "metadata": {{"content": f"document number {{i}}", "n": i}}}}])
        # Announce durability only *after* insert() returns, so the parent knows
        # exactly which ids were acknowledged.
        print(i, flush=True)
    print("DONE", flush=True)
    # Deliberately no close(): the parent kills us here, so nothing is
    # checkpointed and recovery has to come entirely from the WAL.
    import time
    time.sleep(60)
    """
)


def _spawn_inserter(root: Path, count: int) -> subprocess.Popen:
    script = CHILD_SCRIPT.format(
        repo=str(REPO_ROOT), root=str(root), count=count, dim=DIM
    )
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _read_acknowledged(proc: subprocess.Popen, until: int) -> int:
    """Block until the child acknowledges ``until`` inserts. Returns the count."""
    acked = 0
    assert proc.stdout is not None
    while acked < until:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"child died early: {stderr}")
        if line.strip() == "DONE":
            break
        acked = int(line.strip()) + 1
    return acked


@pytest.mark.parametrize("kill_after", [1, 7, 25])
def test_hard_kill_mid_insert_recovers_acknowledged_writes(data_root, kill_after):
    """The headline guarantee, with a genuinely uncatchable kill.

    ``Popen.kill()`` is ``SIGKILL`` on POSIX and ``TerminateProcess`` on Windows;
    neither can be intercepted, so no cleanup code runs in the child.
    """
    proc = _spawn_inserter(data_root, count=200)
    try:
        acked = _read_acknowledged(proc, kill_after)
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()

    assert acked >= kill_after

    reopened = Collection.open(data_root / "docs")
    try:
        # Every acknowledged insert must be present and queryable.
        for i in range(acked):
            assert reopened.contains(f"d{i}"), f"lost acknowledged insert d{i}"
            assert reopened.get(f"d{i}")["metadata"]["n"] == i

        # The collection must be internally consistent, not merely non-empty.
        assert len(reopened) >= acked
        assert len(reopened._id_map) == len(reopened._rev_map)
        for ext, internal in reopened._id_map.items():
            assert reopened._rev_map[internal] == ext
        assert reopened._next_id >= len(reopened)

        # And it must still work as a database.
        hits = reopened.search(reopened.get("d0")["vector"], k=1)
        assert hits and hits[0].id == "d0"
        assert reopened.search_text("document number", k=5)
    finally:
        reopened.close()


def test_collection_is_usable_after_recovery(data_root):
    """Recovery has to leave a *writable* collection, not a read-only husk."""
    proc = _spawn_inserter(data_root, count=200)
    try:
        acked = _read_acknowledged(proc, 10)
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()

    reopened = Collection.open(data_root / "docs")
    try:
        reopened.insert([{"id": "after-crash", "vector": [1.0] * DIM}])
        assert reopened.contains("after-crash")
        reopened.delete("d0")
        assert not reopened.contains("d0")
        reopened.checkpoint()
        assert reopened.wal.size_bytes == 0
    finally:
        reopened.close()

    # And it survives one more round trip.
    again = Collection.open(data_root / "docs")
    try:
        assert again.contains("after-crash")
        assert not again.contains("d0")
        assert again.contains(f"d{acked - 1}")
    finally:
        again.close()


def test_repeated_crashes_do_not_compound_damage(data_root):
    """Recovery must be idempotent across several successive hard kills."""
    for round_no in range(3):
        proc = _spawn_inserter(data_root / f"r{round_no}", count=50)
        try:
            acked = _read_acknowledged(proc, 5)
            proc.kill()
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
            for stream in (proc.stdout, proc.stderr):
                if stream:
                    stream.close()

        for attempt in range(2):
            c = Collection.open(data_root / f"r{round_no}" / "docs")
            try:
                assert len(c) >= acked, f"round {round_no} attempt {attempt}"
                assert c.contains(f"d{acked - 1}")
            finally:
                c.close()


# --------------------------------------------------------------------------- #
# Layer 2: byte-level WAL damage
# --------------------------------------------------------------------------- #


def _uncheckpointed_collection(root: Path, n: int) -> tuple[Path, list[dict]]:
    """Build a collection whose data lives only in the WAL, not the snapshot."""
    rng = np.random.default_rng(1)
    items = [
        {
            "id": f"d{i}",
            "vector": rng.normal(size=DIM).astype(np.float32).tolist(),
            "metadata": {"content": f"doc {i}", "n": i},
        }
        for i in range(n)
    ]
    c = Collection.create(
        "docs", root, dimension=DIM, metric="l2", index_type="flat",
        text_field="content",
    )
    c.insert(items)
    # Close the handles without checkpointing, exactly as a crash would leave it.
    c.wal.close()
    c.store.close()
    c._closed = True
    return root / "docs", items


class TestTornWAL:
    def test_a_torn_tail_entry_is_discarded_and_the_rest_survives(self, data_root):
        path, items = _uncheckpointed_collection(data_root, 20)
        wal_path = path / "wal.log"
        original = wal_path.stat().st_size
        with open(wal_path, "r+b") as f:
            f.truncate(original - 9)  # mid-entry: a partial write

        c = Collection.open(path)
        try:
            assert c.recovery_report["wal_bytes_discarded"] > 0
            # 19 of 20 definitely survive; the torn one is the only casualty.
            assert len(c) >= 19
            for i in range(19):
                assert c.contains(f"d{i}")
            assert c.search(items[0]["vector"], k=1)[0].id == "d0"
        finally:
            c.close()

    def test_a_flipped_bit_stops_replay_without_corrupting_state(self, data_root):
        """The CRC's whole purpose: never replay a damaged entry as if it were
        real. Without it, a corrupted vector or id would be written into the
        store during the recovery meant to protect it."""
        path, _ = _uncheckpointed_collection(data_root, 20)
        wal_path = path / "wal.log"
        data = bytearray(wal_path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        wal_path.write_bytes(bytes(data))

        c = Collection.open(path)
        try:
            assert len(c) < 20, "replay should have stopped at the damaged entry"
            # Whatever survived must be coherent.
            for ext, internal in c._id_map.items():
                assert c._rev_map[internal] == ext
                if internal not in c._deleted:
                    assert c.get(ext)["id"] == ext
            c.insert([{"id": "fresh", "vector": [1.0] * DIM}])
        finally:
            c.close()

    def test_trailing_garbage_is_ignored(self, data_root):
        path, _ = _uncheckpointed_collection(data_root, 10)
        with open(path / "wal.log", "ab") as f:
            f.write(b"\xde\xad\xbe\xef" * 4)

        c = Collection.open(path)
        try:
            assert len(c) == 10
        finally:
            c.close()

    def test_a_zero_length_wal_recovers_the_snapshot_only(self, data_root):
        items = [{"id": f"d{i}", "vector": [float(i)] * DIM} for i in range(5)]
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(items)
        c.checkpoint()  # everything is in the snapshot now
        c.close()
        (data_root / "docs" / "wal.log").write_bytes(b"")

        reopened = Collection.open(data_root / "docs")
        try:
            assert len(reopened) == 5
        finally:
            reopened.close()

    def test_a_wal_containing_only_garbage_leaves_the_snapshot_intact(self, data_root):
        items = [{"id": f"d{i}", "vector": [float(i)] * DIM} for i in range(5)]
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert(items)
        c.checkpoint()
        c.close()
        (data_root / "docs" / "wal.log").write_bytes(os.urandom(64))

        reopened = Collection.open(data_root / "docs")
        try:
            assert len(reopened) == 5
            assert reopened.contains("d3")
        finally:
            reopened.close()


# --------------------------------------------------------------------------- #
# Layer 3: interrupted checkpoints
# --------------------------------------------------------------------------- #


class TestInterruptedCheckpoint:
    def test_a_crash_between_snapshot_and_wal_truncation_replays_harmlessly(
        self, data_root
    ):
        """The window that makes idempotent replay a requirement.

        Checkpoint order is: flush, metadata, index, manifest, *then* truncate the
        WAL. Dying before that last step leaves entries in the log that are
        already in the snapshot. Replaying them must be a no-op, not a duplicate.
        """
        items = [
            {"id": f"d{i}", "vector": [float(i)] * DIM, "metadata": {"n": i}}
            for i in range(20)
        ]
        c = Collection.create(
            "docs", data_root, dimension=DIM, metric="l2", index_type="flat"
        )
        c.insert(items)
        wal_bytes = (data_root / "docs" / "wal.log").read_bytes()

        # Everything up to (but not including) the truncate.
        c.store.flush()
        c._write_metadata()
        c.dense.save(data_root / "docs" / "dense.idx")
        c._write_manifest()
        c.wal.close()
        c.store.close()
        c._closed = True
        # Put the un-truncated log back, simulating the crash.
        (data_root / "docs" / "wal.log").write_bytes(wal_bytes)

        reopened = Collection.open(data_root / "docs")
        try:
            assert len(reopened) == 20, "replay duplicated or dropped rows"
            assert reopened.recovery_report["skipped"] == 20, (
                "entries already in the snapshot should be recognised and skipped"
            )
            assert len(reopened._id_map) == 20
            assert len(set(reopened._id_map.values())) == 20
            for i in range(20):
                assert reopened.get(f"d{i}")["metadata"]["n"] == i
        finally:
            reopened.close()

    def test_recovery_folds_replayed_entries_into_a_fresh_checkpoint(self, data_root):
        """After replaying, the log should be truncated so a second crash has
        nothing left to redo."""
        path, _ = _uncheckpointed_collection(data_root, 15)
        assert (path / "wal.log").stat().st_size > 0

        c = Collection.open(path)
        try:
            assert c.recovery_report["insert"] > 0
            assert c.wal.size_bytes == 0, "recovery should checkpoint what it replayed"
        finally:
            c.close()

    def test_a_stale_temp_file_does_not_break_open(self, data_root):
        """Atomic writes go temp-then-rename; a crash can leave the temp behind."""
        c = Collection.create("docs", data_root, dimension=DIM, index_type="flat")
        c.insert([{"id": "a", "vector": [1.0] * DIM}])
        c.close()
        (data_root / "docs" / f"{MANIFEST_FILE}.tmp").write_text("{ garbage")

        reopened = Collection.open(data_root / "docs")
        try:
            assert reopened.contains("a")
        finally:
            reopened.close()

    def test_deletes_survive_a_crash(self, data_root):
        items = [{"id": f"d{i}", "vector": [float(i)] * DIM} for i in range(10)]
        c = Collection.create(
            "docs", data_root, dimension=DIM, metric="l2", index_type="flat"
        )
        c.insert(items)
        c.checkpoint()
        c.delete("d4")  # logged, not yet checkpointed
        c.wal.close()
        c.store.close()
        c._closed = True

        reopened = Collection.open(data_root / "docs")
        try:
            assert not reopened.contains("d4")
            assert len(reopened) == 9
            assert reopened.recovery_report["delete"] == 1
        finally:
            reopened.close()


# --------------------------------------------------------------------------- #
# Durability policy
# --------------------------------------------------------------------------- #


class TestFsyncPolicy:
    def test_entry_policy_syncs_every_write(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat",
            fsync_policy=FsyncPolicy.ENTRY,
        )
        try:
            c.insert([{"id": "a", "vector": [1.0] * DIM}])
            assert c.wal.stats()["unsynced_entries"] == 0
        finally:
            c.close()

    def test_batch_policy_leaves_entries_unsynced_until_asked(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat",
            fsync_policy=FsyncPolicy.BATCH,
        )
        try:
            # create() logs a CREATE_INDEX entry, which is also still unsynced.
            before = c.wal.stats()["unsynced_entries"]
            c.insert([{"id": f"d{i}", "vector": [float(i)] * DIM} for i in range(5)])
            assert c.wal.stats()["unsynced_entries"] == before + 5
            c.wal.sync()
            assert c.wal.stats()["unsynced_entries"] == 0
        finally:
            c.close()

    def test_the_policy_is_persisted(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat",
            fsync_policy=FsyncPolicy.BATCH,
        )
        c.close()
        manifest = json.loads((data_root / "docs" / MANIFEST_FILE).read_text())
        assert manifest["fsync_policy"] == "batch"
        reopened = Collection.open(data_root / "docs")
        try:
            assert reopened.wal.fsync_policy is FsyncPolicy.BATCH
        finally:
            reopened.close()

    def test_wal_can_be_disabled_for_benchmarking(self, data_root):
        """BENCHMARKS.md benchmark 4 config 3: unsafe mode, for measuring the
        cost of durability. Data is then only as safe as the next checkpoint."""
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="flat", wal_enabled=False
        )
        try:
            c.insert([{"id": "a", "vector": [1.0] * DIM}])
            assert not (data_root / "docs" / "wal.log").exists()
            c.checkpoint()
        finally:
            c.close()

        reopened = Collection.open(data_root / "docs")
        try:
            assert reopened.contains("a")  # survived via the checkpoint
        finally:
            reopened.close()


class TestWALContents:
    def test_the_log_records_the_index_configuration(self, data_root):
        """ARCHITECTURE.md op type CREATE_INDEX."""
        c = Collection.create(
            "docs", data_root, dimension=DIM, index_type="hnsw",
            index_params={"M": 8},
        )
        try:
            entries = list(WAL(data_root / "docs" / "wal.log").replay())
            create = [e for e in entries if e.op is OpType.CREATE_INDEX]
            assert create, "index config should be logged"
            payload = create[0].as_json()
            assert payload["index_type"] == "hnsw"
            assert payload["params"] == {"M": 8}
            assert payload["dimension"] == DIM
        finally:
            c.close()

    def test_insert_entries_carry_the_full_record(self, data_root):
        c = Collection.create(
            "docs", data_root, dimension=DIM, metric="l2", index_type="flat"
        )
        try:
            vec = [0.5] * DIM
            c.insert([{"id": "a", "vector": vec, "metadata": {"k": "v"}}])
            inserts = [
                e for e in WAL(data_root / "docs" / "wal.log").replay()
                if e.op is OpType.INSERT
            ]
            assert len(inserts) == 1
            internal, ext, metadata, restored = inserts[0].as_insert(DIM)
            assert (internal, ext, metadata) == (0, "a", {"k": "v"})
            np.testing.assert_allclose(restored, vec)
        finally:
            c.close()
