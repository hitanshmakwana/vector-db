"""Append-only write-ahead log.

The contract (ADR-005, and the interview answer in RESUME.md): the log records
the *intent* to change state before the state changes, so any crash leaves a
recoverable trail. Insert path::

    receive insert -> append WAL entry -> fsync WAL -> write vector to mmap
                   -> return 201

* Crash after the fsync but before the mmap write: the entry is on disk, replay
  applies it. No loss.
* Crash before the fsync: the client never received a 201, so there is nothing
  to be consistent with. No violation.

That is the whole guarantee, and it is what PRD NF4 asks for: ``kill -9`` during
insert leaves the collection consistent on next open.

Entry layout, per ARCHITECTURE.md §6 plus one addition::

    op_type      u8
    timestamp    u64   nanoseconds since epoch
    payload_len  u32
    payload      bytes
    crc32        u32   <-- addition

**Why the CRC.** The documented header alone cannot distinguish "the log ends
here" from "the last write was torn". A ``kill -9`` mid-``write`` can leave a
partial entry, and a partial entry whose length field happens to survive would
be replayed as garbage — silently corrupting the collection during the very
recovery meant to protect it. With a checksum, replay stops at the first entry
that fails to verify, treats it as the tail, and truncates. This is the
difference between "we have a WAL" and "we have a WAL that works", so it is
worth the four bytes.

Payload encoding is per-op and lives in this module; the log itself treats
payloads as opaque bytes. INSERT carries raw float32 vector bytes after a JSON
header rather than JSON-encoded floats — a 768-dim vector is ~3KB of float32
against ~15KB of JSON text, and the parse cost on replay is the difference
between a memcpy and 768 ``float()`` calls.
"""

from __future__ import annotations

import json
import os
import struct
import time
import zlib
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from pyvec.core.types import VECTOR_DTYPE, InternalId

__all__ = ["WAL", "WALEntry", "OpType", "FsyncPolicy", "WAL_FILE"]

WAL_FILE = "wal.log"

_HEADER = struct.Struct("<BQI")  # op_type, timestamp_ns, payload_len
_CRC = struct.Struct("<I")

#: Refuse to allocate a payload buffer larger than this during replay. A corrupt
#: length field could otherwise ask for gigabytes before the CRC check runs.
_MAX_PAYLOAD = 64 * 1024 * 1024


class OpType(IntEnum):
    """Op codes. ARCHITECTURE.md lists INSERT, DELETE, CREATE_INDEX."""

    INSERT = 1
    DELETE = 2
    CREATE_INDEX = 3
    #: Marks the point a checkpoint captured. Everything before it is already in
    #: the snapshot; recovery replays only what follows.
    CHECKPOINT = 4


class FsyncPolicy(str, Enum):
    """When to force the log to stable storage.

    BENCHMARKS.md benchmark 4 measures exactly this trade-off, so it has to be a
    runtime choice rather than a constant.
    """

    #: fsync every entry. Durable to the last acknowledged write. The default —
    #: "durability > throughput for a learning project" (ARCHITECTURE.md §6).
    ENTRY = "entry"
    #: fsync once per batch/flush call. Group commit, roughly what production
    #: systems do. A crash can lose the tail of the last batch.
    BATCH = "batch"
    #: Never fsync explicitly; rely on the OS. Fast and unsafe. For benchmarking.
    NEVER = "never"


@dataclass(slots=True)
class WALEntry:
    op: OpType
    timestamp_ns: int
    payload: bytes

    # -- typed payload accessors ------------------------------------------ #

    def as_insert(self, dim: int) -> tuple[InternalId, str, dict, np.ndarray]:
        """Decode an INSERT payload into ``(internal_id, id, metadata, vector)``."""
        (json_len,) = struct.unpack_from("<I", self.payload, 0)
        head = json.loads(self.payload[4 : 4 + json_len].decode("utf-8"))
        raw = self.payload[4 + json_len :]
        vector = np.frombuffer(raw, dtype=VECTOR_DTYPE)
        if vector.shape[0] != dim:
            from pyvec.core.errors import CorruptDataError

            raise CorruptDataError(
                f"WAL insert entry has {vector.shape[0]} dims, expected {dim}"
            )
        return (
            int(head["internal_id"]),
            str(head["id"]),
            head.get("metadata") or {},
            vector,
        )

    def as_json(self) -> Any:
        return json.loads(self.payload.decode("utf-8"))


def encode_insert(
    internal_id: InternalId,
    external_id: str,
    metadata: dict | None,
    vector: np.ndarray,
) -> bytes:
    """``[json_len u32][json header][float32 vector bytes]``."""
    head = json.dumps(
        {
            "internal_id": int(internal_id),
            "id": external_id,
            "metadata": metadata or {},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    vec = np.ascontiguousarray(vector, dtype=VECTOR_DTYPE).reshape(-1)
    return struct.pack("<I", len(head)) + head + vec.tobytes()


def encode_json(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class WAL:
    """Append-only log with configurable fsync policy and torn-tail recovery."""

    def __init__(
        self,
        path: Path,
        fsync_policy: FsyncPolicy | str = FsyncPolicy.ENTRY,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.fsync_policy = FsyncPolicy(fsync_policy)
        self.enabled = bool(enabled)
        self._f = None
        self._bytes_written = 0
        self._entries_written = 0
        self._unsynced = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._open()

    def _open(self) -> None:
        self._f = open(self.path, "ab", buffering=0)

    # ------------------------------------------------------------------ #
    # Append
    # ------------------------------------------------------------------ #

    def append(self, op: OpType, payload: bytes) -> None:
        """Write one entry, honouring the fsync policy."""
        if not self.enabled:
            return
        assert self._f is not None
        header = _HEADER.pack(int(op), time.time_ns(), len(payload))
        # CRC covers header and payload together, so a corrupted length field is
        # caught rather than trusted.
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        # A single write() call keeps the entry as atomic as the OS allows;
        # splitting it into three would widen the torn-write window for no gain.
        self._f.write(header + payload + _CRC.pack(crc))
        self._bytes_written += _HEADER.size + len(payload) + _CRC.size
        self._entries_written += 1
        self._unsynced += 1
        if self.fsync_policy is FsyncPolicy.ENTRY:
            self.sync()

    def append_insert(
        self,
        internal_id: InternalId,
        external_id: str,
        metadata: dict | None,
        vector: np.ndarray,
    ) -> None:
        self.append(
            OpType.INSERT, encode_insert(internal_id, external_id, metadata, vector)
        )

    def append_delete(self, internal_id: InternalId, external_id: str) -> None:
        self.append(
            OpType.DELETE,
            encode_json({"internal_id": int(internal_id), "id": external_id}),
        )

    def append_create_index(self, config: dict) -> None:
        self.append(OpType.CREATE_INDEX, encode_json(config))

    def append_checkpoint(self, state: dict) -> None:
        self.append(OpType.CHECKPOINT, encode_json(state))

    def sync(self) -> None:
        """Force buffered entries to stable storage."""
        if not self.enabled or self._f is None:
            return
        if self.fsync_policy is FsyncPolicy.NEVER:
            self._unsynced = 0
            return
        self._f.flush()
        os.fsync(self._f.fileno())
        self._unsynced = 0

    # ------------------------------------------------------------------ #
    # Replay
    # ------------------------------------------------------------------ #

    def replay(self) -> Iterator[WALEntry]:
        """Yield every intact entry in order, stopping at the first bad one.

        A trailing partial or corrupt entry is treated as the end of the log —
        that write never completed, so the client never got an acknowledgement
        for it. :meth:`truncate_torn_tail` removes it.
        """
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            while True:
                header = f.read(_HEADER.size)
                if len(header) < _HEADER.size:
                    break  # clean EOF or torn header
                op_raw, ts, length = _HEADER.unpack(header)
                if length > _MAX_PAYLOAD:
                    break
                payload = f.read(length)
                if len(payload) < length:
                    break  # torn payload
                crc_raw = f.read(_CRC.size)
                if len(crc_raw) < _CRC.size:
                    break  # torn checksum
                (crc,) = _CRC.unpack(crc_raw)
                if zlib.crc32(header + payload) & 0xFFFFFFFF != crc:
                    break  # corrupt entry: stop, do not guess
                try:
                    op = OpType(op_raw)
                except ValueError:
                    break  # unknown op code from a newer/garbled writer
                yield WALEntry(op=op, timestamp_ns=ts, payload=payload)

    def valid_length(self) -> int:
        """Byte offset just past the last intact entry."""
        if not self.path.exists():
            return 0
        offset = 0
        with open(self.path, "rb") as f:
            while True:
                header = f.read(_HEADER.size)
                if len(header) < _HEADER.size:
                    break
                _op, _ts, length = _HEADER.unpack(header)
                if length > _MAX_PAYLOAD:
                    break
                payload = f.read(length)
                if len(payload) < length:
                    break
                crc_raw = f.read(_CRC.size)
                if len(crc_raw) < _CRC.size:
                    break
                (crc,) = _CRC.unpack(crc_raw)
                if zlib.crc32(header + payload) & 0xFFFFFFFF != crc:
                    break
                offset += _HEADER.size + length + _CRC.size
        return offset

    def truncate_torn_tail(self) -> int:
        """Cut a partial trailing entry. Returns bytes removed."""
        if not self.path.exists():
            return 0
        good = self.valid_length()
        size = self.path.stat().st_size
        if good == size:
            return 0
        was_open = self._f is not None
        if was_open:
            self.close()
        with open(self.path, "r+b") as f:
            f.truncate(good)
            f.flush()
            os.fsync(f.fileno())
        if was_open:
            self._open()
        return size - good

    # ------------------------------------------------------------------ #
    # Checkpoint / truncation
    # ------------------------------------------------------------------ #

    def truncate(self) -> None:
        """Empty the log. Called *after* a checkpoint has been made durable.

        Order matters: snapshot first, then truncate. Truncating first would
        create a window in which neither the log nor the snapshot has the data.
        """
        if not self.enabled:
            return
        if self._f is not None:
            self.close()
        with open(self.path, "wb") as f:
            f.truncate(0)
            f.flush()
            os.fsync(f.fileno())
        self._open()
        self._bytes_written = 0
        self._entries_written = 0
        self._unsynced = 0

    # ------------------------------------------------------------------ #
    # Lifecycle / stats
    # ------------------------------------------------------------------ #

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fsync_policy": self.fsync_policy.value,
            "size_bytes": self.size_bytes,
            "entries_written": self._entries_written,
            "unsynced_entries": self._unsynced,
        }

    def close(self) -> None:
        if self._f is not None:
            try:
                if self._unsynced and self.fsync_policy is not FsyncPolicy.NEVER:
                    self._f.flush()
                    os.fsync(self._f.fileno())
            finally:
                self._f.close()
                self._f = None

    def __enter__(self) -> "WAL":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
