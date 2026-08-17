"""Memory-mapped vector storage.

ADR-005: vectors are a big contiguous ``float32`` array, which is precisely the
shape ``mmap`` is good at. The OS maps ``vectors.bin`` into the process address
space and its page cache decides what stays resident. The payoff:

* **Zero deserialisation.** A row of an mmap-ed float32 array *is* a NumPy
  array. No parse step between disk and BLAS.
* **The OS does the caching.** ANN scans read either sequentially (IVF bucket
  scans) or scatter-gather (HNSW graph hops); the page cache handles both
  without us writing a buffer pool.
* **Collections larger than RAM still open.** They just page.

The costs, all real:

* **Fixed capacity.** Growing means allocating a new file and copying, so we
  double each time to amortise it (ADR-005).
* **Deletes are tombstones** (ADR-010). A flag byte per row marks it dead;
  nothing is compacted until ``optimize()``.
* **Windows quirks.** A memmap cannot be resized while mapped and an open map
  blocks deletion of the underlying file, so growth here always goes
  copy-to-temp then atomic rename, with every map explicitly closed first.
  ADR-005 anticipated "limited Windows support"; this module pays the small
  cost of doing it properly instead.

Layout, per collection directory:

    vectors.bin   capacity * dim * 4 bytes, row i == internal id i
    flags.bin     capacity bytes, one status byte per row

``flags.bin`` is a byte per row rather than a packed bit per row. A true bitmap
would be 8x smaller, but at 1M vectors that is 1MB against the 512MB of vectors
it describes — and byte-per-row means a tombstone write is a single store with
no read-modify-write, so concurrent readers never see a torn byte.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from pyvec.core.errors import CorruptDataError
from pyvec.core.types import VECTOR_DTYPE, InternalId

__all__ = ["VectorStore", "FLAG_LIVE", "FLAG_DELETED"]

VECTORS_FILE = "vectors.bin"
FLAGS_FILE = "flags.bin"

FLAG_LIVE = 0
FLAG_DELETED = 1

#: Rows allocated for a brand-new collection when the caller names no capacity.
DEFAULT_CAPACITY = 1024

#: Rows copied per chunk when growing, so a 512MB grow does not need 512MB of
#: extra heap on top of the two maps.
_COPY_CHUNK = 65_536


class VectorStore:
    """Fixed-capacity, doubling, mmap-backed float32 matrix with tombstones.

    Implements the :class:`~pyvec.core.types.VectorSource` protocol, so indexes
    read vectors straight out of the map.
    """

    def __init__(self, path: Path, dim: int, capacity: int | None = None) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self.path.mkdir(parents=True, exist_ok=True)

        self._vectors_path = self.path / VECTORS_FILE
        self._flags_path = self.path / FLAGS_FILE
        self._vectors: np.memmap | None = None
        self._flags: np.memmap | None = None
        self._num_rows = 0  # high-water mark of allocated rows

        if self._vectors_path.exists():
            self._open_existing()
        else:
            self._create(int(capacity or DEFAULT_CAPACITY))

    # ------------------------------------------------------------------ #
    # Open / create
    # ------------------------------------------------------------------ #

    def _create(self, capacity: int) -> None:
        capacity = max(1, capacity)
        itemsize = np.dtype(VECTOR_DTYPE).itemsize
        # Preallocate by seeking to the end and writing one byte. On NTFS and
        # ext4 this creates a sparse file, so an oversized `capacity` costs
        # address space rather than disk until rows are actually written.
        with open(self._vectors_path, "wb") as f:
            f.truncate(capacity * self.dim * itemsize)
        with open(self._flags_path, "wb") as f:
            f.truncate(capacity)
        self._capacity = capacity
        self._map()

    def _open_existing(self) -> None:
        itemsize = np.dtype(VECTOR_DTYPE).itemsize
        size = self._vectors_path.stat().st_size
        row_bytes = self.dim * itemsize
        if row_bytes == 0 or size % row_bytes:
            raise CorruptDataError(
                f"{self._vectors_path}: size {size} is not a multiple of "
                f"{row_bytes} bytes/row — wrong dimension or truncated file"
            )
        self._capacity = size // row_bytes
        if not self._flags_path.exists():
            with open(self._flags_path, "wb") as f:
                f.truncate(self._capacity)
        elif self._flags_path.stat().st_size < self._capacity:
            # Flags file lagging the vector file means a crash between the two
            # truncations during a grow. Extending it is safe: the missing rows
            # were never written, so defaulting them to LIVE-but-unused matches
            # what `num_rows` says about them.
            with open(self._flags_path, "r+b") as f:
                f.truncate(self._capacity)
        self._map()

    def _map(self) -> None:
        self._vectors = np.memmap(
            self._vectors_path,
            dtype=VECTOR_DTYPE,
            mode="r+",
            shape=(self._capacity, self.dim),
        )
        self._flags = np.memmap(
            self._flags_path, dtype=np.uint8, mode="r+", shape=(self._capacity,)
        )

    def _unmap(self) -> None:
        """Drop both maps. Required before renaming files on Windows."""
        for attr in ("_vectors", "_flags"):
            m = getattr(self, attr)
            if m is not None:
                m.flush()
                # numpy exposes the underlying mmap through ._mmap; closing it
                # releases the file handle, which Windows needs before a rename.
                base = getattr(m, "_mmap", None)
                if base is not None:
                    base.close()
                setattr(self, attr, None)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_rows(self) -> int:
        """Rows handed out so far, including tombstoned ones."""
        return self._num_rows

    @num_rows.setter
    def num_rows(self, value: int) -> None:
        # The collection owns the id counter and restores it during recovery.
        self._num_rows = int(value)
        if self._num_rows > self._capacity:
            self.reserve(self._num_rows)

    @property
    def num_deleted(self) -> int:
        assert self._flags is not None
        return int(np.count_nonzero(self._flags[: self._num_rows] == FLAG_DELETED))

    @property
    def num_live(self) -> int:
        return self._num_rows - self.num_deleted

    @property
    def disk_bytes(self) -> int:
        total = 0
        for p in (self._vectors_path, self._flags_path):
            if p.exists():
                total += p.stat().st_size
        return total

    @property
    def memory_bytes(self) -> int:
        """Bytes of *live* vector data. What is resident is up to the OS."""
        return self._num_rows * self.dim * np.dtype(VECTOR_DTYPE).itemsize

    @property
    def array(self) -> np.memmap:
        """The raw map. Rows past ``num_rows`` are unallocated garbage."""
        assert self._vectors is not None
        return self._vectors

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #

    def reserve(self, needed: int) -> None:
        """Ensure capacity for at least ``needed`` rows, doubling as required."""
        if needed <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < needed:
            new_capacity *= 2
        self._grow(new_capacity)

    def _grow(self, new_capacity: int) -> None:
        """Allocate a larger file, copy, then atomically swap it in.

        Copy-and-rename rather than in-place truncate: if the process dies
        mid-grow, the original file is still whole and the temp file is garbage
        that gets cleaned up on the next open. Truncating a live mmap in place
        risks a half-extended file that looks valid but has an inconsistent row
        count.
        """
        itemsize = np.dtype(VECTOR_DTYPE).itemsize
        tmp_vectors = Path(str(self._vectors_path) + ".grow")
        tmp_flags = Path(str(self._flags_path) + ".grow")

        with open(tmp_vectors, "wb") as f:
            f.truncate(new_capacity * self.dim * itemsize)
        with open(tmp_flags, "wb") as f:
            f.truncate(new_capacity)

        dst = np.memmap(
            tmp_vectors,
            dtype=VECTOR_DTYPE,
            mode="r+",
            shape=(new_capacity, self.dim),
        )
        dst_flags = np.memmap(
            tmp_flags, dtype=np.uint8, mode="r+", shape=(new_capacity,)
        )
        assert self._vectors is not None and self._flags is not None
        for start in range(0, self._num_rows, _COPY_CHUNK):
            stop = min(start + _COPY_CHUNK, self._num_rows)
            dst[start:stop] = self._vectors[start:stop]
            dst_flags[start:stop] = self._flags[start:stop]
        dst.flush()
        dst_flags.flush()
        for m in (dst, dst_flags):
            base = getattr(m, "_mmap", None)
            if base is not None:
                base.close()

        self._unmap()
        os.replace(tmp_vectors, self._vectors_path)
        os.replace(tmp_flags, self._flags_path)
        self._capacity = new_capacity
        self._map()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def append(self, vector: np.ndarray) -> InternalId:
        """Write to the next free row and return its internal id."""
        row = self._num_rows
        self.write(row, vector)
        return row

    def append_batch(self, vectors: np.ndarray) -> list[InternalId]:
        """Write ``n`` rows contiguously. One map write, not ``n``."""
        vectors = np.ascontiguousarray(vectors, dtype=VECTOR_DTYPE)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            from pyvec.core.errors import InvalidDimensionError

            raise InvalidDimensionError(
                f"expected dimension {self.dim}, got {vectors.shape[1]}"
            )
        n = vectors.shape[0]
        start = self._num_rows
        self.reserve(start + n)
        assert self._vectors is not None and self._flags is not None
        self._vectors[start : start + n] = vectors
        self._flags[start : start + n] = FLAG_LIVE
        self._num_rows = start + n
        return list(range(start, start + n))

    def write(self, row: InternalId, vector: np.ndarray) -> None:
        """Write a specific row. Used by append and by WAL replay."""
        row = int(row)
        if row < 0:
            raise ValueError(f"row must be non-negative, got {row}")
        self.reserve(row + 1)
        assert self._vectors is not None and self._flags is not None
        v = np.ascontiguousarray(vector, dtype=VECTOR_DTYPE).reshape(-1)
        if v.shape[0] != self.dim:
            from pyvec.core.errors import InvalidDimensionError

            raise InvalidDimensionError(
                f"expected dimension {self.dim}, got {v.shape[0]}"
            )
        self._vectors[row] = v
        self._flags[row] = FLAG_LIVE
        if row >= self._num_rows:
            self._num_rows = row + 1

    def mark_deleted(self, row: InternalId) -> None:
        assert self._flags is not None
        row = int(row)
        if 0 <= row < self._capacity:
            self._flags[row] = FLAG_DELETED

    def mark_live(self, row: InternalId) -> None:
        assert self._flags is not None
        row = int(row)
        if 0 <= row < self._capacity:
            self._flags[row] = FLAG_LIVE

    def is_deleted(self, row: InternalId) -> bool:
        assert self._flags is not None
        row = int(row)
        if not 0 <= row < self._num_rows:
            return True
        return bool(self._flags[row] == FLAG_DELETED)

    def deleted_ids(self) -> set[InternalId]:
        assert self._flags is not None
        rows = np.flatnonzero(self._flags[: self._num_rows] == FLAG_DELETED)
        return {int(r) for r in rows}

    # ------------------------------------------------------------------ #
    # Reads — the VectorSource protocol
    # ------------------------------------------------------------------ #

    def get(self, internal_id: InternalId) -> np.ndarray:
        assert self._vectors is not None
        return self._vectors[int(internal_id)]

    def gather(self, internal_ids: Sequence[InternalId]) -> np.ndarray:
        """Fancy-index a batch of rows out of the map.

        NumPy fancy indexing on a memmap copies into a fresh array, which is
        what we want: the result is contiguous, so the downstream matmul runs at
        full speed instead of striding across pages.
        """
        assert self._vectors is not None
        idx = np.asarray(internal_ids, dtype=np.int64)
        return self._vectors[idx]

    # ------------------------------------------------------------------ #
    # Durability
    # ------------------------------------------------------------------ #

    def flush(self) -> None:
        """Push dirty pages to the OS and fsync.

        ADR-005: the insert hot path deliberately does *not* do this — the WAL
        provides durability and mmap writes are made durable at checkpoint. This
        is the checkpoint half of that bargain.
        """
        for m in (self._vectors, self._flags):
            if m is not None:
                m.flush()
        # memmap.flush() is msync; it does not necessarily fsync the file's
        # metadata, so do that explicitly.
        for p in (self._vectors_path, self._flags_path):
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
            except OSError:
                # Some filesystems reject fsync on a read-only descriptor.
                pass
            finally:
                os.close(fd)

    def close(self) -> None:
        self.flush()
        self._unmap()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self._num_rows

    def __repr__(self) -> str:
        return (
            f"VectorStore(path={self.path.name!r}, dim={self.dim}, "
            f"rows={self._num_rows}/{self._capacity}, "
            f"deleted={self.num_deleted})"
        )

    # ------------------------------------------------------------------ #
    # Compaction — ADR-010's optimize()
    # ------------------------------------------------------------------ #

    def compact(self, keep: Sequence[InternalId]) -> dict[InternalId, InternalId]:
        """Rewrite the store keeping only ``keep``, in order.

        Returns the ``old_row -> new_row`` mapping so the caller can rewrite its
        id maps and rebuild indexes. This is the only operation in PyVec that
        renumbers internal ids, which is exactly why ``optimize()`` has to
        rebuild the dense index rather than patch it.
        """
        keep = [int(i) for i in keep]
        itemsize = np.dtype(VECTOR_DTYPE).itemsize
        tmp_vectors = Path(str(self._vectors_path) + ".compact")
        tmp_flags = Path(str(self._flags_path) + ".compact")
        new_capacity = max(len(keep), 1)

        with open(tmp_vectors, "wb") as f:
            f.truncate(new_capacity * self.dim * itemsize)
        with open(tmp_flags, "wb") as f:
            f.truncate(new_capacity)

        dst = np.memmap(
            tmp_vectors,
            dtype=VECTOR_DTYPE,
            mode="r+",
            shape=(new_capacity, self.dim),
        )
        dst_flags = np.memmap(
            tmp_flags, dtype=np.uint8, mode="r+", shape=(new_capacity,)
        )
        assert self._vectors is not None
        mapping: dict[InternalId, InternalId] = {}
        for new_row, old_row in enumerate(keep):
            dst[new_row] = self._vectors[old_row]
            mapping[old_row] = new_row
        dst_flags[: len(keep)] = FLAG_LIVE
        dst.flush()
        dst_flags.flush()
        for m in (dst, dst_flags):
            base = getattr(m, "_mmap", None)
            if base is not None:
                base.close()

        self._unmap()
        os.replace(tmp_vectors, self._vectors_path)
        os.replace(tmp_flags, self._flags_path)
        self._capacity = new_capacity
        self._num_rows = len(keep)
        self._map()
        return mapping
