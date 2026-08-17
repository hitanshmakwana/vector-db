"""The Collection — PyVec's top-level object.

A collection owns, per ARCHITECTURE.md §2:

* a **dense index** (HNSW, IVF-Flat, or flat), chosen at create time
* optionally a **sparse index** (BM25), if ``text_field`` is set
* a **vector store** (mmap float32 rows)
* a **metadata store** (id -> dict, held in RAM, checkpointed as JSON)
* a **WAL** for durability
* a **read-write lock** for concurrency

It is also where the two id spaces meet: clients use string ids, everything
internal uses the monotonic int row number.

Write ordering is the important part of this file. Every mutation follows::

    acquire WRITE lock
    validate
    normalise vector (cosine only)
    append WAL entry + fsync      <-- durability point
    write vector to mmap store    <-- no fsync; checkpoint handles it
    update metadata dict (dirty)
    dense_index.add()
    bm25_index.add()
    release lock

and every checkpoint follows::

    flush mmap
    write metadata.json  via temp + atomic rename
    write index snapshot via temp + atomic rename
    write manifest.json  via temp + atomic rename
    truncate WAL                  <-- last, so a crash before it just replays

Recovery reverses it: load the snapshot, then replay whatever the WAL still
holds. Replay is idempotent, so replaying an op that made it into the snapshot
is harmless — which is what makes the "crash between snapshot and truncate"
window safe.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pyvec.core.distance import as_vector, normalize, score_from_distance
from pyvec.core.errors import (
    CorruptDataError,
    IdExistsError,
    IdNotFoundError,
    InvalidDimensionError,
    InvalidRequestError,
    NoTextFieldError,
)
from pyvec.core.rwlock import RWLock
from pyvec.core.types import (
    VECTOR_DTYPE,
    ExternalId,
    IndexType,
    InternalId,
    Metric,
)
from pyvec.fusion.rrf import DEFAULT_RRF_K, FusedResult, reciprocal_rank_fusion
from pyvec.indexes import build_dense_index
from pyvec.indexes.bm25 import BM25Index
from pyvec.storage.mmap_store import VectorStore
from pyvec.storage.wal import WAL, FsyncPolicy, OpType

__all__ = ["Collection", "SearchHit", "MANIFEST_FILE", "METADATA_FILE"]

MANIFEST_FILE = "manifest.json"
METADATA_FILE = "metadata.json"
DENSE_INDEX_FILE = "dense.idx"
SPARSE_INDEX_FILE = "sparse.json"
SNAPSHOT_DIR = "snapshots"

MANIFEST_VERSION = 1

#: API_SPEC: batch insert over this size is rejected with PAYLOAD_TOO_LARGE.
MAX_BATCH = 1000

#: How many extra candidates to pull from an index when a metadata filter is in
#: play. API_SPEC is explicit that filtering happens *after* retrieval and "may
#: return fewer than k if restrictive" — over-fetching narrows that gap
#: substantially for cheap, without pretending to be real pre-filtering.
FILTER_OVERFETCH = 10

#: Floor on filtered retrieval depth. Scaling purely by ``k`` makes small-``k``
#: filtered queries nearly useless: ``k=1`` would look at just 10 candidates, so
#: any filter not satisfied by the very nearest handful returns nothing at all.
#: A fixed floor costs one wider scan and makes the common
#: "nearest match where category = x" query behave the way people expect.
MIN_FILTER_FETCH = 100

#: Ceiling, so a large ``k`` with a filter cannot turn into a full table scan.
MAX_OVERFETCH = 1000


class SearchHit:
    """One result row, in the shape the API returns."""

    __slots__ = ("id", "internal_id", "score", "metadata", "ranks")

    def __init__(
        self,
        id: ExternalId,
        internal_id: InternalId,
        score: float,
        metadata: dict | None = None,
        ranks: dict[str, int] | None = None,
    ) -> None:
        self.id = id
        self.internal_id = internal_id
        self.score = score
        self.metadata = metadata or {}
        self.ranks = ranks or {}

    def __repr__(self) -> str:
        return f"SearchHit(id={self.id!r}, score={self.score:.4f})"

    def to_dict(self, include_ranks: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata,
        }
        if include_ranks:
            out["rrf_score"] = self.score
            out.pop("score")
            out["dense_rank"] = self.ranks.get("dense")
            out["sparse_rank"] = self.ranks.get("sparse")
        return out


class Collection:
    """A named set of vectors with its indexes, storage and durability."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        name: str,
        path: Path,
        dimension: int,
        metric: Metric | str = Metric.COSINE,
        index_type: IndexType | str = IndexType.HNSW,
        index_params: Mapping[str, Any] | None = None,
        text_field: str | None = None,
        capacity: int | None = None,
        bm25_params: Mapping[str, Any] | None = None,
        fsync_policy: FsyncPolicy | str = FsyncPolicy.ENTRY,
        wal_enabled: bool = True,
        created_at: str | None = None,
    ) -> None:
        self.name = name
        self.path = Path(path)
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise InvalidDimensionError(
                f"dimension must be positive, got {self.dimension}"
            )
        self.metric = Metric.parse(metric)
        self.index_type = IndexType.parse(index_type)
        self.index_params = dict(index_params or {})
        self.text_field = text_field
        self.bm25_params = dict(bm25_params or {})
        self.created_at = created_at or _utc_now()

        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = RWLock()

        self.store = VectorStore(self.path, self.dimension, capacity=capacity)
        self.wal = WAL(
            self.path / "wal.log", fsync_policy=fsync_policy, enabled=wal_enabled
        )

        #: External id -> internal row, and back.
        self._id_map: dict[ExternalId, InternalId] = {}
        self._rev_map: dict[InternalId, ExternalId] = {}
        self._metadata: dict[InternalId, dict] = {}
        self._deleted: set[InternalId] = set()
        self._next_id: InternalId = 0
        self._metadata_dirty = False
        #: Live vector count, maintained incrementally.
        #:
        #: It cannot be derived as ``len(_id_map) - len(_deleted)``: an upsert
        #: repoints an external id at a new row and tombstones the old one, so
        #: the tombstone outlives its entry in ``_id_map`` and that subtraction
        #: undercounts by one per upsert.
        self._live_count = 0

        self.dense = build_dense_index(
            self.index_type,
            dim=self.dimension,
            metric=self.metric,
            source=self.store,
            params=self.index_params,
        )
        self.sparse = (
            BM25Index(**self.bm25_params) if self.text_field else None
        )

        self._closed = False
        #: Populated by :meth:`_recover`; empty for a freshly created collection.
        self.recovery_report: dict[str, Any] = {}
        # Counters surfaced by GET /metrics.
        self.stats_counters: dict[str, int] = {
            "inserts": 0,
            "deletes": 0,
            "queries_dense": 0,
            "queries_text": 0,
            "queries_hybrid": 0,
        }

    # -- factories --------------------------------------------------------- #

    @classmethod
    def create(
        cls,
        name: str,
        root: Path | str,
        dimension: int,
        *,
        metric: Metric | str = Metric.COSINE,
        index_type: IndexType | str = IndexType.HNSW,
        index_params: Mapping[str, Any] | None = None,
        text_field: str | None = None,
        capacity: int | None = None,
        bm25_params: Mapping[str, Any] | None = None,
        fsync_policy: FsyncPolicy | str = FsyncPolicy.ENTRY,
        wal_enabled: bool = True,
    ) -> "Collection":
        """Create a new collection under ``root/name`` and write its manifest."""
        path = Path(root) / name
        c = cls(
            name=name,
            path=path,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
            index_params=index_params,
            text_field=text_field,
            capacity=capacity,
            bm25_params=bm25_params,
            fsync_policy=fsync_policy,
            wal_enabled=wal_enabled,
        )
        # The index configuration is a durable decision, so it goes in the log
        # as well as the manifest (ARCHITECTURE.md op type CREATE_INDEX).
        c.wal.append_create_index(
            {
                "index_type": c.index_type.value,
                "params": c.index_params,
                "metric": c.metric.value,
                "dimension": c.dimension,
            }
        )
        c._write_manifest()
        return c

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        fsync_policy: FsyncPolicy | str | None = None,
        wal_enabled: bool | None = None,
    ) -> "Collection":
        """Open an existing collection: load the snapshot, then replay the WAL."""
        path = Path(path)
        manifest_path = path / MANIFEST_FILE
        if not manifest_path.exists():
            raise CorruptDataError(f"{path}: no {MANIFEST_FILE}")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        version = int(manifest.get("version", 0))
        if version != MANIFEST_VERSION:
            raise CorruptDataError(
                f"{manifest_path}: manifest version {version}, expected "
                f"{MANIFEST_VERSION}"
            )

        c = cls(
            name=manifest["name"],
            path=path,
            dimension=int(manifest["dimension"]),
            metric=manifest["metric"],
            index_type=manifest["index_type"],
            index_params=manifest.get("index_params") or {},
            text_field=manifest.get("text_field"),
            bm25_params=manifest.get("bm25_params") or {},
            fsync_policy=(
                fsync_policy
                if fsync_policy is not None
                else manifest.get("fsync_policy", FsyncPolicy.ENTRY)
            ),
            wal_enabled=(
                wal_enabled
                if wal_enabled is not None
                else bool(manifest.get("wal_enabled", True))
            ),
            created_at=manifest.get("created_at"),
        )
        c._load_snapshot(manifest)
        c._recover()
        return c

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._live_count

    def _is_live(self, ext_id: ExternalId) -> bool:
        internal_id = self._id_map.get(ext_id)
        return internal_id is not None and internal_id not in self._deleted

    def _recount_live(self) -> None:
        """Recompute the live count from scratch, after a bulk state change."""
        self._live_count = sum(
            1 for i in self._id_map.values() if i not in self._deleted
        )

    @property
    def num_vectors(self) -> int:
        return len(self)

    @property
    def num_deleted(self) -> int:
        return len(self._deleted)

    def stats(self) -> dict[str, Any]:
        """Everything ``GET /collections/{name}`` reports."""
        with self.lock.read():
            dense_stats = self.dense.stats()
            return {
                "name": self.name,
                "dimension": self.dimension,
                "metric": self.metric.value,
                "index": {
                    "type": self.index_type.value,
                    "params": self.dense.params,
                },
                "text_field": self.text_field,
                "num_vectors": len(self),
                "num_deleted": len(self._deleted),
                "memory_bytes": (
                    self.store.memory_bytes
                    + int(dense_stats.get("memory_bytes", 0))
                    + (self.sparse.memory_bytes() if self.sparse else 0)
                ),
                "disk_bytes": self._disk_bytes(),
                "created_at": self.created_at,
                "index_stats": dense_stats,
                "sparse_stats": self.sparse.stats() if self.sparse else None,
                "wal": self.wal.stats(),
                "counters": dict(self.stats_counters),
            }

    def _disk_bytes(self) -> int:
        total = self.store.disk_bytes + self.wal.size_bytes
        for name in (MANIFEST_FILE, METADATA_FILE, SPARSE_INDEX_FILE):
            p = self.path / name
            if p.exists():
                total += p.stat().st_size
        for p in self.path.glob("dense.*"):
            total += p.stat().st_size
        return total

    # ------------------------------------------------------------------ #
    # Insert
    # ------------------------------------------------------------------ #

    def insert(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        upsert: bool = False,
    ) -> dict[str, int]:
        """Insert ``(id, vector, metadata)`` records.

        Args:
            items: each needs ``id`` and ``vector``; ``metadata`` is optional.
            upsert: overwrite an existing id instead of raising
                :class:`IdExistsError`. Implemented as delete-then-insert, so the
                row number changes — HNSW cannot move a node in place, and
                pretending otherwise would leave stale edges (ADR-010).

        Returns:
            ``{"inserted": n, "duplicates_skipped": m}`` per API_SPEC.
            ``duplicates_skipped`` is always 0 in this implementation: a
            duplicate id is either an error (409) or an overwrite (``upsert``),
            never a silent skip. The field is kept because the documented
            response shape includes it.
        """
        if not items:
            raise InvalidRequestError("insert requires at least one item")
        if len(items) > MAX_BATCH:
            from pyvec.core.errors import PayloadTooLargeError

            raise PayloadTooLargeError(
                f"batch of {len(items)} exceeds the {MAX_BATCH}-item limit"
            )

        # Validate and coerce everything *before* taking the write lock or
        # touching the log: a batch that fails validation must not leave half of
        # itself durably applied.
        prepared: list[tuple[ExternalId, np.ndarray, dict]] = []
        seen_in_batch: set[ExternalId] = set()
        for item in items:
            if "id" not in item:
                raise InvalidRequestError("each item needs an 'id'")
            ext_id = str(item["id"])
            if "vector" not in item:
                raise InvalidRequestError(f"item {ext_id!r} has no 'vector'")
            if ext_id in seen_in_batch:
                raise InvalidRequestError(
                    f"id {ext_id!r} appears twice in the same batch"
                )
            seen_in_batch.add(ext_id)
            vec = as_vector(item["vector"], self.dimension)
            if not np.all(np.isfinite(vec)):
                raise InvalidRequestError(
                    f"item {ext_id!r} vector contains NaN or infinity"
                )
            if self.metric.normalize_on_insert:
                vec = normalize(vec)
            metadata = dict(item.get("metadata") or {})
            prepared.append((ext_id, vec, metadata))

        inserted = 0
        skipped = 0
        with self.lock.write():
            for ext_id, vec, metadata in prepared:
                if ext_id in self._id_map:
                    existing = self._id_map[ext_id]
                    if existing not in self._deleted:
                        if not upsert:
                            raise IdExistsError(
                                f"id {ext_id!r} already exists; pass upsert=true "
                                f"to overwrite"
                            )
                        self._delete_locked(ext_id)
                    else:
                        # Reinserting a tombstoned id: free the name for reuse.
                        self._forget_locked(ext_id)

                internal_id = self._next_id
                # --- durability point -------------------------------------- #
                self.wal.append_insert(internal_id, ext_id, metadata, vec)
                # --- apply ------------------------------------------------- #
                self._apply_insert(internal_id, ext_id, metadata, vec)
                inserted += 1

        self.stats_counters["inserts"] += inserted
        return {"inserted": inserted, "duplicates_skipped": skipped}

    def _apply_insert(
        self,
        internal_id: InternalId,
        ext_id: ExternalId,
        metadata: dict,
        vec: np.ndarray,
    ) -> None:
        """Mutate in-memory + mmap state. Shared by the insert path and replay."""
        was_live = self._is_live(ext_id)
        self.store.write(internal_id, vec)
        self._id_map[ext_id] = internal_id
        self._rev_map[internal_id] = ext_id
        self._metadata[internal_id] = metadata
        self._deleted.discard(internal_id)
        self._metadata_dirty = True
        self._next_id = max(self._next_id, internal_id + 1)
        if not was_live:
            self._live_count += 1

        # Vector is in the store before the index reads it back — that ordering
        # is why DenseIndex.add() can take ids alone.
        self.dense.add([internal_id])
        if self.sparse is not None:
            text = metadata.get(self.text_field or "", "")
            if isinstance(text, str) and text:
                self.sparse.add(internal_id, text)

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #

    def delete(self, ext_id: ExternalId) -> None:
        """Tombstone one id. Raises :class:`IdNotFoundError` if not live."""
        with self.lock.write():
            internal_id = self._id_map.get(ext_id)
            if internal_id is None or internal_id in self._deleted:
                raise IdNotFoundError(f"id {ext_id!r} not found")
            self.wal.append_delete(internal_id, ext_id)
            self._delete_locked(ext_id)
        self.stats_counters["deletes"] += 1

    def _delete_locked(self, ext_id: ExternalId) -> None:
        internal_id = self._id_map[ext_id]
        # Idempotent: WAL replay can hand us a delete for an already-tombstoned
        # row, and that must not decrement the count twice.
        if internal_id not in self._deleted:
            self._live_count -= 1
        self._deleted.add(internal_id)
        self.store.mark_deleted(internal_id)
        self.dense.remove([internal_id])
        if self.sparse is not None:
            self.sparse.remove([internal_id])
        self._metadata_dirty = True

    def _forget_locked(self, ext_id: ExternalId) -> None:
        """Drop all trace of a tombstoned external id so it can be reinserted."""
        internal_id = self._id_map.pop(ext_id, None)
        if internal_id is None:
            return
        self._rev_map.pop(internal_id, None)
        self._metadata.pop(internal_id, None)
        self._metadata_dirty = True

    # ------------------------------------------------------------------ #
    # Point lookup
    # ------------------------------------------------------------------ #

    def get(self, ext_id: ExternalId) -> dict[str, Any]:
        """Fetch a stored vector and its metadata."""
        with self.lock.read():
            internal_id = self._id_map.get(ext_id)
            if internal_id is None or internal_id in self._deleted:
                raise IdNotFoundError(f"id {ext_id!r} not found")
            return {
                "id": ext_id,
                "vector": self.store.get(internal_id).astype(float).tolist(),
                "metadata": self._metadata.get(internal_id, {}),
            }

    def contains(self, ext_id: ExternalId) -> bool:
        with self.lock.read():
            internal_id = self._id_map.get(ext_id)
            return internal_id is not None and internal_id not in self._deleted

    # ------------------------------------------------------------------ #
    # Dense search
    # ------------------------------------------------------------------ #

    def search(
        self,
        vector: np.ndarray | Sequence[float],
        k: int = 10,
        *,
        filter: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """ANN search by vector similarity."""
        if k <= 0:
            raise InvalidRequestError(f"k must be positive, got {k}")
        q = as_vector(vector, self.dimension)
        if self.metric.normalize_on_insert:
            q = normalize(q)
        search_params = dict(params or {})

        with self.lock.read():
            fetch = self._fetch_size(k, filter)
            raw = self.dense.search(
                q, fetch, exclude=self._deleted, **search_params
            )
            hits = [
                SearchHit(
                    id=self._rev_map[i],
                    internal_id=i,
                    score=score_from_distance(self.metric, d),
                    metadata=self._metadata.get(i, {}),
                )
                for i, d in raw
                if i in self._rev_map
            ]
            hits = self._apply_filter(hits, filter)
        self.stats_counters["queries_dense"] += 1
        return hits[:k]

    # ------------------------------------------------------------------ #
    # Sparse search
    # ------------------------------------------------------------------ #

    def search_text(
        self,
        text: str,
        k: int = 10,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """BM25 search over the collection's ``text_field``."""
        if self.sparse is None:
            raise NoTextFieldError(
                f"collection {self.name!r} has no text_field, so BM25 search is "
                f"unavailable; recreate it with text_field set"
            )
        if k <= 0:
            raise InvalidRequestError(f"k must be positive, got {k}")

        with self.lock.read():
            fetch = self._fetch_size(k, filter)
            raw = self.sparse.search(text, fetch, exclude=self._deleted)
            hits = [
                SearchHit(
                    id=self._rev_map[i],
                    internal_id=i,
                    score=float(s),
                    metadata=self._metadata.get(i, {}),
                )
                for i, s in raw
                if i in self._rev_map
            ]
            hits = self._apply_filter(hits, filter)
        self.stats_counters["queries_text"] += 1
        return hits[:k]

    # ------------------------------------------------------------------ #
    # Hybrid search
    # ------------------------------------------------------------------ #

    def search_hybrid(
        self,
        vector: np.ndarray | Sequence[float],
        text: str,
        k: int = 10,
        *,
        filter: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Dense + BM25, fused with RRF.

        Both retrievers run over the same collection and the same insert
        (ADR-011), so there is no client-side coordination and no chance of the
        two sides disagreeing about what exists.
        """
        if self.sparse is None:
            raise NoTextFieldError(
                f"collection {self.name!r} has no text_field, so hybrid search "
                f"is unavailable"
            )
        if k <= 0:
            raise InvalidRequestError(f"k must be positive, got {k}")

        p = dict(params or {})
        rrf_k = int(p.pop("rrf_k", DEFAULT_RRF_K))
        # API_SPEC: both candidate counts default to 10 * k.
        dense_n = int(p.pop("dense_candidates", 10 * k))
        sparse_n = int(p.pop("sparse_candidates", 10 * k))

        q = as_vector(vector, self.dimension)
        if self.metric.normalize_on_insert:
            q = normalize(q)

        with self.lock.read():
            dense_raw = self.dense.search(
                q, dense_n, exclude=self._deleted, **p
            )
            sparse_raw = self.sparse.search(text, sparse_n, exclude=self._deleted)

            fused: list[FusedResult] = reciprocal_rank_fusion(
                {
                    "dense": [i for i, _ in dense_raw],
                    "sparse": [i for i, _ in sparse_raw],
                },
                k=rrf_k,
            )
            hits = [
                SearchHit(
                    id=self._rev_map[r.id],
                    internal_id=r.id,
                    score=r.score,
                    metadata=self._metadata.get(r.id, {}),
                    ranks=r.ranks,
                )
                for r in fused
                if r.id in self._rev_map
            ]
            hits = self._apply_filter(hits, filter)
        self.stats_counters["queries_hybrid"] += 1
        return hits[:k]

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    def _fetch_size(self, k: int, filter: Mapping[str, Any] | None) -> int:
        if not filter:
            return k
        return min(max(k * FILTER_OVERFETCH, MIN_FILTER_FETCH), MAX_OVERFETCH)

    @staticmethod
    def _apply_filter(
        hits: list[SearchHit], filter: Mapping[str, Any] | None
    ) -> list[SearchHit]:
        """Shallow-equality metadata filter, applied post-retrieval.

        ARCHITECTURE.md "On filter placement": this is a post-filter, so a very
        restrictive predicate can return fewer than ``k`` results even though
        matching documents exist deeper in the index. Real pre-filtering means
        modifying the graph walk to skip non-matching nodes (Qdrant does this);
        that is a v2 item, and the honest thing is to document the limit rather
        than hide it.

        A list value means "any of", which costs nothing to support and covers
        most of what people reach for first.
        """
        if not filter:
            return hits
        out = []
        for hit in hits:
            md = hit.metadata
            ok = True
            for key, expected in filter.items():
                actual = md.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        ok = False
                        break
                elif actual != expected:
                    ok = False
                    break
            if ok:
                out.append(hit)
        return out

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def optimize(self) -> dict[str, Any]:
        """Compact tombstones and rebuild the dense index (ADR-010).

        Renumbers internal ids, so it rebuilds rather than patches: the mmap rows
        move, and every index keyed on row number has to be regenerated. That is
        the cost of tombstoning, paid once, when the user asks.
        """
        started = time.perf_counter()
        with self.lock.write():
            live_internal = sorted(
                i for i in self._rev_map if i not in self._deleted
            )
            removed = len(self._deleted)

            mapping = self.store.compact(live_internal)

            old_metadata = self._metadata
            old_rev = self._rev_map
            self._metadata = {}
            self._rev_map = {}
            self._id_map = {}
            for old_row, new_row in mapping.items():
                ext = old_rev[old_row]
                self._id_map[ext] = new_row
                self._rev_map[new_row] = ext
                self._metadata[new_row] = old_metadata.get(old_row, {})
            self._deleted.clear()
            self._next_id = len(live_internal)
            self._recount_live()

            # Fresh dense index over the renumbered rows.
            self.dense = build_dense_index(
                self.index_type,
                dim=self.dimension,
                metric=self.metric,
                source=self.store,
                params=self.index_params,
            )
            if self._next_id:
                self.dense.add(list(range(self._next_id)))

            if self.sparse is not None:
                self.sparse = BM25Index(**self.bm25_params)
                self._rebuild_sparse_locked()

            self._metadata_dirty = True
            self._checkpoint_locked()

        return {
            "compacted": removed,
            "num_vectors": len(self),
            "took_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _rebuild_sparse_locked(self) -> None:
        """Rebuild BM25 from metadata — the startup path from ARCHITECTURE.md."""
        if self.sparse is None or not self.text_field:
            return
        for internal_id, md in self._metadata.items():
            if internal_id in self._deleted:
                continue
            text = md.get(self.text_field, "")
            if isinstance(text, str) and text:
                self.sparse.add(internal_id, text)

    # ------------------------------------------------------------------ #
    # Checkpoint / snapshot
    # ------------------------------------------------------------------ #

    def checkpoint(self) -> dict[str, Any]:
        """Make current state durable and truncate the WAL."""
        with self.lock.write():
            return self._checkpoint_locked()

    def _checkpoint_locked(self) -> dict[str, Any]:
        started = time.perf_counter()

        # 1. Vector pages to disk.
        self.store.flush()
        # 2. Metadata, atomically.
        self._write_metadata()
        # 3. Index snapshots, atomically (each index does its own temp+rename).
        self.dense.save(self.path / DENSE_INDEX_FILE)
        if self.sparse is not None:
            self.sparse.save(self.path / SPARSE_INDEX_FILE)
        # 4. Manifest last among the state files — it is what `open` trusts to
        #    describe everything else.
        self._write_manifest()
        # 5. Only now is it safe to drop the log.
        self.wal.truncate()

        return {
            "num_vectors": len(self),
            "took_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def snapshot(self) -> dict[str, Any]:
        """Checkpoint, then copy the durable files into ``snapshots/<ts>/``."""
        import shutil

        with self.lock.write():
            self._checkpoint_locked()
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            dest = self.path / SNAPSHOT_DIR / stamp
            dest.mkdir(parents=True, exist_ok=True)
            for pattern in (
                MANIFEST_FILE,
                METADATA_FILE,
                SPARSE_INDEX_FILE,
                "vectors.bin",
                "flags.bin",
                "dense.*",
            ):
                for src in self.path.glob(pattern):
                    if src.is_file():
                        shutil.copy2(src, dest / src.name)
            return {"snapshot_id": stamp, "path": str(dest)}

    def _write_manifest(self) -> None:
        manifest = {
            "version": MANIFEST_VERSION,
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "index_type": self.index_type.value,
            "index_params": self.index_params,
            "text_field": self.text_field,
            "bm25_params": self.bm25_params,
            "created_at": self.created_at,
            "fsync_policy": self.wal.fsync_policy.value,
            "wal_enabled": self.wal.enabled,
            # Checkpointed counters. Recovery starts from these and the WAL
            # carries it the rest of the way.
            "next_internal_id": self._next_id,
            "num_rows": self.store.num_rows,
            "capacity": self.store.capacity,
        }
        _atomic_write_json(self.path / MANIFEST_FILE, manifest)

    def _write_metadata(self) -> None:
        if not self._metadata_dirty and (self.path / METADATA_FILE).exists():
            return
        payload = {
            "ids": {str(i): self._rev_map[i] for i in sorted(self._rev_map)},
            "metadata": {
                str(i): md for i, md in sorted(self._metadata.items())
            },
            "deleted": sorted(self._deleted),
        }
        _atomic_write_json(self.path / METADATA_FILE, payload)
        self._metadata_dirty = False

    # ------------------------------------------------------------------ #
    # Load / recovery
    # ------------------------------------------------------------------ #

    def _load_snapshot(self, manifest: Mapping[str, Any]) -> None:
        metadata_path = self.path / METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                payload = json.load(f)
            self._rev_map = {int(i): str(e) for i, e in payload["ids"].items()}
            self._id_map = {e: i for i, e in self._rev_map.items()}
            self._metadata = {
                int(i): md for i, md in payload["metadata"].items()
            }
            self._deleted = {int(i) for i in payload.get("deleted", [])}
            self._recount_live()

        self._next_id = int(manifest.get("next_internal_id", 0))
        self.store.num_rows = max(
            int(manifest.get("num_rows", 0)), self._next_id
        )

        self.dense.load(self.path / DENSE_INDEX_FILE)
        if self.sparse is not None:
            sparse_path = self.path / SPARSE_INDEX_FILE
            if sparse_path.exists():
                self.sparse.load(sparse_path)
            else:
                # ARCHITECTURE.md startup path: BM25 is cheap to rebuild from
                # metadata, so a missing sparse snapshot is not an error.
                self._rebuild_sparse_locked()
        self._metadata_dirty = False

    def _recover(self) -> None:
        """Replay WAL entries the snapshot does not already contain.

        Idempotent by construction: an INSERT whose ``internal_id`` is already
        mapped to the same external id is skipped, so replaying the overlap
        between snapshot and log changes nothing. That is what makes the
        "snapshot written, crash before truncate" window safe.
        """
        removed = self.wal.truncate_torn_tail()
        applied = {"insert": 0, "delete": 0, "skipped": 0}

        for entry in self.wal.replay():
            if entry.op is OpType.INSERT:
                internal_id, ext_id, metadata, vector = entry.as_insert(
                    self.dimension
                )
                if self._rev_map.get(internal_id) == ext_id:
                    applied["skipped"] += 1
                    self._next_id = max(self._next_id, internal_id + 1)
                    continue
                # An id being reused means the snapshot had an older occupant of
                # this row; the log is authoritative, so drop the stale mapping.
                stale = self._rev_map.get(internal_id)
                if stale is not None and stale != ext_id:
                    self._id_map.pop(stale, None)
                previous = self._id_map.get(ext_id)
                if previous is not None and previous != internal_id:
                    self._deleted.add(previous)
                self._apply_insert(internal_id, ext_id, metadata, vector)
                applied["insert"] += 1
            elif entry.op is OpType.DELETE:
                payload = entry.as_json()
                ext_id = str(payload["id"])
                if ext_id in self._id_map:
                    self._delete_locked(ext_id)
                    applied["delete"] += 1
                else:
                    applied["skipped"] += 1
            elif entry.op is OpType.CHECKPOINT:
                # Informational; the snapshot files are the real checkpoint.
                continue
            elif entry.op is OpType.CREATE_INDEX:
                continue

        self.recovery_report = {
            "wal_bytes_discarded": removed,
            **applied,
            "num_vectors": len(self),
        }
        # Tombstone flags live in the mmap and may predate the replayed entries.
        for internal_id in self.store.deleted_ids():
            if internal_id in self._rev_map:
                self._deleted.add(internal_id)
        self._recount_live()
        self.recovery_report["num_vectors"] = len(self)
        if applied["insert"] or applied["delete"] or removed:
            # Fold the replayed ops into the snapshot immediately so a second
            # crash does not have to replay them again.
            self._checkpoint_locked()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Checkpoint and release the mmap and log handles."""
        if self._closed:
            return
        try:
            with self.lock.write():
                self._checkpoint_locked()
        finally:
            self.wal.close()
            self.store.close()
            self._closed = True

    def drop(self) -> None:
        """Close and delete every on-disk file for this collection."""
        import shutil

        self.wal.close()
        self.store.close()
        self._closed = True
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Collection(name={self.name!r}, dim={self.dimension}, "
            f"metric={self.metric.value}, index={self.index_type.value}, "
            f"n={len(self)})"
        )

    # ------------------------------------------------------------------ #
    # Iteration helpers, used by benchmarks and tests
    # ------------------------------------------------------------------ #

    def iter_ids(self) -> Iterable[ExternalId]:
        for ext_id, internal_id in self._id_map.items():
            if internal_id not in self._deleted:
                yield ext_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via temp file + fsync + rename.

    ``os.replace`` is atomic on POSIX and on Windows, so a reader either sees the
    whole old file or the whole new one — never a half-written one. Without the
    fsync before the rename, the rename can land while the contents are still in
    the page cache, which on a power loss gives you an atomically-renamed empty
    file.
    """
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
