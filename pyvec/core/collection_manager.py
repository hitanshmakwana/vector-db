"""Registry of open collections.

ARCHITECTURE.md §2: "The manager is a registry of open collections. It handles
create/drop/load-on-startup."

On startup it scans the data root for directories holding a ``manifest.json`` and
opens each one, which is where WAL replay happens. A collection that fails to
open does not take the whole server down — it is recorded and reported, because a
single corrupt collection should not make the other nine unavailable.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator, Mapping

from pyvec.core.collection import MANIFEST_FILE, Collection
from pyvec.core.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    InvalidRequestError,
)
from pyvec.core.types import IndexType, Metric
from pyvec.storage.wal import FsyncPolicy

__all__ = ["CollectionManager"]

#: Collection names become directory names, so they have to be filesystem-safe.
_FORBIDDEN = set('/\\:*?"<>|') | {".."}


class CollectionManager:
    """Owns the data root and the set of open collections."""

    def __init__(
        self,
        root: Path | str = "./data",
        *,
        autoload: bool = True,
        fsync_policy: FsyncPolicy | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, Collection] = {}
        #: Guards the registry dict only. Per-collection concurrency is the
        #: collection's own RWLock.
        self._lock = threading.Lock()
        self.load_errors: dict[str, str] = {}
        self._fsync_policy = fsync_policy
        if autoload:
            self.load_all()

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    def load_all(self) -> dict[str, Any]:
        """Open every collection found under the data root."""
        loaded: list[str] = []
        for child in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not child.is_dir() or not (child / MANIFEST_FILE).exists():
                continue
            name = child.name
            if name in self._collections:
                continue
            try:
                collection = Collection.open(
                    child, fsync_policy=self._fsync_policy
                )
            except Exception as exc:  # noqa: BLE001 — one bad collection is not fatal
                self.load_errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            with self._lock:
                self._collections[name] = collection
            loaded.append(name)
        return {"loaded": loaded, "errors": dict(self.load_errors)}

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_name(name: str) -> str:
        if not name or not name.strip():
            raise InvalidRequestError("collection name must not be empty")
        if any(ch in name for ch in _FORBIDDEN) or name in {".", ".."}:
            raise InvalidRequestError(
                f"collection name {name!r} contains characters that are not "
                f"allowed in a directory name"
            )
        return name

    def create(
        self,
        name: str,
        dimension: int,
        *,
        metric: Metric | str = Metric.COSINE,
        index_type: IndexType | str = IndexType.HNSW,
        index_params: Mapping[str, Any] | None = None,
        text_field: str | None = None,
        capacity: int | None = None,
        bm25_params: Mapping[str, Any] | None = None,
        fsync_policy: FsyncPolicy | str | None = None,
        wal_enabled: bool = True,
    ) -> Collection:
        self.validate_name(name)
        with self._lock:
            if name in self._collections:
                raise CollectionExistsError(f"collection {name!r} already exists")
            if (self.root / name / MANIFEST_FILE).exists():
                raise CollectionExistsError(
                    f"collection {name!r} already exists on disk but is not "
                    f"loaded; restart or call load_all()"
                )
        collection = Collection.create(
            name=name,
            root=self.root,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
            index_params=index_params,
            text_field=text_field,
            capacity=capacity,
            bm25_params=bm25_params,
            fsync_policy=(
                fsync_policy
                if fsync_policy is not None
                else (self._fsync_policy or FsyncPolicy.ENTRY)
            ),
            wal_enabled=wal_enabled,
        )
        with self._lock:
            self._collections[name] = collection
        return collection

    def get(self, name: str) -> Collection:
        with self._lock:
            collection = self._collections.get(name)
        if collection is None:
            raise CollectionNotFoundError(f"collection {name!r} not found")
        return collection

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._collections

    def drop(self, name: str) -> None:
        """Close and delete a collection, including its files."""
        with self._lock:
            collection = self._collections.pop(name, None)
        if collection is None:
            raise CollectionNotFoundError(f"collection {name!r} not found")
        collection.drop()

    def list(self) -> list[dict[str, Any]]:
        """Summary rows for ``GET /collections``."""
        with self._lock:
            collections = list(self._collections.values())
        return [
            {
                "name": c.name,
                "num_vectors": len(c),
                "dimension": c.dimension,
                "metric": c.metric.value,
                "index": c.index_type.value,
            }
            for c in collections
        ]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def checkpoint_all(self) -> dict[str, Any]:
        with self._lock:
            collections = list(self._collections.values())
        return {c.name: c.checkpoint() for c in collections}

    def close(self) -> None:
        """Checkpoint and close everything. Called on server shutdown."""
        with self._lock:
            collections = list(self._collections.values())
            self._collections.clear()
        for c in collections:
            try:
                c.close()
            except Exception:  # noqa: BLE001 — keep closing the rest
                continue

    def __len__(self) -> int:
        with self._lock:
            return len(self._collections)

    def __iter__(self) -> Iterator[Collection]:
        with self._lock:
            return iter(list(self._collections.values()))

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._collections

    def __enter__(self) -> "CollectionManager":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
