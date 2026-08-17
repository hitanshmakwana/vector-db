"""Brute-force index.

The correctness oracle. PRD NF5: "all indexes agree with brute-force baseline on
small collections (<=1k vectors) modulo floating-point ties." Every recall number
in the project is measured against this, so it must be exactly right and it must
never take a shortcut.

It is also a legitimate index choice for small collections: at a few thousand
vectors a single ``(n, dim)`` matmul beats any graph walk, and it has zero build
cost. API_SPEC exposes it as ``index.type == "flat"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyvec.core.distance import as_vector, distance
from pyvec.core.types import InternalId, Metric, VectorSource

__all__ = ["FlatIndex"]

#: Rows scored per matmul. Bounds the peak temporary allocation on a large scan
#: while staying big enough that BLAS is not call-overhead-bound.
SCAN_BLOCK = 32_768


class FlatIndex:
    """Exhaustive k-NN over every live vector."""

    name = "flat"

    def __init__(
        self,
        dim: int,
        metric: Metric,
        source: VectorSource,
        *,
        block: int = SCAN_BLOCK,
    ) -> None:
        self.dim = int(dim)
        self.metric = Metric.parse(metric)
        self.source = source
        self.block = int(block)
        self._ids: list[InternalId] = []
        self._deleted: set[InternalId] = set()

    # -- properties --------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._ids) - len(self._deleted)

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def stats(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "num_vectors": len(self),
            "num_deleted": len(self._deleted),
            "memory_bytes": self.memory_bytes(),
        }

    def memory_bytes(self) -> int:
        """Index-owned memory. Vectors live in the store, not here."""
        return len(self._ids) * 8 + len(self._deleted) * 8

    # -- mutation ----------------------------------------------------------- #

    def add(
        self, ids: Sequence[InternalId], vectors: np.ndarray | None = None
    ) -> None:
        """Register ids. Vectors are read through ``source`` at query time."""
        self._ids.extend(int(i) for i in ids)

    def remove(self, ids: Sequence[InternalId]) -> None:
        self._deleted.update(int(i) for i in ids)

    # -- query -------------------------------------------------------------- #

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude: set[InternalId] | None = None,
        **params: Any,
    ) -> list[tuple[InternalId, float]]:
        """Top-``k`` by ordering distance (lower first).

        ``params`` is accepted and ignored — flat has no search knobs, and the
        API passes the same ``params`` dict to whichever index a collection has.
        """
        q = as_vector(query, self.dim)
        dead = self._deleted if not exclude else self._deleted | set(exclude)

        live = [i for i in self._ids if i not in dead]
        if not live or k <= 0:
            return []

        k = min(k, len(live))
        best_ids: list[np.ndarray] = []
        best_ds: list[np.ndarray] = []

        for start in range(0, len(live), self.block):
            chunk = live[start : start + self.block]
            x = self.source.gather(chunk)
            d = distance(self.metric, q, x)
            # argpartition is O(n) vs. a full O(n log n) sort; on a 1M-row scan
            # this is a measurable share of query time.
            take = min(k, d.shape[0])
            part = np.argpartition(d, take - 1)[:take]
            best_ids.append(np.asarray(chunk, dtype=np.int64)[part])
            best_ds.append(d[part])

        ids = np.concatenate(best_ids)
        ds = np.concatenate(best_ds)
        order = np.argsort(ds, kind="stable")[:k]
        return [(int(ids[i]), float(ds[i])) for i in order]

    # -- persistence -------------------------------------------------------- #
    # Nothing to learn, so nothing to serialise beyond the id roster: the
    # vectors themselves are already durable in the collection's mmap store.

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path.with_suffix(".npz"),
            ids=np.asarray(self._ids, dtype=np.int64),
            deleted=np.asarray(sorted(self._deleted), dtype=np.int64),
        )

    def load(self, path: Path) -> None:
        path = Path(path).with_suffix(".npz")
        if not path.exists():
            return
        with np.load(path) as z:
            self._ids = [int(i) for i in z["ids"]]
            self._deleted = {int(i) for i in z["deleted"]}
