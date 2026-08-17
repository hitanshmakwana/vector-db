"""IVF-Flat — Inverted File index with flat (uncompressed) storage.

Much simpler than HNSW and it exists here to be *compared* against it
(ADR-002): different characteristics, different failure modes, and a second
Pareto curve turns a single number into an engineering result.

Mental model (LEARNING.md layer 2):

1. **Train** — k-means over the vectors gives ``nlist`` centroids.
2. **Assign** — every vector joins its nearest centroid's posting list. The
   dataset is now partitioned into ``nlist`` buckets. That table of
   centroid -> vector ids is the "inverted file".
3. **Query** — score the query against all ``nlist`` centroids, take the
   ``nprobe`` closest, and brute-force scan only those buckets.

The trade-offs you have to be able to articulate:

* Higher ``nlist`` -> smaller buckets -> less scanning, but more centroid
  comparisons and a longer k-means.
* Higher ``nprobe`` -> more recall, more vectors scanned, slower.
* IVF loves clustered data and suffers on uniformly distributed data, because
  cluster boundaries cut through true neighbourhoods. Natural-language
  embeddings cluster well; random Gaussian vectors do not.
* Build time is dominated by k-means, not by insertion. The opposite of HNSW.

**Incremental inserts** (ARCHITECTURE.md "Note on IVF and incremental
inserts"): textbook IVF fixes centroids at build time. New vectors here are
assigned to the nearest *existing* centroid, which is correct but degrades as
the data distribution drifts away from centroids trained on an earlier sample.
:meth:`IVFFlatIndex.optimize` retrains and reassigns. This is documented loudly
because it is a real limitation, not an implementation gap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyvec.core.distance import as_vector, distance
from pyvec.core.errors import CorruptDataError
from pyvec.core.kmeans import DEFAULT_TRAIN_SAMPLE, assign, kmeans
from pyvec.core.types import VECTOR_DTYPE, InternalId, Metric, VectorSource

__all__ = ["IVFFlatIndex"]

#: PRD NF1 pins the acceptance target at ``nlist=256, nprobe=16``.
DEFAULT_NLIST = 256
DEFAULT_NPROBE = 16

#: Vectors buffered before an automatic first training run. Below this a scan of
#: everything is faster than clustering it, and k-means on a handful of points
#: produces meaningless centroids.
MIN_TRAIN_SIZE = 256

#: Automatically retrain once the collection has grown this many times larger
#: than it was at the last training run. Without a guard like this, a collection
#: filled by streaming inserts would keep serving centroids learned from its
#: first few hundred vectors — technically "online assignment to the nearest
#: existing centroid", but with recall quietly collapsing as the distribution
#: drifts (ARCHITECTURE.md's warning about IVF and incremental inserts). Set to
#: ``None`` to get the textbook behaviour of never retraining unless asked.
DEFAULT_RETRAIN_GROWTH = 4.0


class IVFFlatIndex:
    """Coarse-quantiser index: k-means buckets plus exhaustive in-bucket scan."""

    name = "ivf"

    def __init__(
        self,
        dim: int,
        metric: Metric,
        source: VectorSource,
        *,
        nlist: int = DEFAULT_NLIST,
        nprobe: int = DEFAULT_NPROBE,
        max_iter: int = 25,
        train_sample: int | None = DEFAULT_TRAIN_SAMPLE,
        retrain_growth_factor: float | None = DEFAULT_RETRAIN_GROWTH,
        seed: int = 42,
        **_ignored: Any,
    ) -> None:
        self.dim = int(dim)
        self.metric = Metric.parse(metric)
        self.source = source
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.max_iter = int(max_iter)
        self.train_sample = train_sample
        self.retrain_growth_factor = retrain_growth_factor
        self.seed = int(seed)

        #: ``(nlist, dim)`` once trained, else ``None``.
        self.centroids: np.ndarray | None = None
        #: The inverted file itself: centroid id -> internal ids.
        self.postings: dict[int, list[InternalId]] = {}
        #: Ids added but not yet in a posting list (index still untrained).
        self._pending: list[InternalId] = []
        self._deleted: set[InternalId] = set()
        self._num_assigned = 0
        #: Vector count at the last training run, for the growth-based retrain.
        self._trained_size = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._num_assigned + len(self._pending) - len(self._deleted)

    @property
    def is_trained(self) -> bool:
        return self.centroids is not None

    @property
    def params(self) -> dict[str, Any]:
        return {"nlist": self.nlist, "nprobe": self.nprobe}

    def stats(self) -> dict[str, Any]:
        sizes = [len(v) for v in self.postings.values()]
        return {
            "type": self.name,
            "num_vectors": len(self),
            "num_deleted": len(self._deleted),
            "trained": self.is_trained,
            "num_centroids": 0 if self.centroids is None else len(self.centroids),
            "pending_unassigned": len(self._pending),
            "posting_list_min": min(sizes) if sizes else 0,
            "posting_list_max": max(sizes) if sizes else 0,
            "posting_list_mean": float(np.mean(sizes)) if sizes else 0.0,
            "memory_bytes": self.memory_bytes(),
            **self.params,
        }

    def memory_bytes(self) -> int:
        centroid_bytes = 0 if self.centroids is None else self.centroids.nbytes
        posting_bytes = sum(len(v) for v in self.postings.values()) * 8
        return centroid_bytes + posting_bytes + len(self._pending) * 8

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, ids: Sequence[InternalId] | None = None) -> None:
        """Learn centroids from ``ids`` (default: everything known) and assign.

        Safe to call repeatedly; each call rebuilds the posting lists from
        scratch, which is exactly what :meth:`optimize` wants.
        """
        all_ids = self._all_ids() if ids is None else [int(i) for i in ids]
        if not all_ids:
            return

        x = self.source.gather(all_ids)
        result = kmeans(
            x,
            self.nlist,
            metric=self.metric,
            max_iter=self.max_iter,
            seed=self.seed,
            sample=self.train_sample,
        )
        self.centroids = result.centroids
        self.postings = {c: [] for c in range(len(self.centroids))}
        self._num_assigned = 0
        self._pending = []
        self._assign_batch(all_ids, x)
        self._trained_size = len(all_ids)

    def _all_ids(self) -> list[InternalId]:
        ids: list[InternalId] = []
        for bucket in self.postings.values():
            ids.extend(bucket)
        ids.extend(self._pending)
        return ids

    def _assign_batch(
        self, ids: Sequence[InternalId], vectors: np.ndarray | None = None
    ) -> None:
        assert self.centroids is not None
        if not len(ids):
            return
        x = self.source.gather(ids) if vectors is None else vectors
        labels, _ = assign(x, self.centroids, self.metric)
        for i, label in zip(ids, labels):
            self.postings.setdefault(int(label), []).append(int(i))
        self._num_assigned += len(ids)

    def optimize(self) -> dict[str, Any]:
        """Retrain k-means over current data and drop tombstoned ids.

        This is what the ``POST /collections/{name}/optimize`` endpoint calls for
        an IVF collection. Users need to run it periodically if they insert
        continuously — see the module docstring.
        """
        live = [i for i in self._all_ids() if i not in self._deleted]
        self._deleted.clear()
        self.centroids = None
        self.postings = {}
        self._pending = []
        self._num_assigned = 0
        self._trained_size = 0
        if live:
            self.train(live)
        return {"retrained": True, "num_vectors": len(live), "nlist": self.nlist}

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add(
        self, ids: Sequence[InternalId], vectors: np.ndarray | None = None
    ) -> None:
        """Assign to the nearest existing centroid, or buffer until trainable."""
        ids = [int(i) for i in ids]
        if not ids:
            return
        if self.centroids is None:
            self._pending.extend(ids)
            # First time we have enough data to cluster meaningfully, do it.
            if len(self._pending) >= max(MIN_TRAIN_SIZE, self.nlist):
                self.train()
            return

        self._assign_batch(ids, vectors)

        if self.retrain_growth_factor and self._trained_size:
            size = self._num_assigned + len(self._pending)
            if size >= self._trained_size * self.retrain_growth_factor:
                self.train()

    def remove(self, ids: Sequence[InternalId]) -> None:
        """Tombstone (ADR-010). Space returns on :meth:`optimize`."""
        self._deleted.update(int(i) for i in ids)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        nprobe: int | None = None,
        exclude: set[InternalId] | None = None,
        **_params: Any,
    ) -> list[tuple[InternalId, float]]:
        """Top-``k`` from the ``nprobe`` nearest buckets."""
        if k <= 0:
            return []
        q = as_vector(query, self.dim)
        dead = self._deleted if not exclude else self._deleted | set(exclude)

        if self.centroids is None:
            # Untrained: everything is still pending, so this degenerates to a
            # full scan. Correct, just not fast — and the only honest answer
            # before there is enough data to cluster.
            candidates = [i for i in self._pending if i not in dead]
            return self._scan(q, candidates, k)

        probe = max(1, min(int(nprobe or self.nprobe), len(self.centroids)))

        # One matmul against all centroids, then argpartition for the top probe.
        cd = distance(self.metric, q, self.centroids)
        if probe >= cd.shape[0]:
            chosen = np.argsort(cd)
        else:
            chosen = np.argpartition(cd, probe - 1)[:probe]
            chosen = chosen[np.argsort(cd[chosen])]

        candidates: list[InternalId] = []
        for c in chosen:
            bucket = self.postings.get(int(c))
            if bucket:
                candidates.extend(bucket)
        # Vectors inserted before the first training run are not in any bucket;
        # including them keeps results correct rather than silently invisible.
        if self._pending:
            candidates.extend(self._pending)
        if dead:
            candidates = [i for i in candidates if i not in dead]
        return self._scan(q, candidates, k)

    def _scan(
        self, q: np.ndarray, candidates: Sequence[InternalId], k: int
    ) -> list[tuple[InternalId, float]]:
        """Exhaustive scan of the selected candidates — the "Flat" in IVF-Flat."""
        if not candidates:
            return []
        x = self.source.gather(candidates)
        d = distance(self.metric, q, x)
        take = min(k, d.shape[0])
        part = np.argpartition(d, take - 1)[:take]
        part = part[np.argsort(d[part], kind="stable")]
        ids = np.asarray(candidates, dtype=np.int64)
        return [(int(ids[i]), float(d[i])) for i in part]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    # ARCHITECTURE.md §3: "centroids as .npy, posting lists as JSON. Trivial."
    # Both written to temp files and atomically renamed so a crash mid-save
    # leaves the previous snapshot intact rather than a half-written one.

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        centroid_path = path.with_suffix(".centroids.npy")
        posting_path = path.with_suffix(".postings.json")

        if self.centroids is not None:
            tmp = Path(str(centroid_path) + ".tmp")
            # Write through an open handle: np.save would otherwise "helpfully"
            # append another .npy to a path that does not end in one.
            with open(tmp, "wb") as f:
                np.save(f, self.centroids)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(centroid_path)
        elif centroid_path.exists():
            centroid_path.unlink()

        payload = {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "seed": self.seed,
            "max_iter": self.max_iter,
            "trained": self.centroids is not None,
            "num_assigned": self._num_assigned,
            "trained_size": self._trained_size,
            "retrain_growth_factor": self.retrain_growth_factor,
            # JSON object keys must be strings; parsed back to int on load.
            "postings": {str(c): v for c, v in self.postings.items() if v},
            "pending": self._pending,
            "deleted": sorted(self._deleted),
        }
        tmp = Path(str(posting_path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(posting_path)

    def load(self, path: Path) -> None:
        path = Path(path)
        posting_path = path.with_suffix(".postings.json")
        centroid_path = path.with_suffix(".centroids.npy")
        if not posting_path.exists():
            return

        with open(posting_path, encoding="utf-8") as f:
            payload = json.load(f)

        self.nlist = int(payload["nlist"])
        self.nprobe = int(payload["nprobe"])
        self.seed = int(payload.get("seed", self.seed))
        self.max_iter = int(payload.get("max_iter", self.max_iter))
        self.postings = {
            int(c): [int(i) for i in v] for c, v in payload["postings"].items()
        }
        self._pending = [int(i) for i in payload.get("pending", [])]
        self._deleted = {int(i) for i in payload.get("deleted", [])}
        self._num_assigned = int(payload.get("num_assigned", 0))
        self._trained_size = int(payload.get("trained_size", self._num_assigned))
        if "retrain_growth_factor" in payload:
            self.retrain_growth_factor = payload["retrain_growth_factor"]

        if payload.get("trained"):
            if not centroid_path.exists():
                raise CorruptDataError(
                    f"{posting_path} says trained but {centroid_path} is missing"
                )
            self.centroids = np.ascontiguousarray(
                np.load(centroid_path), dtype=VECTOR_DTYPE
            )
            if self.centroids.shape[1] != self.dim:
                raise CorruptDataError(
                    f"{centroid_path}: centroid dim {self.centroids.shape[1]} "
                    f"!= collection dim {self.dim}"
                )
        else:
            self.centroids = None
