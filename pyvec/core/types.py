"""Shared types.

Two id spaces exist and it is important to keep them straight:

* **External id** (``str``) — what the client sends. Opaque.
* **Internal id** (``int``, monotonic from 0) — the row index into the mmap
  vector store, and the node id inside the indexes. Per ARCHITECTURE.md the
  vector at row ``i`` *is* the vector for internal id ``i``.

The mapping lives in the ``Collection`` and is checkpointed alongside metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

ExternalId = str
InternalId = int

#: Dtype for all stored vectors. ADR-012: float32 only, no float16/int8.
VECTOR_DTYPE = np.float32


class Metric(str, Enum):
    """Distance metric. ADR-009: fixed per collection at create time."""

    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"

    @classmethod
    def parse(cls, value: "str | Metric") -> "Metric":
        from pyvec.core.errors import InvalidMetricError

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise InvalidMetricError(
                f"unknown metric {value!r}; expected one of "
                f"{[m.value for m in cls]}"
            ) from None

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger user-facing ``score`` means a better match.

        API_SPEC: for ``cosine``/``dot`` higher is better, for ``l2`` lower is.
        """
        return self is not Metric.L2

    @property
    def normalize_on_insert(self) -> bool:
        """Cosine collections unit-normalise so the hot path is a dot product.

        LEARNING.md layer 0: with unit vectors, cosine == dot, so we only need
        one kernel in the inner loop.
        """
        return self is Metric.COSINE


class IndexType(str, Enum):
    """Dense index implementation. One per collection (PRD open question #1)."""

    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"

    @classmethod
    def parse(cls, value: "str | IndexType") -> "IndexType":
        from pyvec.core.errors import InvalidIndexTypeError

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise InvalidIndexTypeError(
                f"unknown index type {value!r}; expected one of "
                f"{[t.value for t in cls]}"
            ) from None


@runtime_checkable
class VectorSource(Protocol):
    """Read-only random access to stored vectors, keyed by internal id.

    Indexes never own vector memory — they hold a reference to the collection's
    store and read through it. That keeps exactly one copy of the float32 data
    in the process (the mmap) instead of one per index.
    """

    @property
    def dim(self) -> int: ...

    def get(self, internal_id: InternalId) -> np.ndarray:
        """One vector, shape ``(dim,)``."""

    def gather(self, internal_ids: Sequence[InternalId]) -> np.ndarray:
        """Many vectors, shape ``(len(ids), dim)``, in the order requested."""


class ArrayVectorSource:
    """A :class:`VectorSource` over a plain in-memory array.

    Used by tests and benchmarks that want to exercise an index without
    standing up a whole collection on disk.
    """

    __slots__ = ("_a",)

    def __init__(self, array: np.ndarray) -> None:
        a = np.ascontiguousarray(array, dtype=VECTOR_DTYPE)
        if a.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {a.shape}")
        self._a = a

    @property
    def dim(self) -> int:
        return int(self._a.shape[1])

    @property
    def array(self) -> np.ndarray:
        return self._a

    def get(self, internal_id: InternalId) -> np.ndarray:
        return self._a[internal_id]

    def gather(self, internal_ids: Sequence[InternalId]) -> np.ndarray:
        return self._a[np.asarray(internal_ids, dtype=np.int64)]


@runtime_checkable
class DenseIndex(Protocol):
    """The dense index interface from ARCHITECTURE.md.

    ``vectors`` is optional on :meth:`add`: the collection writes vectors to the
    store *before* calling the index, so the index can read them back through
    its :class:`VectorSource`. Passing them explicitly just saves a re-read.
    """

    def add(
        self, ids: Sequence[InternalId], vectors: np.ndarray | None = None
    ) -> None: ...

    def search(
        self, query: np.ndarray, k: int, **params: Any
    ) -> list[tuple[InternalId, float]]: ...

    def remove(self, ids: Sequence[InternalId]) -> None: ...

    def save(self, path: Any) -> None: ...

    def load(self, path: Any) -> None: ...
