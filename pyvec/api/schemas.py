"""Pydantic request/response models.

These mirror API_SPEC.md exactly. Validation that can be expressed as a type
lives here so the handlers stay thin (ARCHITECTURE.md §1: "no business logic"),
and anything requiring collection state — dimension checks, id existence —
belongs to the domain layer instead.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "IndexConfig",
    "CreateCollectionRequest",
    "CreateCollectionResponse",
    "CollectionSummary",
    "ListCollectionsResponse",
    "CollectionDetail",
    "InsertItem",
    "InsertRequest",
    "InsertResponse",
    "QueryRequest",
    "TextQueryRequest",
    "HybridQueryRequest",
    "QueryResult",
    "QueryResponse",
    "HybridResult",
    "HybridResponse",
    "VectorResponse",
    "OptimizeResponse",
    "SnapshotResponse",
    "HealthResponse",
    "ErrorResponse",
]

#: API_SPEC: batches over 1000 items are rejected with PAYLOAD_TOO_LARGE. The
#: schema enforces it too, so an oversized batch is refused before the whole
#: body is turned into NumPy arrays.
MAX_BATCH_ITEMS = 1000


class IndexConfig(BaseModel):
    type: Literal["hnsw", "ivf", "flat"] = "hnsw"
    params: dict[str, Any] = Field(default_factory=dict)


class CreateCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    dimension: int = Field(gt=0, le=65536)
    metric: Literal["cosine", "l2", "dot"] = "cosine"
    index: IndexConfig = Field(default_factory=IndexConfig)
    text_field: str | None = None
    capacity: int | None = Field(default=None, gt=0)
    bm25_params: dict[str, Any] = Field(default_factory=dict)


class CreateCollectionResponse(BaseModel):
    name: str
    created_at: str


class CollectionSummary(BaseModel):
    name: str
    num_vectors: int
    dimension: int
    metric: str
    index: str


class ListCollectionsResponse(BaseModel):
    collections: list[CollectionSummary]


class CollectionDetail(BaseModel):
    """Loose by design: ``index_stats`` differs per index type."""

    model_config = ConfigDict(extra="allow")

    name: str
    dimension: int
    metric: str
    index: dict[str, Any]
    text_field: str | None
    num_vectors: int
    num_deleted: int
    memory_bytes: int
    disk_bytes: int


class InsertItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def _non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("vector must not be empty")
        return v


class InsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[InsertItem] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)
    upsert: bool = False


class InsertResponse(BaseModel):
    inserted: int
    duplicates_skipped: int = 0


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: list[float] = Field(min_length=1)
    k: int = Field(default=10, gt=0, le=10_000)
    filter: dict[str, Any] | None = None
    #: Index-specific knobs: ``ef_search`` for HNSW, ``nprobe`` for IVF.
    params: dict[str, Any] = Field(default_factory=dict)


class TextQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    k: int = Field(default=10, gt=0, le=10_000)
    filter: dict[str, Any] | None = None


class HybridQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: list[float] = Field(min_length=1)
    text: str = Field(min_length=1)
    k: int = Field(default=10, gt=0, le=10_000)
    filter: dict[str, Any] | None = None
    #: ``ef_search``/``nprobe`` plus ``dense_candidates``, ``sparse_candidates``
    #: (both default to ``10 * k``) and ``rrf_k`` (default 60).
    params: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    results: list[QueryResult]
    took_ms: float


class HybridResult(BaseModel):
    """No single similarity score: RRF works on ranks.

    The per-retriever ranks are returned instead, so a surprising result can be
    traced back to which side produced it. ``None`` means that retriever did not
    return this document at all.
    """

    id: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridResponse(BaseModel):
    results: list[HybridResult]
    took_ms: float


class VectorResponse(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizeResponse(BaseModel):
    job_id: str
    status: str
    compacted: int
    num_vectors: int
    took_ms: float


class SnapshotResponse(BaseModel):
    snapshot_id: str
    path: str


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    version: str
    collections: int


class ErrorResponse(BaseModel):
    """API_SPEC: ``{"error": "message", "code": "ERROR_CODE"}``."""

    error: str
    code: str
