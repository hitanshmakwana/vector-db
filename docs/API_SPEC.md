# API Specification

HTTP/JSON via FastAPI. Base URL: `http://localhost:8080`.

## Conventions

- All request/response bodies are JSON.
- Errors follow: `{"error": "message", "code": "ERROR_CODE"}`.
- Vectors are JSON arrays of numbers (parsed to float32 server-side).
- IDs are strings client-side; internally mapped to a monotonic uint64.

---

## Collections

### `POST /collections`

Create a new collection.

**Request**
```json
{
  "name": "docs",
  "dimension": 384,
  "metric": "cosine",
  "index": {
    "type": "hnsw",
    "params": {"M": 16, "ef_construction": 200}
  },
  "text_field": "content",
  "capacity": 100000
}
```

- `metric`: one of `cosine`, `l2`, `dot`.
- `index.type`: `hnsw`, `ivf`, or `flat` (brute force, for correctness testing).
- `text_field` (optional): if present, BM25 index is built over this field.
- `capacity`: initial vector store size. Auto-doubles on overflow.

**Response** `201 Created`
```json
{"name": "docs", "created_at": "2026-08-10T12:00:00Z"}
```

---

### `GET /collections`

List all collections.

**Response**
```json
{
  "collections": [
    {"name": "docs", "num_vectors": 12500, "dimension": 384}
  ]
}
```

---

### `GET /collections/{name}`

Collection details and stats.

**Response**
```json
{
  "name": "docs",
  "dimension": 384,
  "metric": "cosine",
  "index": {"type": "hnsw", "params": {"M": 16, "ef_construction": 200}},
  "text_field": "content",
  "num_vectors": 12500,
  "num_deleted": 42,
  "memory_bytes": 21504000,
  "disk_bytes": 24000000
}
```

---

### `DELETE /collections/{name}`

Drop a collection. Removes on-disk files.

**Response** `204 No Content`

---

## Vectors

### `POST /collections/{name}/insert`

Insert one or many. Batched.

**Request**
```json
{
  "items": [
    {
      "id": "doc-1",
      "vector": [0.12, -0.45, ...],
      "metadata": {"content": "the quick brown fox", "category": "animals"}
    },
    {
      "id": "doc-2",
      "vector": [0.33, 0.11, ...],
      "metadata": {"content": "jumps over the lazy dog", "category": "animals"}
    }
  ]
}
```

**Response** `201 Created`
```json
{"inserted": 2, "duplicates_skipped": 0}
```

**Behavior**
- Inserting an existing `id` returns `409 Conflict` (or `upsert=true` param to overwrite).
- Vector dimension must match the collection's `dimension` or `400 Bad Request`.

---

### `POST /collections/{name}/query`

Vector search (dense only).

**Request**
```json
{
  "vector": [0.12, -0.45, ...],
  "k": 10,
  "filter": {"category": "animals"},
  "params": {"ef_search": 64}
}
```

- `params` overrides index-specific search knobs (HNSW: `ef_search`, IVF: `nprobe`).
- `filter` is a shallow-equality dict over metadata. Post-filter applied
  after ANN retrieval; may return fewer than `k` if restrictive.

**Response**
```json
{
  "results": [
    {"id": "doc-1", "score": 0.943, "metadata": {"content": "..."}},
    {"id": "doc-7", "score": 0.891, "metadata": {"content": "..."}}
  ],
  "took_ms": 3
}
```

`score` is the distance metric's raw value. For `cosine`/`dot`: higher is
better. For `l2`: lower is better. Documented; the client should know.

---

### `POST /collections/{name}/query/text`

BM25 search over the collection's `text_field`.

**Request**
```json
{"text": "quick fox", "k": 10, "filter": {"category": "animals"}}
```

**Response**
```json
{
  "results": [
    {"id": "doc-1", "score": 2.34, "metadata": {"content": "..."}}
  ],
  "took_ms": 1
}
```

---

### `POST /collections/{name}/query/hybrid`

Hybrid search (dense + BM25 + RRF).

**Request**
```json
{
  "vector": [0.12, -0.45, ...],
  "text": "quick fox",
  "k": 10,
  "filter": {"category": "animals"},
  "params": {
    "ef_search": 64,
    "dense_candidates": 50,
    "sparse_candidates": 50,
    "rrf_k": 60
  }
}
```

- `dense_candidates`, `sparse_candidates` — how many results to fetch from
  each side before fusion. Both default to `10 * k`.
- `rrf_k` — the RRF constant. Default 60.

**Response**
```json
{
  "results": [
    {
      "id": "doc-1",
      "rrf_score": 0.0325,
      "dense_rank": 1,
      "sparse_rank": 3,
      "metadata": {"content": "..."}
    }
  ],
  "took_ms": 5
}
```

Note: no single similarity score is returned since RRF operates on ranks.
The individual ranks from each retriever are included for debuggability.

---

### `GET /collections/{name}/vectors/{id}`

Fetch a stored vector and its metadata.

**Response**
```json
{
  "id": "doc-1",
  "vector": [0.12, -0.45, ...],
  "metadata": {"content": "...", "category": "animals"}
}
```

`404 Not Found` if the id doesn't exist or has been deleted.

---

### `DELETE /collections/{name}/vectors/{id}`

Delete by id. Tombstoned, not physically removed until `optimize()`.

**Response** `204 No Content`

---

## Maintenance

### `POST /collections/{name}/optimize`

Rebuild the index, compacting tombstoned entries.

**Response** `202 Accepted`
```json
{"job_id": "opt-abc123", "status": "running"}
```

For a learning project this can be synchronous and just block. Async
version is P2.

---

### `POST /collections/{name}/snapshot`

Create an on-disk snapshot. Truncates WAL after success.

**Response** `201 Created`
```json
{"snapshot_id": "2026-08-10T12-00-00", "path": "/data/docs/snapshots/..."}
```

---

## Health & meta

### `GET /health`

**Response**
```json
{"status": "ok", "uptime_s": 12345}
```

### `GET /metrics`

Prometheus-format metrics. Counters for inserts, queries, errors.
Histograms for query latency. This is P1 — implement if time.

---

## Error codes

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `INVALID_DIMENSION` | Vector dim doesn't match collection |
| 400 | `INVALID_METRIC` | Unknown distance metric |
| 400 | `INVALID_INDEX_TYPE` | Unknown index type |
| 404 | `COLLECTION_NOT_FOUND` | No such collection |
| 404 | `ID_NOT_FOUND` | No such id (or tombstoned) |
| 409 | `ID_EXISTS` | Insert of existing id without `upsert=true` |
| 413 | `PAYLOAD_TOO_LARGE` | Batch exceeds 1000 items |
| 500 | `INTERNAL_ERROR` | Unhandled; log and return |

---

## Client SDK sketch (Python)

Not required but nice to have. Thin wrapper:

```python
from pyvec.client import PyVecClient

c = PyVecClient("http://localhost:8080")
c.create_collection("docs", dimension=384, metric="cosine", index="hnsw", text_field="content")
c.insert("docs", items=[{"id": "d1", "vector": vec, "metadata": {"content": "..."}}])
results = c.hybrid("docs", vector=q_vec, text="quick fox", k=10)
```

---

## What's not in the API

Deliberately absent:

- **Batch delete** by filter. Add if a benchmark needs it; otherwise no.
- **Update** (change vector but keep id). Delete + insert is fine.
- **Aggregations** (count by category, etc). This is a search engine, not
  an analytics engine.
- **Time-travel queries.** Snapshots only.
