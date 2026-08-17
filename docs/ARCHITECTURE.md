# Architecture

## System overview

```
                         ┌──────────────────────────────┐
                         │        HTTP Client           │
                         │  (curl, Python SDK, RAG app) │
                         └──────────────┬───────────────┘
                                        │  JSON over HTTP
                                        ▼
                         ┌──────────────────────────────┐
                         │      API Layer (FastAPI)     │
                         │   /collections /insert /query│
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │      Collection Manager      │
                         │  routes ops to right collection
                         └──────────────┬───────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
    ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
    │   Dense Index     │   │   Sparse Index    │   │  Fusion Layer     │
    │  (HNSW or IVF)    │   │      (BM25)       │   │      (RRF)        │
    └────────┬──────────┘   └────────┬──────────┘   └───────────────────┘
             │                       │
             ▼                       ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                     Storage Layer                             │
    │  ┌────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
    │  │  Vector Store  │  │  Metadata Store │  │      WAL       │ │
    │  │  (numpy.memmap)│  │  (JSON on disk) │  │ (append-only)  │ │
    │  └────────────────┘  └─────────────────┘  └────────────────┘ │
    └───────────────────────────────────────────────────────────────┘
```

## Components

### 1. API Layer (`pyvec.api`)

FastAPI application. Thin — no business logic here, just:

- Request validation via Pydantic schemas
- Serialization/deserialization
- Error mapping (domain exception → HTTP status)
- Request logging

**Design choice:** one endpoint per operation, RESTful-ish. Not GraphQL, not
gRPC. HTTP/JSON is the pragmatic choice for a demo project — every language
can consume it, and the payload sizes here (top-k results) don't justify
protobuf.

See [API_SPEC.md](./API_SPEC.md) for endpoint details.

### 2. Collection Manager (`pyvec.core.collection`)

A `Collection` is the top-level object. It owns:

- A **dense index** (HNSW or IVF, chosen at create time)
- Optionally, a **sparse index** (BM25) if metadata has a text field
- A **vector store** (the raw float32 blobs)
- A **metadata store** (per-id metadata dicts)
- A **WAL** for durability
- A **read-write lock** for concurrency

The manager (`pyvec.core.collection_manager`) is a registry of open
collections. It handles create/drop/load-on-startup.

### 3. Dense Index

Interface:

```python
class DenseIndex(Protocol):
    def add(self, ids: list[int], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int, **params) -> list[tuple[int, float]]: ...
    def remove(self, ids: list[int]) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

Two implementations:

**HNSW (`pyvec.indexes.hnsw`)**
- In-memory graph, one adjacency list per layer per node.
- Layer 0: dense connectivity (`2M` neighbors max).
- Layers 1+: sparser (`M` neighbors max).
- Persistence: serialize adjacency structure via pickle or a custom binary
  format. HNSW graphs are pointer-heavy; simple pickle is fine for the
  learning goal.

**IVF-Flat (`pyvec.indexes.ivf`)**
- `nlist` centroids from k-means++.
- Posting lists: `dict[centroid_id, list[vector_id]]`.
- Query: L2-scan centroids, pick top `nprobe`, brute-force scan their
  posting lists.
- Persistence: centroids as `.npy`, posting lists as JSON. Trivial.

Both indexes call into `pyvec.core.distance` for actual distance
computations. This is where NumPy vectorization matters most — a naive
Python loop over 1M vectors is ~1000× slower than the NumPy version.

### 4. Sparse Index (`pyvec.indexes.bm25`)

Classic inverted index over a designated text field in metadata.

Data structures:
- `postings: dict[str, list[tuple[doc_id, term_freq]]]`
- `doc_lens: dict[doc_id, int]`
- `avg_doc_len: float`
- `num_docs: int`
- `idf_cache: dict[str, float]` — computed on first use per term

Tokenization: lowercase → strip punctuation → whitespace split. That's it.
Skip stemming and stopword removal for v1; both are configurable in
production systems.

BM25 params: `k1 = 1.5`, `b = 0.75`. Expose them but default to these.

### 5. Fusion Layer (`pyvec.fusion.rrf`)

Given `list[dense_result]` and `list[bm25_result]`, both ordered by rank:

```python
def rrf(rankings: list[list[int]], k: int = 60, top_k: int = 10) -> list[int]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=scores.get, reverse=True)[:top_k]
```

That's the whole fusion component. RRF is one function. Its power is that
it doesn't require score normalization across the different rankers.

### 6. Storage Layer

Three concerns, three files per collection.

**Vector store — `vectors.bin`**
- Contiguous float32 array, `num_vectors * dim * 4` bytes.
- Accessed via `numpy.memmap(shape=(N, dim), dtype='float32')`.
- Fixed capacity chosen at create time, doubled on overflow (with rewrite).
- Vector at row `i` is the vector for internal id `i`.
- **Deletes are tombstones** — mark the row as free in a bitmap, don't
  compact. Compaction happens on explicit `optimize()` call (P2 feature).

**Metadata store — `metadata.json`**
- One JSON object per doc, keyed by internal id.
- Loaded fully into RAM on startup. Metadata is small.
- Written back on flush (dirty tracking, not every write).

**WAL — `wal.log`**
- Append-only binary file.
- Each entry: `[op_type: u8, timestamp: u64, payload_len: u32, payload: bytes]`
- Op types: `INSERT`, `DELETE`, `CREATE_INDEX`.
- fsynced after each entry (durability > throughput for a learning project).
- Truncated after successful snapshot / checkpoint.

## Data flow

### Insert path

```
Client sends POST /collections/{name}/insert
  │
  ▼
FastAPI validates payload
  │
  ▼
Collection acquires WRITE lock
  │
  ▼
Vector normalized (if metric == cosine)
  │
  ▼
WAL entry appended + fsynced
  │
  ▼
Vector written to mmap store at next free row
  │
  ▼
Metadata stored in in-memory dict, marked dirty
  │
  ▼
Dense index.add() → for HNSW: insert into graph
                    for IVF: mark for next rebuild (or online-assign to nearest centroid)
  │
  ▼
BM25 index.add() → tokenize text field, update postings
  │
  ▼
Release lock, return 201
```

**Note on IVF and incremental inserts:** true IVF assumes centroids are
fixed at build time. For online inserts, you assign new vectors to the
nearest existing centroid. Recall degrades over time as data distribution
drifts from centroids — user must periodically call `optimize()` to rerun
k-means. Document this loudly.

### Query path (hybrid)

```
Client sends POST /collections/{name}/query with {vector, text, k}
  │
  ▼
Collection acquires READ lock
  │
  ├─────────────────────┬──────────────────────┐
  ▼                     ▼                      │
Dense search           BM25 search             │
(HNSW or IVF)          (inverted index)        │
Returns [(id, dist)]   Returns [(id, score)]   │
  │                     │                      │
  └──────────┬──────────┘                      │
             ▼                                 │
      RRF fusion                               │
             │                                 │
             ▼                                 │
   Apply metadata filter (if any)              │
             │                                 │
             ▼                                 │
   Hydrate results with metadata               │
             │                                 │
             ▼                                 │
   Release lock, return top-k                  │
```

**On filter placement:** we apply filter *after* fusion, not before. This is
easier to implement but can produce fewer than `k` results if filters are
restrictive. Real systems (Qdrant) do pre-filtering with clever graph
traversal modifications; that's a v2.

### Startup / recovery path

```
Process starts
  │
  ▼
Scan data/ directory for collections
  │
  ▼
For each collection:
   ├─ Load metadata.json into RAM
   ├─ mmap vectors.bin
   ├─ Load dense index from disk
   ├─ Rebuild BM25 index in memory from metadata (fast, ~seconds for 1M docs)
   └─ Replay WAL entries after last checkpoint
  │
  ▼
Ready to serve
```

## Concurrency model

**Single writer, many readers** per collection. Enforced by a
`threading.RWLock` (implemented via `readerwriterlock` package or roll your
own with a `Condition`).

- Reads (search, get) take the read lock. Multiple concurrent.
- Writes (insert, delete) take the write lock. Exclusive.
- The API layer is `async def` but the collection layer is synchronous.
  FastAPI runs handlers in a thread pool for blocking calls; this is fine
  for our load levels.

**Why not lock-free structures?** Because you have 50 hours and this is
Python. The GIL already limits you. A simple RW lock is correct and easy to
reason about.

## What's deliberately simple

- **No query planner.** Query type is chosen by which endpoint was hit.
- **No caching layer.** Rely on OS page cache for mmap files.
- **No auth.** Trust the network.
- **No compaction/vacuum.** Deleted vectors leave tombstones. Fine at
  demo scale.

Every one of these is a valid interview follow-up. Have the answer ready.
