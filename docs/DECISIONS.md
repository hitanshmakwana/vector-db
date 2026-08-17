# Architecture Decision Records

Every non-obvious choice with its rationale. Interviewers will ask "why did
you do X instead of Y?" — the answers live here.

Format: [Nygard ADR](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

---

## ADR-001: Python as the implementation language

**Status:** Accepted

**Context.** Vector databases are typically written in C++ (FAISS, hnswlib) or
Rust (Qdrant, LanceDB) for performance. Python is 50–100× slower for
tight inner loops.

**Decision.** Use Python 3.11+.

**Rationale.**

- The project's primary goal is *learning* — a language you already write
  fluently reduces friction on the ideas, which are the hard part.
- NumPy pushes the hot loops (batched distance computation, k-means)
  into C. The user-facing time budget for search is dominated by
  vectorized ops, not by Python overhead.
- Python's ecosystem (FastAPI, pytest, matplotlib, sentence-transformers)
  removes plumbing work.
- The performance gap doesn't invalidate the résumé claim. "Implemented
  HNSW from scratch in Python; within 3× of FAISS QPS" is still
  impressive.

**Consequences.** QPS numbers will be lower than production systems. This
is fine as long as *recall* matches — recall is an algorithmic correctness
claim, not a language claim. If a specific hot path becomes a bottleneck we
can drop into Cython or Numba for that function only.

---

## ADR-002: Both HNSW and IVF-Flat; skip Product Quantization

**Status:** Accepted

**Context.** ANN indexes come in many flavors. FAISS ships >20. What
subset to implement?

**Decision.** HNSW (primary) + IVF-Flat (secondary). No PQ, no OPQ, no
scalar quantization.

**Rationale.**

- **HNSW** is the industry default for the small-to-medium regime (<100M
  vectors). Every serious vector DB uses it. Not implementing it would be
  a résumé hole.
- **IVF-Flat** is the natural comparison point. It's simpler, has
  fundamentally different characteristics (build-time k-means dominates,
  query-time nprobe tunable), and gives you a real Pareto plot instead of
  a single number.
- **PQ** is where things get expensive in the >100M regime. It's a whole
  compressed-vector sub-domain with its own algorithms (asymmetric distance
  computation, ADC lookup tables). Adds 15+ hours minimum. Explicit
  non-goal per PRD N3.
- Implementing two indexes forces you to design a clean `DenseIndex`
  interface, which is itself a good software-engineering exercise.

**Consequences.** Cannot handle billion-scale collections. Cannot claim
"memory-efficient" without qualification. Interview question "how would you
scale to 1B vectors?" — answer is "add PQ + sharding," which we know how to
discuss even without implementing.

---

## ADR-003: Hybrid search via Reciprocal Rank Fusion, not weighted sum

**Status:** Accepted

**Context.** Two ways to combine dense and sparse retrieval scores:

1. **Weighted sum:** `final = α * dense_score + (1-α) * bm25_score`.
   Requires score normalization since cosine ∈ [-1,1] and BM25 ∈ [0,∞).
2. **Reciprocal Rank Fusion (RRF):** ignore scores entirely, combine by
   *rank* position: `Σ 1 / (k + rank_i)`.

**Decision.** RRF with `k=60`.

**Rationale.**

- **No score normalization headache.** BM25 scores are unbounded, cosine
  is bounded, dot product depends on norms. Normalizing across them is a
  research problem in itself.
- RRF is the default in Elasticsearch (`rrf` retriever), Vespa,
  OpenSearch. Aligning with industry standard is the right call for a
  résumé project.
- The Cormack et al. 2009 paper shows RRF beats weighted-sum baselines
  on TREC data.
- **`k=60`** is the canonical constant. Robust across datasets.

**Consequences.** Cannot expose a "boost the dense side by 2×" knob. Users
who want fine control can call the two endpoints separately and fuse
themselves. Explicit knob → deferred.

---

## ADR-004: Single-node only; no sharding, no replication

**Status:** Accepted

**Context.** Real vector DBs shard collections across nodes and replicate
shards for HA.

**Decision.** Single-node. One process, one machine.

**Rationale.**

- **The résumé already covers distributed systems** via PulseQueue (job
  orchestration with Raft-ish consensus, fault tolerance, 79 req/s @
  P95=130ms). Building another distributed system in the same résumé
  is diminishing returns.
- **Depth over breadth.** With 40–50 hours, going deep on HNSW internals
  produces a stronger interview conversation than going shallow on
  distribution.
- **Sharding is architecturally straightforward** once you have a
  single-node engine. The gap is not the design; it's implementation
  time (query routing, coordinator, rebalance protocol). Cheap to
  discuss, expensive to build.

**Consequences.** Interviewer may ask "how would you distribute this?" —
have the answer: consistent hashing over vector ids → coordinator node
fans out queries → merge top-k from each shard → apply RRF at the merge
stage. Replication via raft-lite (2N+1 nodes, leader accepts writes, followers
replicate WAL).

---

## ADR-005: mmap for vector storage, append-only WAL for metadata

**Status:** Accepted

**Context.** Where do vectors live between process restarts?

Options:
1. **Pickle** the whole thing on shutdown. Simple. Not crash-safe.
2. **SQLite** with BLOBs. Robust but slow for large batch scans and
   doesn't match how ANN algorithms actually access memory.
3. **mmap.** OS handles paging. Vectors are contiguous float32 arrays —
   perfect for mmap.

**Decision.** `numpy.memmap` for vectors + append-only WAL for structural
changes.

**Rationale.**

- **mmap matches the workload.** ANN scans read vectors sequentially or
  in scatter-gather patterns; the OS page cache handles this well.
- **Zero deserialization cost.** A vector in an mmap-ed float32 array is
  already usable by NumPy.
- **WAL gives crash safety** without paying the cost on the hot path.
  Insert appends to WAL (fsync), writes to mmap (no fsync), and returns.
  On crash we replay the WAL from the last snapshot.
- Real systems (LanceDB, DiskANN) use mmap heavily.

**Consequences.**

- Fixed-capacity file; resizing means creating a new file and copying.
  Grow by doubling to amortize.
- Deletes are tombstones; no in-place compaction. `optimize()` endpoint
  compacts later.
- On non-Linux systems mmap has quirks (Windows sparse files). Test on
  Linux first, accept limited Windows support.

---

## ADR-006: FastAPI for HTTP, no gRPC

**Status:** Accepted

**Context.** Options for the API surface:
1. **FastAPI (HTTP+JSON).** Simple. Slower serialization.
2. **gRPC.** Fast. Requires protobufs, code gen.
3. **Custom TCP protocol** (like Redis RESP).

**Decision.** FastAPI.

**Rationale.**

- Every language can talk HTTP+JSON. Any RAG demo can consume it.
- Serialization is not the bottleneck for our payload sizes (top-10
  results ≈ 10 ids + 10 floats = 200 bytes).
- Async + Pydantic → good ergonomics for a small team of one.
- gRPC is the right answer for a production system where clients are also
  under your control. Ours aren't.

**Consequences.** Larger wire payloads than gRPC. Irrelevant at demo scale.

---

## ADR-007: No third-party ANN library — implement from scratch

**Status:** Accepted

**Context.** hnswlib, FAISS, NMSLIB, annoy — all pip-installable. The
project could use one and focus on the DB shell around it.

**Decision.** Implement HNSW and IVF ourselves. Third-party libraries are
allowed only as **benchmark baselines** in `benchmarks/`.

**Rationale.**

- **This is the entire point.** "Wrapped FAISS in a FastAPI server" is
  not a résumé project.
- Implementing HNSW forces you to actually understand the paper —
  neighbor selection, layer assignment, entry point handling.
- The correctness bar is: agree with brute force on small inputs (modulo
  ties), match hnswlib within a few percent recall on SIFT-1M.

**Consequences.** ~15 hours of the budget goes to HNSW implementation
alone. Non-negotiable. If time is tight, drop P2 features from the PRD, not
this.

---

## ADR-008: sentence-transformers for embedding generation (test data only)

**Status:** Accepted

**Context.** The DB stores whatever floats you give it. It does not
generate embeddings. But the *tests and demos* need embeddings from
somewhere.

**Decision.** Use `sentence-transformers` in test/demo scripts.
Specifically `all-MiniLM-L6-v2` (384-dim, small, fast).

**Rationale.**

- Zero-effort text-to-vector for MS MARCO experiments.
- CPU-friendly.
- Widely known; interview-safe reference.
- Kept strictly outside `pyvec/` core — lives in `benchmarks/` and
  `examples/`.

**Consequences.** Adds a heavy dep (torch, transformers) for demos. Not a
core dep. `pip install pyvec[demo]` extras convention.

---

## ADR-009: Per-collection distance metric, fixed at create time

**Status:** Accepted

**Context.** Could allow the distance metric (cosine, L2, dot) to be
specified per query, or fix it per collection.

**Decision.** Per collection, at create time.

**Rationale.**

- HNSW graph structure depends on the metric. A graph built for L2 does
  not answer cosine queries correctly.
- Unit-normalizing at insert time is metric-specific — cosine wants it,
  L2 doesn't.
- Users almost never want to change metrics mid-life on the same data.
- Simplifies the API and prevents footguns.

**Consequences.** Users who want two metrics create two collections.

---

## ADR-010: Deletes are tombstones, not immediate compaction

**Status:** Accepted

**Context.** How to handle vector deletion in an HNSW graph? True deletion
requires re-wiring neighbor connections, which is expensive.

**Decision.** Soft delete. Mark the id as deleted in a bitmap. Skip
deleted ids in search results. Space is reclaimed on manual `optimize()`
call, which rebuilds the index.

**Rationale.**

- HNSW deletion is a research problem (see "FreshDiskANN" for the
  state of the art).
- Soft delete is O(1) and correct.
- Recall degrades slowly as deletions accumulate — bounded by
  `deleted_count / total_count`. Users can call `optimize()` when it
  matters.
- **Real systems do this too.** Milvus, Qdrant use soft delete + periodic
  compaction.

**Consequences.** Memory doesn't shrink until compaction. Document this.
Add an `optimize()` endpoint (P2).

---

## ADR-011: BM25 lives in the same collection as vectors, not separately

**Status:** Accepted

**Context.** Should hybrid search be a first-class feature (one collection,
one insert, both indexes updated) or a composition (two collections, client
coordinates)?

**Decision.** Same collection. Metadata schema declares which field is the
"text" field; BM25 indexes it automatically.

**Rationale.**

- One insert, one delete, one atomic operation. No client-side coordination.
- Matches how Elastic/OpenSearch do it (`dense_vector` + `text` fields in
  the same doc).
- The alternative — two collections — makes atomic updates a
  distributed-transaction problem in a single-node system. Absurd.

**Consequences.** Slight coupling between index types. The `DenseIndex`
and `SparseIndex` interfaces are still cleanly separated internally.

---

## ADR-012: Fixed float32; no float16, no int8

**Status:** Accepted

**Context.** Half-precision (float16) halves memory. Int8 quarters it, at
a small recall cost.

**Decision.** float32 only.

**Rationale.**

- Simplicity. Quantization is a whole sub-project (calibration, per-dim
  scale/offset).
- 1M × 128d × 4B = 512MB. Fits in RAM on a laptop. Not a real problem at
  our scale.
- PQ (deferred per ADR-002) is the more sophisticated compression path
  when we need it.

**Consequences.** Cannot claim memory efficiency vs. quantized systems.
Fine.

---

## Rejected alternatives (kept here so we don't relitigate)

- **"Just use Rust for the index and Python bindings"** — would be
  faster, but a Rust rewrite is a different project. Time budget forbids.
- **"Use LMDB or RocksDB for the WAL"** — more robust, but the point is to
  understand WAL, not to shell out to one. Own implementation.
- **"Support cosine + L2 in one collection via query-time metric"** —
  breaks HNSW graph invariants. See ADR-009.
- **"Auto-select HNSW vs IVF based on collection size"** — cute, but
  would need a good cost model. Explicit user choice is fine and honest.
