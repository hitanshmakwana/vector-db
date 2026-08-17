# PyVec — A Vector Database in Python

> A single-node vector database built from scratch in Python. Implements HNSW and
> IVF-Flat approximate nearest neighbor indexes, BM25 sparse retrieval, and
> Reciprocal Rank Fusion for hybrid search. ~40–50 hour build budget.

## Documentation

- **[Architecture & Design (`docs/ARCHITECTURE.md`)](./docs/ARCHITECTURE.md)** — System architecture, storage layout, indexing mechanisms, and concurrency model.
- **[API Specification (`docs/API_SPEC.md`)](./docs/API_SPEC.md)** — Complete REST API endpoint contracts and JSON schema definitions.
- **[Architectural Decisions (`docs/DECISIONS.md`)](./docs/DECISIONS.md)** — Technical ADRs detailing key design decisions and trade-offs.
- **[Benchmark Targets & Methodology (`docs/BENCHMARKS.md`)](./docs/BENCHMARKS.md)** — Evaluation setup, benchmark goals, and methodology.
- **[Measured Results (`docs/RESULTS.md`)](./docs/RESULTS.md)** — Comprehensive measured performance numbers across SIFT-1M, MS MARCO, and concurrent load tests.

## Project at a glance

| Aspect | Choice |
|---|---|
| Language | Python 3.11+ (NumPy for hot paths) |
| ANN indexes | HNSW + IVF-Flat (both from scratch) |
| Sparse retrieval | BM25 with inverted index |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Persistence | mmap for vectors + append-only WAL for metadata |
| API | FastAPI (HTTP/JSON) |
| Topology | Single-node (no sharding, no replication) |
| Concurrency | Read-write lock, thread-safe reads |

## What you should NOT use

To keep the project honest as a "from scratch" build:

- **No FAISS, hnswlib, Annoy, ScaNN.** You are building these. Use them only as benchmark baselines.
- **No LangChain, LlamaIndex, or any RAG framework.** This is infrastructure, not an app.
- **No Qdrant/Weaviate/Chroma/Pinecone client libraries.** Same reason.

## What you CAN use

- **NumPy** — for vectorized distance computations. Writing your own SIMD in Python is silly.
- **FastAPI + Pydantic** — HTTP layer. Nobody builds their own web server for this.
- **sentence-transformers** — *only* to generate embeddings for demo/test data. The DB itself never touches this; it just stores whatever floats you give it.
- **pytest, matplotlib, tqdm** — testing and eval plumbing.

## Quickstart

```bash
pip install -e .            # core: numpy, fastapi, pydantic, uvicorn
pip install -e '.[test]'    # + pytest, httpx
```

Run the offline demo — insert, all three query paths, filter, delete, restart,
compact:

```bash
python examples/quickstart.py     # embedded, no server
python examples/http_demo.py      # the same flow over HTTP
```

### As a library

```python
from pyvec import Collection

c = Collection.create("docs", root="./data", dimension=384,
                      metric="cosine", index_type="hnsw",
                      index_params={"M": 16, "ef_construction": 200},
                      text_field="content")   # enables BM25 on the same collection

c.insert([{"id": "d1", "vector": vec, "metadata": {"content": "the quick brown fox"}}])

c.search(query_vector, k=10, params={"ef_search": 64})       # dense
c.search_text("quick fox", k=10)                             # BM25
c.search_hybrid(query_vector, "quick fox", k=10)             # RRF fusion
c.search(query_vector, k=10, filter={"category": "animals"}) # metadata filter

c.close()                                    # checkpoints; reopen with Collection.open
```

### As a server

```bash
pyvec serve --port 8080 --data-dir ./data
# or: uvicorn pyvec.api.server:app --port 8080
# or: docker compose up --build

curl -X POST localhost:8080/collections \
  -d '{"name":"t","dimension":4,"metric":"cosine","index":{"type":"hnsw"}}'
curl -X POST localhost:8080/collections/t/insert \
  -d '{"items":[{"id":"a","vector":[1,0,0,0]}]}'
curl -X POST localhost:8080/collections/t/query \
  -d '{"vector":[1,0,0,0],"k":3}'
```

Full endpoint list in [API_SPEC.md](./docs/API_SPEC.md). There is also a CLI
(`pyvec ls`, `pyvec query`, `pyvec hybrid`, ...) and a zero-dependency Python
client (`pyvec.client.PyVecClient`).

### Tests and benchmarks

```bash
pytest -m "not slow"                                  # 486 tests, ~3.5 min
pytest                                                # + the 10k-vector recall check
python -m benchmarks.sift_1m --synthetic --n 20000    # harness check, no download
python -m benchmarks.sift_1m                          # the real thing (~500MB)
python -m benchmarks.compare_vectordbs --n 50000      # head-to-head vs ChromaDB
python -m benchmarks.plot_pareto                      # CSV -> plots
```

See [benchmarks/README.md](./benchmarks/README.md) for the full set and what to
expect from each.

## Directory layout

```
vector-db/
├── docs/                        # Complete project documentation and design specs
│   ├── API_SPEC.md              # REST API endpoint specification
│   ├── ARCHITECTURE.md          # Architecture, storage, concurrency, lifecycle
│   ├── BENCHMARKS.md            # Benchmark targets and evaluation methodology
│   ├── DECISIONS.md             # ADRs (Architectural Decision Records)
│   ├── LEARNING.md              # Core algorithm theories and mental models
│   ├── NOTES.md                 # Development journal and optimization notes
│   ├── PRD.md                   # Product requirements document
│   ├── PROJECT_COMPLETE.md      # Comprehensive standalone project guide
│   ├── PROJECT_PLAN.md          # Development milestone schedule
│   ├── RESULTS.md               # Measured experimental results & logs
│   └── RESUME.md                # Resume bullet points & interview guide
├── pyvec/                       # Core database package
│   ├── api/                     # FastAPI app, routing, request/response schemas
│   │   ├── schemas.py
│   │   └── server.py
│   ├── core/                    # Engine internals, collection manager, lock, types
│   │   ├── collection.py        # Top-level Collection (WAL ordering, checkpointing)
│   │   ├── collection_manager.py# Multi-collection registry and lifecycle
│   │   ├── distance.py          # Vectorized distance metrics (L2, Cosine, Dot)
│   │   ├── errors.py            # Typed domain errors
│   │   ├── kmeans.py            # K-means++ clustering for IVF centroids
│   │   ├── rwlock.py            # Writer-preferring reader-writer lock
│   │   ├── tokenize.py          # Fast tokenizer for text processing
│   │   └── types.py             # Data types, protocols, and enums
│   ├── fusion/                  # Hybrid search fusion algorithms
│   │   └── rrf.py               # Reciprocal Rank Fusion (RRF)
│   ├── indexes/                 # Vector and lexical indexing implementations
│   │   ├── bm25.py              # Inverted index with vectorized BM25 scoring
│   │   ├── flat.py              # Exact brute-force scan (ground truth oracle)
│   │   ├── hnsw.py              # Malkov & Yashunin (2016) HNSW graph index
│   │   └── ivf.py               # IVF-Flat index with coarse quantization
│   ├── storage/                 # Persistence and durability subsystems
│   │   ├── mmap_store.py        # Memory-mapped contiguous float32 vector store
│   │   └── wal.py               # Append-only CRC32 checksummed write-ahead log
│   ├── cli.py                   # PyVec CLI interface
│   ├── client.py                # Zero-dependency Python HTTP SDK client
│   └── __init__.py
├── benchmarks/                  # Benchmark harness, datasets, scripts & results
│   ├── datasets/                # Cached benchmark datasets
│   ├── plots/                   # Generated Pareto curves and latency charts
│   ├── results/                 # Raw measured CSV benchmarks & run environment logs
│   ├── compare_vectordbs.py     # Comparison against ChromaDB & Qdrant
│   ├── hybrid_msmarco.py        # MS MARCO hybrid evaluation & fusion sweeps
│   ├── load_test.py             # High-concurrency FastAPI load testing
│   ├── persistence.py           # Durability trade-off measurements
│   ├── plot_pareto.py           # Pareto visualization generation
│   ├── sift_1m.py               # SIFT-1M HNSW and IVF evaluation
│   ├── smoke_test.py            # End-to-end HTTP smoke test
│   └── startup.py               # Cold startup and recovery benchmarks
├── examples/                    # Runnable code examples
│   ├── quickstart.py            # Embedded Python API demo
│   └── http_demo.py             # Client SDK over HTTP demo
├── tests/                       # Complete test suite (490 unit & integration tests)
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Persistent volume demo deployment
├── pyproject.toml               # Python package configuration and dependencies
└── README.md                    # Main project overview and entry point
```

## Success = you can honestly say all of these on your resume

- "Implemented HNSW and IVF-Flat from scratch in Python; achieved 95%+ recall@10 vs. brute force on SIFT-1M."
- "Built hybrid retrieval combining BM25 sparse search with dense vectors via Reciprocal Rank Fusion."
- "Designed mmap-backed vector storage with append-only WAL for crash recovery."
- "Benchmarked against FAISS; within 3× QPS at matched recall."

If you can't say those things when you're done, the project isn't finished.

### Status of those claims — measured

The benchmarks have been run. **[RESULTS.md](./docs/RESULTS.md) has every number with
the command that produced it.** Two of the four aspirational claims above survived
contact with the data; two did not.

| Claim | Status |
|---|---|
| HNSW + IVF-Flat implemented from scratch | **Done.** 490 tests: recall vs. brute force, graph invariants, layer-0 reachability, level distribution vs. its closed form. |
| **95%+ recall@10 on SIFT-1M** | **PASS — 0.9619** at `M=16, ef_construction=200, ef_search=64`, on the full 1,000,000 vectors. |
| Crash-safe mmap + WAL persistence | **PASS.** Real `kill -9` mid-insert, torn/corrupt WAL tails, crashes staged inside the checkpoint window. 1M vectors queryable in 2.93s. |
| Hybrid retrieval via RRF | **Built and working, but see below** — on real MS MARCO, unweighted RRF scored *below* dense-only. |
| ~~Within 3× of FAISS QPS~~ | **MISS — 14.1× at matched recall** (11.8×–15.0× across the sweep). Recall matches FAISS (0.9619 vs 0.9639, 0.3 points mean absolute across six operating points); throughput does not. |

**The recall parity is the claim that matters** — it says the algorithm is
implemented correctly, which is what a from-scratch project sets out to show.
The throughput gap says PyVec is written in Python: the graph walk does a dict
lookup and a heap operation per hop, and only the per-frontier distance batch is
vectorisable (it already is). That gap is structural, not a missing optimisation.

Three things worth knowing before you re-run any of it:

- **Recall depends heavily on the dataset and the scale.** On real SIFT at
  `ef_search=64`: 0.9619 at 1M, 0.9904 at 100k. On random Gaussian vectors — the
  hardest case, with high intrinsic dimensionality and no cluster structure for the
  long edges to exploit — the same parameters give ~0.87. Don't quote one as another.
- **Budget hours, not minutes, for the build.** HNSW inserts are inherently
  sequential: **11,579s (3.2h) for 1M vectors** against FAISS's 510s, and 1011s for
  100k. Build scales 11.45× for 10× the data, confirming O(N log N). `--n` works at
  smaller scale while iterating.
- **The HNSW-vs-IVF answer depends on scale.** At 100k, HNSW led IVF by only 1.1× at
  matched recall; at 1M it is **4.6×**, because posting lists grow linearly with N
  while graph search grows logarithmically. Benchmarking only at a convenient scale
  would have given the opposite conclusion. IVF's counterweight: it builds **284×
  faster** at 1M.
- **Hybrid search is not automatically a win.** It helps when the two retrievers are
  complementary *and* comparably strong. On MS MARCO, where the dense model clearly
  beats BM25, unweighted RRF averages the strong side down. See
  [RESULTS.md](./docs/RESULTS.md) for the diagnosis and what weighted fusion would give.
