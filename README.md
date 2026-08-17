# PyVec

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 490 Passed](https://img.shields.io/badge/tests-490%20passed-brightgreen.svg)](./tests)
[![Architecture: Single--Node](https://img.shields.io/badge/architecture-single--node-orange.svg)](./docs/ARCHITECTURE.md)

**PyVec** is a lightweight, single-node vector database and retrieval engine built from scratch in Python and NumPy. It provides high-performance approximate nearest neighbor (ANN) search, sparse BM25 text retrieval, hybrid search via Reciprocal Rank Fusion (RRF), crash-safe persistence, and a production-ready FastAPI HTTP interface.

---

## Key Features

- **Approximate Nearest Neighbor (ANN) Indexing**
  - **HNSW (Hierarchical Navigable Small World)**: Multi-layer graph index with heuristic neighbor selection (Malkov & Yashunin, 2016) achieving **96.2% recall@10 on SIFT-1M**.
  - **IVF-Flat (Inverted File Index)**: Coarse quantization with k-means++ clustering and multi-probe bucket scanning.
  - **Exact Flat Index**: Exhaustive vector scan serving as ground-truth correctness oracle.

- **Hybrid Lexical & Semantic Retrieval**
  - **Vectorized BM25**: Sparse keyword search with inverted posting lists and NumPy `bincount` scatter-add scoring.
  - **Reciprocal Rank Fusion (RRF)**: Rank-based fusion merging dense vector embeddings and sparse lexical hits without score calibration.

- **Crash-Safe Persistence & Durability**
  - **Memory-Mapped Storage (`mmap`)**: Contiguous `float32` vector arrays with instant cold start (**1M vectors queryable in 2.93s**).
  - **CRC32 Write-Ahead Log (WAL)**: Append-only durability layer with automatic checksum verification and torn-tail recovery under unexpected process termination.

- **Concurrency & REST API**
  - **Writer-Preferring RWLock**: Thread-safe multi-reader concurrency that prevents writer starvation during heavy query loads.
  - **FastAPI HTTP Service**: REST API supporting collection lifecycle, batch inserts, dense queries, text search, hybrid retrieval, and metadata filtering.
  - **CLI & Python SDK**: Interactive command-line interface and zero-dependency Python client library.

---

## Benchmark Highlights

Measured single-threaded on Intel Core i5-10400T (Windows 11, Python 3.14 / NumPy 2.4):

| Benchmark / Workload | Metric / Target | Measured Result | Reference |
|---|---|---|---|
| **HNSW Recall vs FAISS** (SIFT-1M, $ef=64$) | Recall@10 Parity | **96.19%** vs FAISS 96.39% (0.3% MAE) | [RESULTS.md](./docs/RESULTS.md#3-benchmark-1--hnsw-recall-qps-vs-faiss-sift-1m) |
| **Startup & Recovery** (1M vectors $\times$ 128d) | Time to First Query | **2.93 seconds** (vs 30s target) | [RESULTS.md](./docs/RESULTS.md#6-benchmark-5--startup-and-recovery-prd-nf3) |
| **HTTP Load & Latency** (20k vectors behind FastAPI) | Throughput & Latency | **224.1 RPS @ 5.48 ms p95** (0 errors) | [RESULTS.md](./docs/RESULTS.md#7-load-test--the-deployed-http-surface) |
| **Durability Trade-off** (100k vectors) | Group Commit vs sync | **23,801 vec/s** vs 2,316 vec/s (10.3× lift) | [RESULTS.md](./docs/RESULTS.md#5-benchmark-4--the-cost-of-durability) |
| **BM25 Scorer Optimization** | Vectorized scatter-add | **6.7× speedup** (35.35 ms $\to$ 5.25 ms/query) | [RESULTS.md](./docs/RESULTS.md#7-load-test--the-deployed-http-surface) |

*Full evaluation reports and reproducibility steps available in [docs/RESULTS.md](./docs/RESULTS.md).*

---

## Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/hitanshmakwana/vector-db.git
cd vector-db

# Install core package
pip install -e .

# Optional: install development, testing & benchmark extras
pip install -e '.[test,demo,bench,compare]'
```

---

### Embedded Python API

```python
import numpy as np
from pyvec import Collection

# 1. Create a collection with HNSW index and BM25 sparse retrieval enabled
collection = Collection.create(
    name="documents",
    root="./data",
    dimension=128,
    metric="cosine",
    index_type="hnsw",
    index_params={"M": 16, "ef_construction": 200},
    text_field="content"  # enables BM25 on this metadata field
)

# 2. Insert records
vector = np.random.randn(128).astype("float32").tolist()
collection.insert([
    {
        "id": "doc_1",
        "vector": vector,
        "metadata": {
            "title": "Vector Databases Overview",
            "content": "Hierarchical Navigable Small World graphs enable efficient ANN search.",
            "category": "database"
        }
    }
])

# 3. Querying
# Dense vector search
dense_hits = collection.search(query_vector=vector, k=5, params={"ef_search": 64})

# Sparse BM25 text search
text_hits = collection.search_text(query_text="HNSW graph search", k=5)

# Hybrid search via Reciprocal Rank Fusion (RRF)
hybrid_hits = collection.search_hybrid(
    query_vector=vector,
    query_text="HNSW graph search",
    k=5
)

# Metadata-filtered search
filtered_hits = collection.search(
    query_vector=vector,
    k=5,
    filter={"category": "database"}
)

# Checkpoint and close
collection.close()
```

---

### Running the HTTP Server

#### Via CLI / Uvicorn

```bash
# Start server with default data directory (./data)
pyvec serve --port 8080 --data-dir ./data

# Or directly with uvicorn
uvicorn pyvec.api.server:app --host 0.0.0.0 --port 8080
```

#### Via Docker Compose

```bash
docker compose up --build
```

#### Example REST API Calls

```bash
# Health check
curl http://localhost:8080/health

# Create a collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "articles",
    "dimension": 4,
    "metric": "cosine",
    "index": {"type": "hnsw", "params": {"M": 16, "ef_construction": 200}},
    "text_field": "body"
  }'

# Insert items
curl -X POST http://localhost:8080/collections/articles/insert \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "art_1",
        "vector": [0.2, 0.8, -0.1, 0.5],
        "metadata": {"body": "Fast vector search engines with Python and NumPy."}
      }
    ]
  }'

# Query collection
curl -X POST http://localhost:8080/collections/articles/query \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.2, 0.8, -0.1, 0.5],
    "k": 5
  }'
```

---

### Using the Python Client SDK

```python
from pyvec.client import PyVecClient

client = PyVecClient(base_url="http://localhost:8080")

# Create and populate
client.create_collection("products", dimension=128, metric="l2", index_type="hnsw")
client.insert("products", [
    {"id": "p1", "vector": [0.1] * 128, "metadata": {"brand": "acme"}}
])

# Query
results = client.query("products", vector=[0.1] * 128, k=10)
for hit in results:
    print(f"ID: {hit.id}, Score: {hit.score}, Metadata: {hit.metadata}")
```

---

## Repository Structure

```
vector-db/
├── docs/                        # Technical documentation & design specifications
│   ├── API_SPEC.md              # REST API endpoint specification
│   ├── ARCHITECTURE.md          # System architecture & concurrency model
│   ├── BENCHMARKS.md            # Benchmark targets & methodology
│   ├── DECISIONS.md             # Architectural Decision Records (ADRs)
│   └── RESULTS.md               # Empirical benchmark logs & verification data
├── pyvec/                       # Core database package
│   ├── api/                     # FastAPI server, routers, and request schemas
│   ├── core/                    # Collection management, distance math, locks & types
│   ├── fusion/                  # Hybrid search Reciprocal Rank Fusion (RRF)
│   ├── indexes/                 # HNSW, IVF-Flat, BM25, and Flat index implementations
│   ├── storage/                 # Memory-mapped vector arrays & CRC-checked WAL
│   ├── cli.py                   # Command-line interface
│   └── client.py                # Zero-dependency Python SDK client
├── benchmarks/                  # Evaluation scripts, Pareto plots & load tests
├── examples/                    # Runnable library & HTTP client demonstrations
├── tests/                       # 490 unit, integration, and crash-safety tests
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Containerized deployment with persistent volume
├── pyproject.toml               # Package build configuration & dependencies
└── README.md                    # Project documentation
```

---

## Testing & Verification

Run the full automated test suite (490 test cases across index math, graph invariants, persistence, API, and crash safety):

```bash
# Run unit & integration tests
pytest -m "not slow"

# Run complete suite including 10k-vector recall validation
pytest

# Run end-to-end HTTP smoke test
python -m benchmarks.smoke_test
```

---

## Documentation

- **[Architecture & Concurrency (`docs/ARCHITECTURE.md`)](./docs/ARCHITECTURE.md)**: Deep dive into the storage engine, graph routing, lock policies, and startup recovery.
- **[REST API Specification (`docs/API_SPEC.md`)](./docs/API_SPEC.md)**: Exhaustive endpoint documentation, query schemas, and status codes.
- **[Architectural Decisions (`docs/DECISIONS.md`)](./docs/DECISIONS.md)**: Design trade-offs (e.g., NumPy vs C++ extensions, RRF vs score weighting, WAL framing).
- **[Benchmark Methodology (`docs/BENCHMARKS.md`)](./docs/BENCHMARKS.md)**: Experimental setup for SIFT-1M, MS MARCO, and throughput evaluations.
- **[Empirical Results (`docs/RESULTS.md`)](./docs/RESULTS.md)**: Verified benchmark measurements, Pareto curves, and system performance logs.

---

## License

Distributed under the MIT License. See `LICENSE` or [pyproject.toml](./pyproject.toml) for details.
