# Benchmarks

Implements the five benchmarks in [BENCHMARKS.md](../docs/BENCHMARKS.md). Results are
written to `results/*.csv`; plots are generated from those CSVs and written to
`plots/`.

## Install the extras

Core PyVec needs only NumPy + FastAPI. Benchmarking needs more:

```bash
pip install -e '.[demo]'     # h5py (datasets), matplotlib (plots), pandas, tqdm
pip install -e '.[bench]'    # faiss-cpu, the baseline for benchmark 1
```

Neither is required to *run* the scripts. Missing `faiss` skips the baseline and
says so; missing `matplotlib` prints ASCII charts instead of writing PNGs;
missing `h5py` is only a problem if you want the real datasets.

## Quick check without downloading 500MB

Every dataset-backed script takes `--synthetic`, which substitutes a seeded
stand-in. Use it to verify the harness end to end:

```bash
python -m benchmarks.sift_1m --synthetic --n 20000 --queries 100
python -m benchmarks.hybrid_msmarco --synthetic --passages 5000
python -m benchmarks.plot_pareto
```

Synthetic numbers are labelled as synthetic in the CSV notes and in stdout. They
are **not** results — at small `n` recall saturates at 1.0 for every parameter,
because ANN difficulty scales with dataset size.

## The real runs

```bash
# Benchmarks 1 + 2: recall-QPS Pareto on SIFT-1M, vs FAISS. Downloads ~500MB.
python -m benchmarks.sift_1m

# Benchmark 3: hybrid vs dense vs BM25 on MS MARCO passages.
pip install datasets
python -m benchmarks.hybrid_msmarco --passages 100000

# Benchmark 4: durability vs throughput.
python -m benchmarks.persistence --n 100000

# Benchmark 5: startup time. This is the PRD NF3 test.
python -m benchmarks.startup --n 1000000 --index hnsw

# Secondary dataset (first on the cut list per PROJECT_PLAN).
python -m benchmarks.glove

# Turn every CSV into plots.
python -m benchmarks.plot_pareto
```

**Budget a few hours for `sift_1m`.** The HNSW build is the slow part: inserts are
inherently sequential (each node routes through the graph the previous ones
built), and in Python that costs milliseconds per vector. Use `--n` to work at a
smaller scale while iterating, and `--skip` to avoid rebuilding an index you have
already measured.

## What to expect

From BENCHMARKS.md, stated in advance so the results can disappoint honestly:

| | Expectation |
|---|---|
| HNSW recall@10 | ≥95% at `M=16, ef_construction=200, ef_search=64` |
| IVF recall@10 | ≥90% at `nlist=256, nprobe=16` |
| PyVec vs FAISS QPS | FAISS faster by 2–5× at matched recall (C++ with SIMD vs Python) |
| PyVec vs FAISS recall | Within a couple of points at matched parameters |
| HNSW vs IVF | HNSW's Pareto curve above and to the right on SIFT-1M |
| Hybrid vs dense | +3–8 nDCG@10 points on MS MARCO |
| Startup | 1M vectors ready to query in <30s |

**Recall is the correctness claim; QPS is the language claim.** If recall lands
where it should, the algorithm is right. If recall trails FAISS by more than ~10
points, something is genuinely wrong — check neighbour selection first, then the
level distribution.

## Smoke test — is the deployment alive and correct?

```bash
python -m benchmarks.smoke_test                        # spawns its own server
python -m benchmarks.smoke_test --url http://host:8080 # test a live one
```

Runs in ~1.5s and exits non-zero on the first broken guarantee. Not a substitute
for `pytest` — this is the check you run against a *freshly started server*, after a
deploy, or in CI before a release: the documented happy path plus the error
contract, over real HTTP, in the shape a user meets it.

Covers liveness, collection lifecycle, all three query paths, per-retriever RRF
ranks, metadata filtering, get/delete, every API_SPEC error code
(`COLLECTION_NOT_FOUND`, `ID_NOT_FOUND`, `ID_EXISTS`, `INVALID_DIMENSION`),
optimize, snapshot, upsert, and a server restart. Each check prints what it
verified, so a failure names the guarantee that broke.

## Load test — what happens under concurrent traffic?

```bash
python -m benchmarks.load_test                                # spawns a server
python -m benchmarks.load_test --url http://host:8080         # hit a live one
python -m benchmarks.load_test --concurrency 1 2 4 8 16 32 64
```

Measures the **deployed system** — FastAPI, JSON, the reader-writer lock, the
thread pool — rather than the index in isolation, which is what `sift_1m.py` does.
Sweeps concurrency across `query`, `query/text`, `query/hybrid`, filtered dense
search, and a mixed read/write run, reporting RPS, p50/p95/p99, error counts and
scaling efficiency against the single-client baseline.

Two things worth knowing about how it is built:

- **It uses HTTP keep-alive, one connection per worker.** This is a correctness
  requirement, not a tuning choice. With a fresh connection per request, sockets
  pile up in `TIME_WAIT` and exhaust the OS ephemeral port range — on Windows that
  is `WinError 10048`, and throughput collapses. Those are *client-side* failures,
  and a load test that reports them as server errors is worse than none. Fixing it
  also cut measured p95 by ~6x, because TCP setup had been dominating the number.
- **HTTP status errors and transport errors are counted separately.** A 5xx is the
  server failing; a connection reset is usually the harness or the OS. One combined
  number conflates a real defect with an artifact.

**Do not expect linear scaling.** Dense search holds the GIL for the interpreted
parts of the graph walk and releases it only inside NumPy, so throughput plateaus
while latency grows with client count. BM25, after vectorisation, spends most of its
time in NumPy and does scale. A concurrent writer visibly costs readers throughput —
that is the write lock serialising, not a bug.

## Head-to-head against production vector databases

```bash
pip install -e '.[compare]'                          # chromadb, qdrant-client
python -m benchmarks.compare_vectordbs --n 50000 --queries 500
python -m benchmarks.compare_vectordbs --synthetic --n 10000   # quick, no download
python -m benchmarks.compare_vectordbs --systems pyvec chromadb
```

Compares PyVec against **ChromaDB** on identical data with identical HNSW parameters
(`M=16, ef_construction=200`) and the same exact ground truth.

**Why ChromaDB and not Pinecone or Milvus.** The comparison only means something if
the systems are in the same category. ChromaDB is embedded, Python, single-node, and
HNSW-backed — PyVec's exact peer, and what people actually reach for when building a
RAG prototype. Pinecone is a managed cloud service, so every measurement would be
dominated by network round-trip; Milvus is a distributed system requiring etcd and
object storage. Benchmarking against either would measure deployment topology rather
than algorithms.

**Qdrant is included but its embedded mode is not a valid HNSW comparison.** Qdrant's
local mode performs **exact brute-force search** — the client itself warns that
`search_params` has no effect — so those rows are labelled `qdrant-exact` and
excluded from the summary. For a real comparison, run a server and point at it:

```bash
docker run -p 6333:6333 qdrant/qdrant
python -m benchmarks.compare_vectordbs --qdrant-url http://localhost:6333
```

**Two things this script is careful about**, because both are easy ways to
manufacture a meaningless result:

1. **It compares at matched recall.** Recall and QPS trade against each other
   continuously, so a single number that pins neither is meaningless. An earlier
   version reported "ChromaDB is 1.0x faster" by comparing each system's fastest
   configuration above 95% recall — which put PyVec at 0.9756 against Chroma at
   0.9994. Same throughput, very different quality.
2. **Threading differences are reported, not hidden.** PyVec is single-threaded by
   construction; the other libraries may use internal thread pools their client APIs
   give no way to disable. Where that is true it appears as a caveat on the row.

## Files

| File | Purpose |
|---|---|
| `compare_vectordbs.py` | Head-to-head vs ChromaDB / Qdrant |
| `harness.py` | `BenchmarkRun`, latency percentiles, recall/nDCG/MRR, ground truth with caching |
| `datasets.py` | Pinned dataset URLs, download cache, synthetic stand-ins |
| `sift_1m.py` | Benchmarks 1 and 2 |
| `glove.py` | Same sweep on GloVe-100 |
| `hybrid_msmarco.py` | Benchmark 3 |
| `persistence.py` | Benchmark 4 |
| `startup.py` | Benchmark 5 |
| `smoke_test.py` | End-to-end deployment check over HTTP |
| `load_test.py` | Concurrency / throughput / latency under load |
| `plot_pareto.py` | CSV → PNG (and ASCII) |

Measured output from all of these lives in [../docs/RESULTS.md](../docs/RESULTS.md).

## Measurement discipline

Baked into the harness, not left to each script:

- **Seeds fixed** everywhere (`--seed`, default 42).
- **Warm-up excluded** — 100 queries before measurement, so mmap pages are
  faulted in and caches are warm. Cold-start is benchmark 5's subject.
- **Per-query latency**, never per-batch, so percentiles mean something.
- **Nearest-rank percentiles**, not interpolated: a reported p99 is a real
  measurement, not an average of two.
- **Single-threaded**, including FAISS (`omp_set_num_threads(1)`). Comparing a
  parallel C++ index against a serial Python one proves nothing.
- **Ground truth cached** to `datasets/gt_*.npy` — computed once by brute force,
  reused across runs.
- **Environment recorded** alongside every CSV in `*.env.json`: CPU, platform,
  Python and NumPy versions. A QPS number without a machine is uninterpretable.
- **Skips are reported**, never silent. A missing baseline appears as a note in
  the CSV sidecar and on stderr.
