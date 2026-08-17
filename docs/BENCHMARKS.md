# Benchmarks

Numbers are the difference between "I built a vector DB" and "I built a
vector DB that works." The plots you commit are the résumé.

## Guiding principles

- **Reproducible.** Fix seeds. Pin dataset URLs and versions. A benchmark
  that gives different numbers on different runs is worthless.
- **Honest.** Compare against strong baselines (FAISS). Do not benchmark
  against your own brute force and call it fast.
- **Multi-point.** A single number is a marketing claim. A Pareto curve is
  an engineering result.
- **Percentiles over means.** p50 tells the median story. p99 tells the
  tail-latency story. Report both.

---

## Datasets

### SIFT-1M (primary)

- 1,000,000 vectors, 128 dimensions, L2 distance
- 10,000 query vectors with pre-computed 100-NN ground truth
- The standard ANN benchmark for over a decade
- Get it from the ANN-Benchmarks repo (`sift-128-euclidean.hdf5`)

### GloVe-100 (secondary)

- 1,183,514 word embeddings, 100 dimensions, cosine
- Represents the "text embedding" domain more honestly than SIFT
- Same source

### MS MARCO passages subset (hybrid eval)

- Take first 100k passages from MS MARCO v1
- Embed with `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Use the standard dev query set (~7k queries)
- Ground truth: MS MARCO relevance judgments

---

## Metrics

### Retrieval quality

- **Recall@k**: `|top_k_from_index ∩ top_k_true| / k`. Average over queries.
- **nDCG@k** (for MS MARCO with graded relevance): normalized discounted
  cumulative gain. Standard IR metric.

### Performance

- **Throughput (QPS)**: single-threaded queries/sec. Measure over ≥1000
  queries after 100 warm-up.
- **Latency percentiles**: p50, p95, p99, p99.9. Measure per query, not
  per batch.
- **Index build time**: wall clock, cold cache.
- **Memory footprint**: RSS after loading collection, before any queries.
- **Startup time**: from process exec to first successful query.

---

## Benchmark 1: HNSW recall-QPS Pareto vs FAISS

**Setup**
- SIFT-1M
- Build once with `M=16, ef_construction=200` for both PyVec-HNSW and
  faiss.IndexHNSWFlat with same params
- Sweep `ef_search ∈ {16, 32, 64, 128, 256, 512}`
- For each `ef_search`, measure recall@10 and QPS

**Output:** `benchmarks/plots/hnsw_sift1m_pareto.png`

```
recall@10
1.00 |                              PyVec ●  FAISS ▲
     |                        ●
0.95 |                    ●         ▲
     |             ●         ▲
0.90 |     ●               ▲
     |  ●          ▲
0.85 |       ▲
     |  ▲
0.80 |________________________________________
     10    50    100   500   1000  5000  QPS
```

**Expected result.** FAISS will beat PyVec on QPS by 2–5× at matched recall
(they're in C++ with SIMD). PyVec should be within a few points of recall
at the same params. Anything within 5× QPS is a genuine achievement in
Python; brag about the recall parity, be honest about the throughput gap.

**Failure mode.** If PyVec recall trails FAISS by >10%, the algorithm is
wrong. Debug before proceeding — usually neighbor selection heuristic.

---

## Benchmark 2: IVF-Flat recall-QPS Pareto

**Setup**
- SIFT-1M
- Train k-means with `nlist ∈ {64, 256, 1024, 4096}` — one Pareto per nlist
- For each `nlist`, sweep `nprobe ∈ {1, 4, 16, 64, nlist/2}`

**Output:** `benchmarks/plots/ivf_sift1m_pareto.png`

**Key insight to surface.** IVF's Pareto curve is *below and to the left*
of HNSW's on this dataset. That's the point of running both. In the
writeup, quantify it: "at 95% recall, HNSW is ~4× faster than IVF-Flat on
SIFT-1M."

---

## Benchmark 3: Hybrid vs. dense-only on MS MARCO

**Setup**
- 100k passages embedded with all-MiniLM-L6-v2
- 7k dev queries
- Compare:
  - Pure dense (HNSW, ef_search=64)
  - Pure BM25
  - Hybrid (RRF, k=60, 50 candidates from each side)
- Report nDCG@10, MRR@10, Recall@100

**Expected result.** Hybrid > Dense > BM25 on nDCG. Lift for hybrid over
dense-only is typically 3–8 nDCG points on MS MARCO. If you don't see a
lift, either BM25 tokenization is broken or RRF is wired wrong.

**Output:** `benchmarks/plots/hybrid_msmarco.png` — bar chart with three
bars per metric.

---

## Benchmark 4: Persistence overhead

Measure the write path with and without WAL fsync:

- Insert 100k vectors, batched 1k at a time
- Config 1: WAL enabled, fsync per batch
- Config 2: WAL enabled, fsync per collection close
- Config 3: WAL disabled (unsafe mode)

Report insert throughput (vectors/sec) for each. This surfaces the
durability-vs-performance trade-off — good interview conversation.

---

## Benchmark 5: Startup / recovery time

- Cold start with 1M vectors already on disk
- Measure time from process start to first successful query
- Break down: load metadata (X ms), mmap vectors (Y ms), load HNSW graph (Z ms), warmup (W ms)

This directly hits NF3 in the PRD.

---

## Reference harness structure

```python
# benchmarks/harness.py
class BenchmarkRun:
    def __init__(self, name: str, seed: int = 42):
        self.name = name
        random.seed(seed); np.random.seed(seed)
        self.results = []

    def measure(self, fn, *args, iters=1, warmup=100):
        # warmup
        for _ in range(warmup):
            fn(*args)
        # measure
        latencies = []
        for _ in range(iters):
            t = time.perf_counter()
            fn(*args)
            latencies.append(time.perf_counter() - t)
        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "qps": iters / sum(latencies),
        }

    def report(self, config, metric):
        self.results.append({"config": config, **metric})

    def save(self, path):
        pd.DataFrame(self.results).to_csv(path, index=False)
```

Every benchmark uses this harness. Results go to CSV, plots read from CSV.
Reproducibility is a feature.

---

## Things you should NOT benchmark

- **PyVec vs pgvector.** Different systems, different targets. Apples/oranges.
- **PyVec vs Pinecone.** They're a managed service; you're a Python library.
- **Compilation time.** Nobody cares.
- **Cold cache with `sync; echo 3 > /proc/sys/vm/drop_caches`.** Unless
  you're doing storage-specific claims, this just adds noise.

---

## How the numbers show up in the résumé

You want to be able to write things like:

> Achieved 96% recall@10 on SIFT-1M with HNSW at 340 QPS (single-threaded
> Python), within 4× of FAISS at matched recall.

> Hybrid retrieval (BM25 + dense fused via RRF) improved nDCG@10 by 5.2
> points over pure dense on MS MARCO passages.

Not:

> Built a fast vector database.

Every claim on the résumé should be a number you can point to in your repo.
