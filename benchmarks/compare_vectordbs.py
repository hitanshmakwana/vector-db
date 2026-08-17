"""Benchmark PyVec against production embedded vector databases.

Compares **PyVec**, **ChromaDB** and **Qdrant** on identical data with identical
index parameters, measuring recall@10, QPS, latency percentiles, build time and
on-disk size.

    python -m benchmarks.compare_vectordbs --n 50000 --queries 500
    python -m benchmarks.compare_vectordbs --synthetic --n 10000    # no download
    python -m benchmarks.compare_vectordbs --systems pyvec chromadb

## Why these three, and not Pinecone or Milvus

The comparison only means something if the systems are in the same category.

* **ChromaDB** — embedded, Python, single-node, HNSW under the hood. This is what
  people actually reach for when building a RAG prototype, and it is PyVec's exact
  peer. The most informative baseline here.
* **Qdrant** (local mode) — same embedded shape, but the engine is Rust. A second
  reference point for "what does a compiled implementation cost you".
* **Pinecone is deliberately excluded.** BENCHMARKS.md already says why: it is a
  managed cloud service, so every measurement would be dominated by network
  round-trip time. You would be benchmarking the internet, not the index.
* **Milvus is deliberately excluded.** It is a distributed system requiring etcd and
  object storage; comparing it to an embedded library measures deployment topology
  rather than algorithms.

## What "fair" means here, and where it breaks down

Fairness is the whole difficulty of this benchmark, so the compromises are explicit:

1. **Identical index parameters.** All three are given `M=16, ef_construction=200`
   and queried at the same `ef_search`, because all three use HNSW. Comparing at
   each library's *defaults* would measure the defaults, not the implementations.
2. **Single-threaded where it can be enforced.** PyVec is inherently
   single-threaded. Chroma and Qdrant may use internal thread pools that cannot
   always be disabled from the client API — where that is true it is **reported as a
   caveat on the row**, because an unacknowledged thread-count difference is the
   easiest way to produce a meaningless 10x.
3. **Same ground truth.** Recall for every system is measured against the same
   exact-kNN ground truth, computed once by brute force.
4. **Build time includes everything each system does on insert.** Chroma writes to
   SQLite and maintains its own metadata; PyVec writes a WAL and mmap. Those costs
   are part of what each system *is*, so they are included and noted rather than
   subtracted out.

The honest framing of any result here: PyVec is a from-scratch educational
implementation, and these are production databases with years of engineering. Being
within an order of magnitude on throughput while matching on recall is the
achievement — not winning.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.datasets import load_sift  # noqa: E402
from benchmarks.harness import (  # noqa: E402
    BenchmarkRun,
    cache_path,
    ground_truth,
    mean_recall_at_k,
    process_rss_bytes,
)

K = 10
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = [16, 32, 64, 128]


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
# PyVec
# --------------------------------------------------------------------------- #


def bench_pyvec(train, test, truth, run, args, root: Path) -> None:
    from pyvec.core.collection import MAX_BATCH, Collection

    print("\n=== PyVec ===", file=sys.stderr)
    path = root / "pyvec"
    collection = Collection.create(
        "bench", path, dimension=train.shape[1], metric="l2", index_type="hnsw",
        index_params={"M": args.M, "ef_construction": args.ef_construction},
        capacity=len(train) + 16,
        # Group commit rather than fsync-per-entry: Chroma and Qdrant both buffer
        # writes, so per-write fsync would be comparing durability policies rather
        # than index construction.
        fsync_policy="batch",
    )
    rss_before = process_rss_bytes()
    started = time.perf_counter()
    try:
        for offset in range(0, len(train), MAX_BATCH):
            chunk = train[offset : offset + MAX_BATCH]
            collection.insert(
                [
                    {"id": str(offset + i), "vector": chunk[i]}
                    for i in range(len(chunk))
                ]
            )
        collection.checkpoint()
        build_s = time.perf_counter() - started
        rss_delta = max(0, process_rss_bytes() - rss_before)
        disk = _dir_size(path)
        print(f"  build {build_s:.1f}s", file=sys.stderr)

        for ef in args.ef_search:
            latency, results = run.measure_each(
                lambda q, ef=ef: collection.search(q, k=K, params={"ef_search": ef}),
                test, warmup=min(50, len(test)), collect=True,
            )
            recall = mean_recall_at_k(
                [[int(h.id) for h in r] for r in results], truth.tolist(), K
            )
            _report(run, "pyvec", ef, recall, latency, build_s, disk, rss_delta,
                    args, threads="1 (single-threaded by construction)")
    finally:
        collection.close()


# --------------------------------------------------------------------------- #
# ChromaDB
# --------------------------------------------------------------------------- #


def bench_chromadb(train, test, truth, run, args, root: Path) -> None:
    try:
        import chromadb
    except ImportError:
        run.note("chromadb not installed — skipped. `pip install chromadb`")
        return

    print(f"\n=== ChromaDB {getattr(chromadb, '__version__', '?')} ===", file=sys.stderr)
    path = root / "chroma"
    client = chromadb.PersistentClient(path=str(path))

    # Chroma exposes HNSW knobs through collection metadata. Key names moved between
    # versions (`hnsw:*` in older releases, a nested `hnsw` dict in 1.x), so try the
    # modern form first and fall back — silently accepting defaults would mean
    # benchmarking Chroma's defaults against PyVec's explicit settings.
    configured = "explicit"
    try:
        collection = client.create_collection(
            name="bench",
            configuration={
                "hnsw": {
                    "space": "l2",
                    "max_neighbors": args.M,
                    "ef_construction": args.ef_construction,
                    "ef_search": max(args.ef_search),
                }
            },
        )
    except Exception:
        try:
            collection = client.create_collection(
                name="bench",
                metadata={
                    "hnsw:space": "l2",
                    "hnsw:M": args.M,
                    "hnsw:construction_ef": args.ef_construction,
                    "hnsw:search_ef": max(args.ef_search),
                },
            )
            configured = "legacy metadata keys"
        except Exception as exc:
            run.note(f"chromadb: could not set HNSW params ({exc}); using defaults")
            collection = client.create_collection(name="bench")
            configured = "DEFAULTS — not matched to PyVec"

    rss_before = process_rss_bytes()
    started = time.perf_counter()
    batch = 1000
    ids = [str(i) for i in range(len(train))]
    for offset in range(0, len(train), batch):
        collection.add(
            ids=ids[offset : offset + batch],
            embeddings=train[offset : offset + batch].tolist(),
        )
    build_s = time.perf_counter() - started
    rss_delta = max(0, process_rss_bytes() - rss_before)
    disk = _dir_size(path)
    print(f"  build {build_s:.1f}s  (params: {configured})", file=sys.stderr)

    # Chroma fixes ef_search at collection level in most versions, so a per-ef sweep
    # is not available through the query API. Report the single operating point.
    latency, results = run.measure_each(
        lambda q: collection.query(query_embeddings=[q.tolist()], n_results=K),
        test, warmup=min(50, len(test)), collect=True,
    )
    recall = mean_recall_at_k(
        [[int(i) for i in r["ids"][0]] for r in results], truth.tolist(), K
    )
    _report(run, "chromadb", max(args.ef_search), recall, latency, build_s, disk,
            rss_delta, args,
            threads="library-managed (not forced to 1)",
            note=f"HNSW params: {configured}")


# --------------------------------------------------------------------------- #
# Qdrant (local mode — no server)
# --------------------------------------------------------------------------- #


def bench_qdrant(train, test, truth, run, args, root: Path) -> None:
    """Qdrant, either embedded (local) or against a running server.

    **Critical caveat, and the reason this needs saying loudly:** Qdrant's *local
    mode* performs **exact brute-force search** — the client itself warns
    "`search_params` has no effect". It builds no HNSW graph. Benchmarking PyVec's
    approximate index against Qdrant's exhaustive scan would compare two different
    algorithms and produce a meaningless ratio, so local-mode rows are labelled
    `qdrant-exact` and excluded from the HNSW comparison.

    For a real HNSW comparison, point this at a server:

        docker run -p 6333:6333 qdrant/qdrant
        python -m benchmarks.compare_vectordbs --qdrant-url http://localhost:6333
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        run.note("qdrant-client not installed — skipped. `pip install qdrant-client`")
        return

    server_mode = bool(args.qdrant_url)
    if server_mode:
        print(f"\n=== Qdrant (server {args.qdrant_url}) ===", file=sys.stderr)
        client = QdrantClient(url=args.qdrant_url)
        label = "qdrant"
    else:
        print("\n=== Qdrant (local mode — EXACT search, not HNSW) ===", file=sys.stderr)
        client = QdrantClient(path=str(root / "qdrant"))
        label = "qdrant-exact"
        run.note(
            "Qdrant local mode does exact brute-force search, not HNSW. Its rows are "
            "labelled 'qdrant-exact' and are NOT a valid HNSW comparison. Use "
            "--qdrant-url against a real server for that."
        )
    path = root / "qdrant"
    dim = int(train.shape[1])

    try:
        try:
            client.delete_collection("bench")
        except Exception:
            pass
        client.create_collection(
            collection_name="bench",
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.EUCLID
            ),
            hnsw_config=models.HnswConfigDiff(
                m=args.M, ef_construct=args.ef_construction
            ),
        )
    except Exception as exc:
        run.note(f"qdrant: create_collection failed ({exc}) — skipped")
        return

    rss_before = process_rss_bytes()
    started = time.perf_counter()
    batch = 1000
    for offset in range(0, len(train), batch):
        chunk = train[offset : offset + batch]
        client.upsert(
            collection_name="bench",
            points=[
                models.PointStruct(id=offset + i, vector=chunk[i].tolist())
                for i in range(len(chunk))
            ],
        )
    build_s = time.perf_counter() - started
    rss_delta = max(0, process_rss_bytes() - rss_before)
    disk = _dir_size(path)
    print(f"  build {build_s:.1f}s", file=sys.stderr)

    for ef in args.ef_search:
        def search(q, ef=ef):
            return client.query_points(
                collection_name="bench",
                query=q.tolist(),
                limit=K,
                search_params=models.SearchParams(hnsw_ef=ef),
            ).points

        latency, results = run.measure_each(
            search, test, warmup=min(50, len(test)), collect=True
        )
        recall = mean_recall_at_k(
            [[int(p.id) for p in r] for r in results], truth.tolist(), K
        )
        _report(run, label, ef, recall, latency, build_s, disk, rss_delta, args,
                threads="library-managed (not forced to 1)",
                note=("server mode: real HNSW" if server_mode
                      else "LOCAL MODE = EXACT SEARCH, not HNSW - not comparable"))
    client.close()


# --------------------------------------------------------------------------- #


def _report(run, system, ef, recall, latency, build_s, disk, rss, args,
            threads="", note=""):
    row = run.report(
        {
            "system": system,
            "n": args._n,
            "dim": args._dim,
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_search": ef,
            "build_s": round(build_s, 2),
            "disk_bytes": disk,
            "rss_delta_bytes": rss,
            "threading": threads,
            "caveat": note,
        },
        {f"recall@{K}": round(recall, 4), **latency},
    )
    print(
        f"  ef={ef:4}  recall@{K}={row[f'recall@{K}']:.4f}  qps={row['qps']:8.1f}  "
        f"p50={row['p50_ms']:.3f}ms  p95={row['p95_ms']:.3f}ms",
        file=sys.stderr,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=50_000, help="vectors to index")
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    parser.add_argument("--ef-search", type=int, nargs="+", default=DEFAULT_EF_SEARCH)
    parser.add_argument("--systems", nargs="+",
                        default=["pyvec", "chromadb", "qdrant"],
                        choices=["pyvec", "chromadb", "qdrant"])
    parser.add_argument("--qdrant-url", default=None,
                        help="benchmark a running Qdrant server (real HNSW) instead "
                             "of local mode (which does exact search)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    dataset = load_sift(synthetic=args.synthetic, n=args.n)
    dataset = dataset.subset(None, args.queries)
    train = np.ascontiguousarray(dataset.train, dtype=np.float32)
    test = np.ascontiguousarray(dataset.test, dtype=np.float32)
    args._n, args._dim = int(train.shape[0]), int(train.shape[1])
    print(dataset.describe(), file=sys.stderr)

    run = BenchmarkRun("compare_vectordbs", seed=args.seed)
    if dataset.synthetic:
        run.note("SYNTHETIC data — not SIFT results.")

    cache = (
        cache_path(Path(__file__).resolve().parent / "datasets",
                   "gt", dataset.name, len(test), K)
        if not dataset.synthetic else None
    )
    if dataset.neighbours is not None and dataset.neighbours.shape[1] >= K:
        truth = dataset.neighbours[:, :K]
        print("  using precomputed ground truth", file=sys.stderr)
    else:
        print(f"  computing exact {K}-NN ground truth", file=sys.stderr)
        truth = ground_truth(test, train, K, metric="l2", cache=cache)

    root = Path(tempfile.mkdtemp(prefix="pyvec_compare_"))
    try:
        runners = {"pyvec": bench_pyvec, "chromadb": bench_chromadb,
                   "qdrant": bench_qdrant}
        for name in args.systems:
            try:
                runners[name](train, test, truth, run, args, root)
            except Exception as exc:  # noqa: BLE001 — one failure must not kill the rest
                import traceback
                run.note(f"{name} failed: {type(exc).__name__}: {exc}")
                traceback.print_exc(file=sys.stderr)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    path = run.save(args.out or "benchmarks/results/compare_vectordbs.csv")
    print(f"\n-> {path}", file=sys.stderr)
    print()
    run.print_table(["system", "ef_search", f"recall@{K}", "qps", "p50_ms",
                     "p95_ms", "build_s", "disk_bytes"])
    _summarise(run, args)
    return 0


def _summarise(run: BenchmarkRun, args) -> None:
    rows = run.results
    if not rows:
        return
    # Compare **at matched recall**, which is the only defensible way to put a single
    # number on an ANN comparison.
    #
    # An earlier version of this function reported "the fastest configuration that
    # clears 95% recall" per system and divided. That produced "ChromaDB is 1.0x
    # faster than PyVec" from PyVec at 0.9756 recall against Chroma at 0.9994 — same
    # throughput, wildly different quality. Recall and QPS trade against each other
    # continuously, so any comparison that does not pin one of them is meaningless.
    hnsw_rows = [r for r in rows if r["system"] != "qdrant-exact"]
    pyvec_rows = [r for r in hnsw_rows if r["system"] == "pyvec"]
    others = sorted({r["system"] for r in hnsw_rows if r["system"] != "pyvec"})
    if not pyvec_rows or not others:
        return

    print()
    print("Compared at MATCHED recall (the only fair single number):")
    for system in others:
        srows = [r for r in hnsw_rows if r["system"] == system]
        for other in srows:
            target = other[f"recall@{K}"]
            # Closest PyVec operating point by recall.
            mine = min(pyvec_rows, key=lambda r: abs(r[f"recall@{K}"] - target))
            gap = abs(mine[f"recall@{K}"] - target)
            ratio = other["qps"] / mine["qps"]
            verdict = (f"{system} {ratio:.1f}x faster" if ratio >= 1
                       else f"PyVec {1 / ratio:.1f}x faster")
            flag = "" if gap < 0.005 else f"  [recall differs by {gap:.4f} - approximate]"
            print(
                f"  recall~{target:.4f}: PyVec {mine['qps']:7.1f} QPS "
                f"(ef={mine['ef_search']})  vs  {system} {other['qps']:7.1f} QPS "
                f"(ef={other['ef_search']})  ->  {verdict}{flag}"
            )

    print()
    print("Build time and footprint:")
    for system in ["pyvec", *others]:
        srows = [r for r in hnsw_rows if r["system"] == system]
        if srows:
            r = srows[0]
            print(f"  {system:10} build {r['build_s']:8.1f}s   "
                  f"on disk {r['disk_bytes'] / 1e6:7.1f} MB")

    print()
    print("Read these with the threading caveats in the CSV: PyVec is single-threaded")
    print("by construction; the others may use internal thread pools the client API")
    print("gives no way to disable. Qdrant local mode is excluded entirely - it does")
    print("exact search, not HNSW.")


if __name__ == "__main__":
    raise SystemExit(main())
