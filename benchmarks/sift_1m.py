"""Benchmarks 1 and 2 from BENCHMARKS.md: recall-QPS Pareto on SIFT-1M.

* **Benchmark 1** — PyVec-HNSW vs ``faiss.IndexHNSWFlat`` at matched build
  parameters (``M=16, ef_construction=200``), sweeping
  ``ef_search in {16, 32, 64, 128, 256, 512}``.
* **Benchmark 2** — PyVec IVF-Flat, one curve per ``nlist``, sweeping ``nprobe``.

Run it::

    python -m benchmarks.sift_1m                     # the real thing (downloads 500MB)
    python -m benchmarks.sift_1m --n 100000          # a 100k prefix
    python -m benchmarks.sift_1m --synthetic --n 20000   # no download, harness check

Results land in ``benchmarks/results/*.csv``; ``plot_pareto.py`` turns them into
the committed plots. Nothing here plots directly — the CSV is the artifact, so a
plot can always be regenerated without re-running a multi-hour build.

**Expectations, stated up front so the results can disappoint honestly**
(BENCHMARKS.md): FAISS should beat PyVec on QPS by 2-5x at matched recall — it is
C++ with SIMD and we are Python. Recall should be within a couple of points at the
same parameters, because recall is a property of the algorithm, not the language.
If PyVec's recall trails FAISS by more than ~10 points, the implementation is
wrong; the usual culprit is neighbour selection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.datasets import load_sift
from benchmarks.harness import (
    BenchmarkRun,
    cache_path,
    ground_truth,
    mean_recall_at_k,
    process_rss_bytes,
    try_import_faiss,
)
from pyvec.core.types import ArrayVectorSource, Metric
from pyvec.indexes.hnsw import HNSWIndex
from pyvec.indexes.ivf import IVFFlatIndex

EF_SEARCH_SWEEP = [16, 32, 64, 128, 256, 512]
NLIST_SWEEP = [64, 256, 1024, 4096]
NPROBE_SWEEP = [1, 4, 16, 64]
K = 10


def _resolve_ground_truth(dataset, k: int, cache: Path | None) -> np.ndarray:
    if dataset.neighbours is not None and dataset.neighbours.shape[1] >= k:
        print(f"  using the dataset's precomputed {k}-NN ground truth", file=sys.stderr)
        return dataset.neighbours[:, :k]
    print(
        f"  computing exact {k}-NN ground truth by brute force "
        f"({dataset.test.shape[0]:,} queries x {dataset.size:,} vectors)",
        file=sys.stderr,
    )
    return ground_truth(
        dataset.test, dataset.train, k, metric=dataset.metric, cache=cache
    )


def bench_hnsw(dataset, truth, run: BenchmarkRun, args) -> None:
    print("\n=== Benchmark 1: HNSW ===", file=sys.stderr)
    source = ArrayVectorSource(dataset.train)
    metric = Metric.parse(dataset.metric)

    index = HNSWIndex(
        dataset.dim, metric, source,
        M=args.M, ef_construction=args.ef_construction, seed=args.seed,
    )
    rss_before = process_rss_bytes()
    with run.time_block(f"HNSW build (M={args.M}, ef_c={args.ef_construction})") as t:
        index.add(range(dataset.size))
    build_time = t.elapsed
    rss_after = process_rss_bytes()

    print(f"  level histogram: {index.level_histogram()}", file=sys.stderr)
    problems = index.validate()
    if problems:
        run.note(f"HNSW structural violations: {problems[:3]}")

    for ef in args.ef_search:
        latency, results = run.measure_each(
            lambda q, ef=ef: index.search(q, K, ef_search=ef),
            dataset.test,
            warmup=min(100, len(dataset.test)),
            collect=True,
        )
        recall = mean_recall_at_k(
            [[i for i, _ in r] for r in results], truth.tolist(), K
        )
        row = run.report(
            {
                "system": "pyvec-hnsw",
                "dataset": dataset.name,
                "n": dataset.size,
                "dim": dataset.dim,
                "M": args.M,
                "ef_construction": args.ef_construction,
                "ef_search": ef,
                "build_s": round(build_time, 3),
                "index_memory_bytes": index.memory_bytes(),
                "build_rss_delta_bytes": max(0, rss_after - rss_before),
            },
            {f"recall@{K}": round(recall, 4), **latency},
        )
        print(
            f"  ef_search={ef:4}  recall@{K}={row[f'recall@{K}']:.4f}  "
            f"qps={row['qps']:8.1f}  p50={row['p50_ms']:.3f}ms  "
            f"p95={row['p95_ms']:.3f}ms  p99={row['p99_ms']:.3f}ms",
            file=sys.stderr,
        )


def bench_faiss(dataset, truth, run: BenchmarkRun, args) -> None:
    faiss = try_import_faiss()
    if faiss is None:
        run.note(
            "FAISS not installed, baseline skipped. `pip install faiss-cpu` to "
            "produce the comparison BENCHMARKS.md asks for."
        )
        return

    print("\n=== Benchmark 1 baseline: FAISS IndexHNSWFlat ===", file=sys.stderr)
    if dataset.metric == "l2":
        index = faiss.IndexHNSWFlat(dataset.dim, args.M, faiss.METRIC_L2)
    else:
        index = faiss.IndexHNSWFlat(dataset.dim, args.M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
    # Single-threaded, to match PyVec. Letting FAISS use every core would compare
    # a parallel C++ index against a serial Python one and prove nothing.
    faiss.omp_set_num_threads(1)

    with run.time_block("FAISS build") as t:
        index.add(dataset.train)
    build_time = t.elapsed

    for ef in args.ef_search:
        index.hnsw.efSearch = ef
        queries = dataset.test

        def search(q, index=index):
            return index.search(q.reshape(1, -1), K)

        latency, results = run.measure_each(
            search, queries, warmup=min(100, len(queries)), collect=True
        )
        recall = mean_recall_at_k(
            [list(ids[0]) for _, ids in results], truth.tolist(), K
        )
        row = run.report(
            {
                "system": "faiss-hnsw",
                "dataset": dataset.name,
                "n": dataset.size,
                "dim": dataset.dim,
                "M": args.M,
                "ef_construction": args.ef_construction,
                "ef_search": ef,
                "build_s": round(build_time, 3),
            },
            {f"recall@{K}": round(recall, 4), **latency},
        )
        print(
            f"  ef_search={ef:4}  recall@{K}={row[f'recall@{K}']:.4f}  "
            f"qps={row['qps']:8.1f}  p95={row['p95_ms']:.3f}ms",
            file=sys.stderr,
        )


def bench_ivf(dataset, truth, run: BenchmarkRun, args) -> None:
    print("\n=== Benchmark 2: IVF-Flat ===", file=sys.stderr)
    source = ArrayVectorSource(dataset.train)
    metric = Metric.parse(dataset.metric)

    for nlist in args.nlist:
        if nlist > dataset.size:
            run.note(f"skipping nlist={nlist}: more centroids than vectors")
            continue
        index = IVFFlatIndex(
            dataset.dim, metric, source,
            nlist=nlist, seed=args.seed, retrain_growth_factor=None,
        )
        with run.time_block(f"IVF build (nlist={nlist}, k-means dominates)") as t:
            index.add(range(dataset.size))
            if not index.is_trained:
                index.train()
        build_time = t.elapsed

        sizes = [len(v) for v in index.postings.values()]
        print(
            f"  posting lists: min={min(sizes)} max={max(sizes)} "
            f"mean={np.mean(sizes):.1f}",
            file=sys.stderr,
        )

        for nprobe in [p for p in args.nprobe if p <= nlist] + [nlist // 2]:
            if nprobe < 1 or nprobe > nlist:
                continue
            latency, results = run.measure_each(
                lambda q, nprobe=nprobe: index.search(q, K, nprobe=nprobe),
                dataset.test,
                warmup=min(100, len(dataset.test)),
                collect=True,
            )
            recall = mean_recall_at_k(
                [[i for i, _ in r] for r in results], truth.tolist(), K
            )
            row = run.report(
                {
                    "system": "pyvec-ivf",
                    "dataset": dataset.name,
                    "n": dataset.size,
                    "dim": dataset.dim,
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "build_s": round(build_time, 3),
                    "index_memory_bytes": index.memory_bytes(),
                },
                {f"recall@{K}": round(recall, 4), **latency},
            )
            print(
                f"  nlist={nlist:5} nprobe={nprobe:5}  "
                f"recall@{K}={row[f'recall@{K}']:.4f}  qps={row['qps']:8.1f}  "
                f"p95={row['p95_ms']:.3f}ms",
                file=sys.stderr,
            )


def bench_flat_baseline(dataset, truth, run: BenchmarkRun, args) -> None:
    """Brute force, for the "how much did ANN actually buy us" number."""
    from pyvec.indexes.flat import FlatIndex

    print("\n=== Reference: brute force ===", file=sys.stderr)
    source = ArrayVectorSource(dataset.train)
    index = FlatIndex(dataset.dim, Metric.parse(dataset.metric), source)
    index.add(range(dataset.size))

    # A full scan is slow, so time a sample rather than all 10k queries.
    sample = dataset.test[: min(50, len(dataset.test))]
    latency, results = run.measure_each(
        lambda q: index.search(q, K), sample, warmup=2, collect=True
    )
    recall = mean_recall_at_k(
        [[i for i, _ in r] for r in results], truth[: len(sample)].tolist(), K
    )
    run.report(
        {
            "system": "pyvec-flat",
            "dataset": dataset.name,
            "n": dataset.size,
            "dim": dataset.dim,
        },
        {f"recall@{K}": round(recall, 4), **latency},
    )
    print(
        f"  recall@{K}={recall:.4f} (must be 1.0)  qps={latency['qps']:.1f}",
        file=sys.stderr,
    )
    if recall < 0.999 and dataset.neighbours is None:
        run.note(f"brute force scored {recall:.4f} against its own ground truth")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--synthetic", action="store_true",
                        help="use a seeded synthetic stand-in, no download")
    parser.add_argument("--n", type=int, default=None,
                        help="index only the first N vectors")
    parser.add_argument("--queries", type=int, default=None,
                        help="use only the first N queries")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=EF_SEARCH_SWEEP)
    parser.add_argument("--nlist", type=int, nargs="+", default=NLIST_SWEEP)
    parser.add_argument("--nprobe", type=int, nargs="+", default=NPROBE_SWEEP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["hnsw", "ivf", "faiss", "flat"])
    parser.add_argument("--out", default=None, help="CSV path")
    args = parser.parse_args(argv)

    dataset = load_sift(synthetic=args.synthetic, n=args.n)
    if args.queries:
        dataset = dataset.subset(None, args.queries)
    print(dataset.describe(), file=sys.stderr)

    run = BenchmarkRun("sift_1m", seed=args.seed)
    if dataset.synthetic:
        run.note(
            "SYNTHETIC data: these numbers verify the harness and are NOT "
            "SIFT-1M results. Do not put them on a resume."
        )
        if dataset.size < 100_000:
            run.note(
                f"n={dataset.size:,} is small enough that recall saturates at "
                f"1.0 for every parameter — ANN difficulty scales with n. Run "
                f"the real dataset for meaningful recall curves."
            )

    cache = (
        cache_path(
            Path(__file__).resolve().parent / "datasets",
            "gt", dataset.name, dataset.test.shape[0], K,
        )
        if not dataset.synthetic
        else None
    )
    truth = _resolve_ground_truth(dataset, K, cache)

    started = time.time()
    if "hnsw" not in args.skip:
        bench_hnsw(dataset, truth, run, args)
    if "faiss" not in args.skip:
        bench_faiss(dataset, truth, run, args)
    if "ivf" not in args.skip:
        bench_ivf(dataset, truth, run, args)
    if "flat" not in args.skip:
        bench_flat_baseline(dataset, truth, run, args)

    path = run.save(args.out)
    print(f"\ntotal {time.time() - started:.1f}s -> {path}", file=sys.stderr)
    print()
    run.print_table(
        ["system", "ef_search", "nlist", "nprobe", f"recall@{K}",
         "qps", "p50_ms", "p95_ms", "p99_ms", "build_s"]
    )
    _summarise(run)
    return 0


def _summarise(run: BenchmarkRun) -> None:
    """The one-line comparison that goes in the README's Results section."""
    hnsw = [r for r in run.results if r["system"] == "pyvec-hnsw"]
    ivf = [r for r in run.results if r["system"] == "pyvec-ivf"]
    faiss_rows = [r for r in run.results if r["system"] == "faiss-hnsw"]
    target = 0.95
    print()

    def at_recall(rows, target):
        ok = [r for r in rows if r.get(f"recall@{K}", 0) >= target]
        return max(ok, key=lambda r: r["qps"]) if ok else None

    best_hnsw = at_recall(hnsw, target)
    if best_hnsw:
        print(
            f"HNSW at >={target:.0%} recall@{K}: {best_hnsw['qps']:.0f} QPS "
            f"(ef_search={best_hnsw['ef_search']}, "
            f"p95={best_hnsw['p95_ms']:.2f}ms)"
        )
    else:
        best = max(hnsw, key=lambda r: r[f"recall@{K}"], default=None)
        if best:
            print(
                f"HNSW never reached {target:.0%} recall@{K}; best was "
                f"{best[f'recall@{K}']:.4f} at ef_search={best['ef_search']}"
            )

    best_ivf = at_recall(ivf, target)
    if best_hnsw and best_ivf:
        ratio = best_hnsw["qps"] / best_ivf["qps"]
        # Phrase the comparison in whichever direction is actually true. On
        # SIFT-1M HNSW is expected to win (BENCHMARKS.md benchmark 2), but at
        # small n IVF often does, and the summary must not misreport that.
        verdict = (
            f"HNSW is {ratio:.1f}x faster"
            if ratio >= 1
            else f"IVF is {1 / ratio:.1f}x faster here"
        )
        print(
            f"IVF at >={target:.0%} recall@{K}: {best_ivf['qps']:.0f} QPS "
            f"(nlist={best_ivf['nlist']}, nprobe={best_ivf['nprobe']})  -> {verdict}"
        )
    best_faiss = at_recall(faiss_rows, target)
    if best_hnsw and best_faiss:
        print(
            f"FAISS at >={target:.0%} recall@{K}: {best_faiss['qps']:.0f} QPS  "
            f"-> PyVec is within {best_faiss['qps'] / best_hnsw['qps']:.1f}x "
            f"of FAISS QPS at matched recall"
        )


if __name__ == "__main__":
    raise SystemExit(main())
