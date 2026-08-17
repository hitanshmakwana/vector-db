"""GloVe-100 sweep — the secondary dataset from BENCHMARKS.md.

1,183,514 word embeddings, 100 dimensions, cosine. It matters because SIFT
descriptors are *not* representative of the workload PyVec is actually for:
GloVe is real text-embedding geometry, which clusters differently and is where
IVF's affinity for clustered data should show up.

This is a thin wrapper over the SIFT sweep with the GloVe loader and cosine
metric substituted, because the measurement procedure is identical and duplicating
it would let the two drift apart.

Note that GloVe is first on the cut list in PROJECT_PLAN's risk plan: SIFT-1M
alone satisfies the deliverable. This exists for when there is budget for it.

    python -m benchmarks.glove --synthetic --n 20000
    python -m benchmarks.glove
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmarks.datasets import load_glove
from benchmarks.harness import BenchmarkRun, cache_path
from benchmarks.sift_1m import (
    EF_SEARCH_SWEEP,
    K,
    NLIST_SWEEP,
    NPROBE_SWEEP,
    _resolve_ground_truth,
    _summarise,
    bench_faiss,
    bench_hnsw,
    bench_ivf,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=EF_SEARCH_SWEEP)
    parser.add_argument("--nlist", type=int, nargs="+", default=NLIST_SWEEP)
    parser.add_argument("--nprobe", type=int, nargs="+", default=NPROBE_SWEEP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["hnsw", "ivf", "faiss"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    dataset = load_glove(synthetic=args.synthetic, n=args.n)
    if args.queries:
        dataset = dataset.subset(None, args.queries)
    print(dataset.describe(), file=sys.stderr)

    run = BenchmarkRun("glove", seed=args.seed)
    if dataset.synthetic:
        run.note("SYNTHETIC data: not GloVe results.")

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

    path = run.save(args.out)
    print(f"\ntotal {time.time() - started:.1f}s -> {path}", file=sys.stderr)
    print()
    run.print_table(
        ["system", "ef_search", "nlist", "nprobe", f"recall@{K}",
         "qps", "p50_ms", "p95_ms", "build_s"]
    )
    _summarise(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
