"""Benchmark 5: startup and recovery time.

Directly targets PRD NF3 — "load a 1M vector collection from disk in <30
seconds" — and reports the breakdown BENCHMARKS.md asks for: metadata load, mmap,
dense index load, and warm-up to first successful query.

The breakdown matters because the components scale differently. mmap is close to
free regardless of size (the OS maps lazily and pages on demand). Metadata is a
JSON parse, linear in the number of vectors. The HNSW graph is the one that hurts:
it is a million-entry adjacency structure, and rebuilding those Python dicts is
what the custom binary format exists to keep bounded.

    python -m benchmarks.startup --n 1000000 --index hnsw     # the NF3 test
    python -m benchmarks.startup --n 20000                    # quick
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.harness import BenchmarkRun, process_rss_bytes
from pyvec.core.collection import (
    DENSE_INDEX_FILE,
    METADATA_FILE,
    Collection,
)

NF3_BUDGET_S = 30.0


def build(root: Path, vectors: np.ndarray, index_type: str, text: bool) -> float:
    """Create and populate the collection, then close it cleanly."""
    collection = Collection.create(
        "bench", root, dimension=int(vectors.shape[1]), metric="l2",
        index_type=index_type, capacity=len(vectors) + 16,
        text_field="content" if text else None,
        # Bulk load: group-commit rather than fsync per entry. The write path is
        # benchmark 4's subject, not this one's.
        fsync_policy="batch",
    )
    started = time.perf_counter()
    try:
        batch = 1000
        for offset in range(0, len(vectors), batch):
            chunk = vectors[offset : offset + batch]
            items = [
                {
                    "id": f"d{offset + i}",
                    "vector": chunk[i],
                    **({"metadata": {"content": f"document number {offset + i}"}}
                       if text else {}),
                }
                for i in range(len(chunk))
            ]
            collection.insert(items)
        elapsed = time.perf_counter() - started
    finally:
        collection.close()  # checkpoints, so the reopen path is the real one
    return elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--index", default="hnsw", choices=["hnsw", "ivf", "flat"])
    parser.add_argument("--text", action="store_true",
                        help="also build a BM25 index, to time its reload")
    parser.add_argument("--repeats", type=int, default=3,
                        help="open the collection this many times")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = BenchmarkRun("startup", seed=args.seed)
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.n, args.dim)).astype(np.float32)

    tmp = Path(tempfile.mkdtemp(prefix="pyvec_startup_"))
    try:
        print(
            f"=== Benchmark 5: startup / recovery ===\n"
            f"building {args.n:,} x {args.dim} ({args.index})",
            file=sys.stderr,
        )
        with run.time_block("build + checkpoint"):
            build_s = build(tmp, vectors, args.index, args.text)

        path = tmp / "bench"
        sizes = {
            "metadata_bytes": (path / METADATA_FILE).stat().st_size
            if (path / METADATA_FILE).exists() else 0,
            "vectors_bytes": (path / "vectors.bin").stat().st_size,
            "dense_index_bytes": sum(
                p.stat().st_size for p in path.glob(f"{DENSE_INDEX_FILE}*")
            ) + sum(p.stat().st_size for p in path.glob("dense.*")),
        }
        print(
            f"  on disk: vectors={sizes['vectors_bytes'] >> 20}MiB  "
            f"metadata={sizes['metadata_bytes'] >> 20}MiB  "
            f"index={sizes['dense_index_bytes'] >> 20}MiB",
            file=sys.stderr,
        )

        query = vectors[0]
        for attempt in range(args.repeats):
            rss_before = process_rss_bytes()
            t0 = time.perf_counter()
            collection = Collection.open(path)
            open_s = time.perf_counter() - t0
            try:
                t1 = time.perf_counter()
                hits = collection.search(query, k=10)
                first_query_s = time.perf_counter() - t1
                assert hits, "reopened collection returned no results"

                # Steady state after the first query has faulted pages in.
                t2 = time.perf_counter()
                for i in range(min(100, args.n)):
                    collection.search(vectors[i], k=10)
                warm_s = (time.perf_counter() - t2) / min(100, args.n)

                rss_after = process_rss_bytes()
                row = run.report(
                    {
                        "index": args.index,
                        "n": args.n,
                        "dim": args.dim,
                        "attempt": attempt + 1,
                        "text_field": bool(args.text),
                        "build_s": round(build_s, 3),
                        **sizes,
                    },
                    {
                        "open_s": round(open_s, 4),
                        "first_query_ms": round(first_query_s * 1000, 3),
                        "warm_query_ms": round(warm_s * 1000, 4),
                        "time_to_first_query_s": round(open_s + first_query_s, 4),
                        "rss_delta_bytes": max(0, rss_after - rss_before),
                        "num_vectors": len(collection),
                        "nf3_budget_s": NF3_BUDGET_S,
                        "meets_nf3": bool(open_s + first_query_s < NF3_BUDGET_S),
                    },
                )
                print(
                    f"  attempt {attempt + 1}: open={row['open_s']:.3f}s  "
                    f"first_query={row['first_query_ms']:.2f}ms  "
                    f"warm_query={row['warm_query_ms']:.3f}ms  "
                    f"-> time to first query {row['time_to_first_query_s']:.3f}s",
                    file=sys.stderr,
                )
            finally:
                collection.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = run.save(args.out)
    print(f"\n-> {out}", file=sys.stderr)
    print()
    run.print_table(
        ["index", "n", "attempt", "open_s", "first_query_ms", "warm_query_ms",
         "time_to_first_query_s", "meets_nf3"]
    )

    if run.results:
        worst = max(r["time_to_first_query_s"] for r in run.results)
        print()
        verdict = "PASS" if worst < NF3_BUDGET_S else "FAIL"
        print(
            f"PRD NF3 (1M vectors ready in <{NF3_BUDGET_S:.0f}s): {verdict} — "
            f"worst time to first query was {worst:.3f}s at n={args.n:,}"
        )
        if args.n < 1_000_000:
            print(
                f"  (measured at n={args.n:,}, not 1M. Extrapolating startup is "
                f"unreliable — the mmap is lazy while metadata and the graph are "
                f"linear — so run --n 1000000 for the real answer.)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
