"""Benchmark 4: the cost of durability.

BENCHMARKS.md: insert 100k vectors, batched 1k at a time, under three
configurations:

* **config 1** — WAL on, fsync per batch entry (the default, safest)
* **config 2** — WAL on, fsync deferred to collection close (group commit)
* **config 3** — WAL off (unsafe, for reference)

Reporting insert throughput for each surfaces the durability-versus-performance
trade-off, which is the interesting conversation. The honest framing is that
config 1 buys you "acknowledged means durable" and configs 2 and 3 do not:

* **config 2** can lose the tail of the last un-synced group on a power failure
  (a process crash is still survivable — the bytes are in the OS page cache and
  the kernel writes them out).
* **config 3** can lose everything since the last checkpoint, under any kind of
  crash at all.

    python -m benchmarks.persistence --n 100000
    python -m benchmarks.persistence --n 5000        # quick
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.harness import BenchmarkRun
from pyvec.core.collection import Collection
from pyvec.storage.wal import FsyncPolicy

CONFIGS = [
    ("wal-fsync-per-entry", {"wal_enabled": True, "fsync_policy": FsyncPolicy.ENTRY},
     "durable on ack"),
    ("wal-group-commit", {"wal_enabled": True, "fsync_policy": FsyncPolicy.BATCH},
     "may lose the last group on power loss"),
    ("wal-disabled", {"wal_enabled": False, "fsync_policy": FsyncPolicy.NEVER},
     "may lose everything since the last checkpoint"),
]


def run_config(
    label: str,
    kwargs: dict,
    durability: str,
    vectors: np.ndarray,
    batch_size: int,
    run: BenchmarkRun,
    index_type: str,
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix=f"pyvec_persist_{label}_"))
    try:
        collection = Collection.create(
            "bench", tmp, dimension=int(vectors.shape[1]), metric="l2",
            index_type=index_type, capacity=len(vectors) + 16, **kwargs,
        )
        try:
            n = len(vectors)
            start = time.perf_counter()
            batch_latencies = []
            for offset in range(0, n, batch_size):
                chunk = vectors[offset : offset + batch_size]
                items = [
                    {"id": f"d{offset + i}", "vector": chunk[i]}
                    for i in range(len(chunk))
                ]
                t0 = time.perf_counter()
                collection.insert(items)
                batch_latencies.append(time.perf_counter() - t0)
            insert_elapsed = time.perf_counter() - start

            # Time the checkpoint separately: for the WAL-disabled config it is the
            # *only* thing making data durable, so folding it into insert
            # throughput would flatter the unsafe mode.
            t0 = time.perf_counter()
            collection.checkpoint()
            checkpoint_elapsed = time.perf_counter() - t0

            wal_bytes = collection.wal.size_bytes
            disk_bytes = collection._disk_bytes()
        finally:
            collection.close()

        row = run.report(
            {
                "config": label,
                "durability": durability,
                "n": n,
                "dim": int(vectors.shape[1]),
                "batch_size": batch_size,
                "index": index_type,
            },
            {
                "insert_s": round(insert_elapsed, 3),
                "vectors_per_s": round(n / insert_elapsed, 1),
                "checkpoint_s": round(checkpoint_elapsed, 3),
                "total_s": round(insert_elapsed + checkpoint_elapsed, 3),
                "durable_vectors_per_s": round(
                    n / (insert_elapsed + checkpoint_elapsed), 1
                ),
                "batch_p50_ms": round(1000 * float(np.percentile(batch_latencies, 50)), 3),
                "batch_p95_ms": round(1000 * float(np.percentile(batch_latencies, 95)), 3),
                "wal_bytes_after_checkpoint": wal_bytes,
                "disk_bytes": disk_bytes,
            },
        )
        print(
            f"  {label:22} {row['vectors_per_s']:10,.0f} vec/s   "
            f"insert={row['insert_s']:.2f}s  checkpoint={row['checkpoint_s']:.2f}s  "
            f"batch_p95={row['batch_p95_ms']:.1f}ms",
            file=sys.stderr,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--index", default="flat", choices=["flat", "hnsw", "ivf"],
                        help="flat by default so the measurement isolates the "
                             "write path rather than graph construction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = BenchmarkRun("persistence", seed=args.seed)
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.n, args.dim)).astype(np.float32)

    print(
        f"=== Benchmark 4: persistence overhead ===\n"
        f"{args.n:,} vectors x {args.dim}, batches of {args.batch_size}, "
        f"index={args.index}",
        file=sys.stderr,
    )
    if args.index != "flat":
        run.note(
            f"index={args.index}: throughput includes index construction, so the "
            f"differences between WAL configs will look smaller than they are."
        )

    for label, kwargs, durability in CONFIGS:
        run_config(
            label, kwargs, durability, vectors, args.batch_size, run, args.index
        )

    path = run.save(args.out)
    print(f"\n-> {path}", file=sys.stderr)
    print()
    run.print_table(
        ["config", "vectors_per_s", "insert_s", "checkpoint_s",
         "durable_vectors_per_s", "batch_p95_ms", "durability"]
    )

    rows = {r["config"]: r for r in run.results}
    safe = rows.get("wal-fsync-per-entry")
    unsafe = rows.get("wal-disabled")
    group = rows.get("wal-group-commit")
    if safe and unsafe:
        print()
        print(
            f"fsync-per-entry costs "
            f"{unsafe['vectors_per_s'] / safe['vectors_per_s']:.1f}x throughput "
            f"versus no WAL at all"
        )
    if safe and group:
        print(
            f"group commit recovers "
            f"{group['vectors_per_s'] / safe['vectors_per_s']:.1f}x of that, and is "
            f"what a production system would default to"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
