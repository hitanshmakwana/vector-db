"""Measurement harness shared by every benchmark.

Built to the four principles in BENCHMARKS.md:

* **Reproducible.** Every run fixes seeds. Results go to CSV; plots read the CSV.
  A benchmark whose numbers move between runs is worthless.
* **Honest.** Baselines are real (FAISS where available), and anything skipped is
  reported as skipped rather than quietly omitted.
* **Multi-point.** Sweeps produce curves. A single number is a marketing claim.
* **Percentiles over means.** p50/p95/p99/p99.9, measured per query rather than
  per batch, because a batch average hides exactly the tail we care about.

The :class:`BenchmarkRun` API follows the reference sketch in BENCHMARKS.md, with
the additions that turn it into something usable: warm-up separated from
measurement, per-query latency collection, and a CSV writer that does not need
pandas.
"""

from __future__ import annotations

import csv
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "BenchmarkRun",
    "LatencyStats",
    "recall_at_k",
    "mean_recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "ground_truth",
    "process_rss_bytes",
    "RESULTS_DIR",
    "PLOTS_DIR",
]

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
PLOTS_DIR = BENCH_DIR / "plots"
DATASETS_DIR = BENCH_DIR / "datasets"

DEFAULT_SEED = 42

#: BENCHMARKS.md: "measure over >=1000 queries after 100 warm-up".
DEFAULT_WARMUP = 100
DEFAULT_ITERS = 1000


@dataclass(slots=True)
class LatencyStats:
    """Per-call timings and the percentiles derived from them."""

    latencies: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.latencies:
            return {}
        s = sorted(self.latencies)
        total = sum(s)
        return {
            "n": len(s),
            "qps": len(s) / total if total > 0 else float("inf"),
            "mean_ms": 1000 * total / len(s),
            "p50_ms": 1000 * _percentile(s, 50),
            "p95_ms": 1000 * _percentile(s, 95),
            "p99_ms": 1000 * _percentile(s, 99),
            "p999_ms": 1000 * _percentile(s, 99.9),
            "min_ms": 1000 * s[0],
            "max_ms": 1000 * s[-1],
        }


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted sequence.

    NumPy's default linear interpolation invents values that never occurred,
    which is misleading for latency: a reported p99 should be a measurement.
    """
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(round(pct / 100 * len(sorted_values) + 0.5)) - 1))
    return sorted_values[k]


class BenchmarkRun:
    """Collects rows of ``{config..., metrics...}`` and writes them to CSV."""

    def __init__(self, name: str, seed: int = DEFAULT_SEED) -> None:
        self.name = name
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.results: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.started = time.time()

    # ------------------------------------------------------------------ #
    # Measurement
    # ------------------------------------------------------------------ #

    def measure(
        self,
        fn: Callable[..., Any],
        *args: Any,
        iters: int = DEFAULT_ITERS,
        warmup: int = DEFAULT_WARMUP,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Time ``fn(*args)`` repeatedly and return latency percentiles.

        Warm-up runs are discarded: the first queries fault mmap pages in, fill
        the CPU caches and let the branch predictors settle, so including them
        would measure cold-start rather than steady-state throughput.
        """
        for _ in range(warmup):
            fn(*args, **kwargs)

        stats = LatencyStats()
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args, **kwargs)
            stats.latencies.append(time.perf_counter() - start)
        return stats.summary()

    def measure_each(
        self,
        fn: Callable[[Any], Any],
        inputs: Sequence[Any],
        *,
        warmup: int = DEFAULT_WARMUP,
        collect: bool = False,
    ) -> tuple[dict[str, float], list[Any]]:
        """Time ``fn`` once per input — the shape a real query sweep needs.

        Returns ``(latency_summary, outputs)``. ``outputs`` is only populated when
        ``collect`` is set, so a recall pass can reuse the same timed run instead
        of querying twice.
        """
        for i in range(min(warmup, len(inputs))):
            fn(inputs[i % len(inputs)])

        stats = LatencyStats()
        outputs: list[Any] = []
        for item in inputs:
            start = time.perf_counter()
            result = fn(item)
            stats.latencies.append(time.perf_counter() - start)
            if collect:
                outputs.append(result)
        return stats.summary(), outputs

    def time_block(self, label: str) -> "_Timer":
        """Context manager for one-off wall-clock timings (build, load, ...)."""
        return _Timer(self, label)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def report(self, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
        row = {**config, **metrics}
        self.results.append(row)
        return row

    def note(self, message: str) -> None:
        """Record something the reader needs to know — a skip, a cap, a caveat."""
        self.notes.append(message)
        print(f"  note: {message}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def save(self, path: Path | str | None = None) -> Path:
        """Write results to CSV. Plots read this, never a live index."""
        path = Path(path) if path else RESULTS_DIR / f"{self.name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            path.write_text("", encoding="utf-8")
            return path

        columns: list[str] = []
        for row in self.results:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)

        self._save_environment(path.with_suffix(".env.json"))
        return path

    def _save_environment(self, path: Path) -> None:
        """Record the machine. A QPS number without a CPU is uninterpretable."""
        payload = {
            "benchmark": self.name,
            "seed": self.seed,
            "started_unix": self.started,
            "duration_s": round(time.time() - self.started, 2),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": _cpu_count(),
            "notes": self.notes,
        }
        try:
            import pyvec

            payload["pyvec"] = pyvec.__version__
        except Exception:
            pass
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def print_table(self, columns: Sequence[str] | None = None) -> None:
        if not self.results:
            print("(no results)")
            return
        if columns is None:
            columns = list(self.results[0].keys())
        columns = [c for c in columns if any(c in r for r in self.results)]

        def cell(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
            return "-" if value is None else str(value)

        widths = {
            c: max(len(c), max(len(cell(r.get(c))) for r in self.results))
            for c in columns
        }
        print("  ".join(c.rjust(widths[c]) for c in columns))
        print("  ".join("-" * widths[c] for c in columns))
        for row in self.results:
            print("  ".join(cell(row.get(c)).rjust(widths[c]) for c in columns))


@dataclass(slots=True)
class _Timer:
    run: BenchmarkRun
    label: str
    elapsed: float = 0.0
    _start: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        print(f"  {self.label}: {self.elapsed:.3f}s", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Quality metrics
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence[Any], relevant: Sequence[Any], k: int) -> float:
    """``|retrieved@k ∩ relevant@k| / k`` — the definition in BENCHMARKS.md."""
    if k <= 0:
        return 0.0
    truth = set(relevant[:k])
    if not truth:
        return 0.0
    got = set(retrieved[:k])
    return len(got & truth) / len(truth)


def mean_recall_at_k(
    retrieved: Sequence[Sequence[Any]], relevant: Sequence[Sequence[Any]], k: int
) -> float:
    if not retrieved:
        return 0.0
    return float(
        np.mean([recall_at_k(r, t, k) for r, t in zip(retrieved, relevant)])
    )


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return float(
        sum(g / np.log2(i + 2) for i, g in enumerate(list(gains)[:k]))
    )


def ndcg_at_k(
    retrieved: Sequence[Any], relevance: dict[Any, float], k: int = 10
) -> float:
    """Normalised discounted cumulative gain.

    The standard IR metric for graded relevance: gains discounted by log of rank,
    divided by the best achievable ordering. Unlike recall it is sensitive to
    *where* in the list a relevant document lands, which is the whole point when
    comparing hybrid against dense-only.
    """
    gains = [relevance.get(doc_id, 0.0) for doc_id in list(retrieved)[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    best = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / best if best > 0 else 0.0


def mrr_at_k(retrieved: Sequence[Any], relevant: set[Any], k: int = 10) -> float:
    """Reciprocal rank of the first relevant hit within the top k."""
    for rank, doc_id in enumerate(list(retrieved)[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ground_truth(
    queries: np.ndarray,
    vectors: np.ndarray,
    k: int,
    metric: str = "l2",
    block: int = 4096,
    cache: Path | None = None,
) -> np.ndarray:
    """Exact top-``k`` per query by brute force, ``(n_queries, k)`` of row ids.

    Cached to disk when ``cache`` is given: BENCHMARKS.md is explicit that
    "recall@k needs ground truth — compute it once from brute-force, cache it,
    reuse across runs". At 1M x 128 this is minutes of work that would otherwise
    be repeated on every sweep.
    """
    if cache is not None and cache.exists():
        loaded = np.load(cache)
        if loaded.shape[0] == queries.shape[0] and loaded.shape[1] >= k:
            return loaded[:, :k]

    from pyvec.core.distance import distance
    from pyvec.core.types import Metric

    m = Metric.parse(metric)
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for qi in range(queries.shape[0]):
        best_ids: list[np.ndarray] = []
        best_ds: list[np.ndarray] = []
        for start in range(0, vectors.shape[0], block):
            stop = min(start + block, vectors.shape[0])
            d = distance(m, queries[qi], vectors[start:stop])
            take = min(k, d.shape[0])
            part = np.argpartition(d, take - 1)[:take]
            best_ids.append(part + start)
            best_ds.append(d[part])
        ids = np.concatenate(best_ids)
        ds = np.concatenate(best_ds)
        order = np.argsort(ds, kind="stable")[:k]
        out[qi] = ids[order]

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def cache_path(directory: Path, *parts: object) -> Path:
    """Build a filesystem-safe cache path from arbitrary label parts.

    Dataset labels carry characters Windows rejects outright (``:`` in a name like
    ``sift-1m[:100000]`` raises ``OSError: Invalid argument``), so anything that is
    not alphanumeric, dash, dot or underscore is folded to an underscore.
    """
    stem = "_".join(str(p) for p in parts)
    safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in stem)
    return Path(directory) / f"{safe}.npy"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def process_rss_bytes() -> int:
    """Resident set size, for the memory-footprint metric. 0 if unavailable.

    Deliberately dependency-free: ``psutil`` would be the obvious answer but it is
    not worth a dependency for one number, and both fallbacks below cover the
    platforms this project is developed and benchmarked on.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _COUNTERS()
            counters.cb = ctypes.sizeof(_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
        except Exception:
            return 0
        return 0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return int(usage * 1024) if sys.platform.startswith("linux") else int(usage)
    except Exception:
        return 0


def _cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 0
    except Exception:
        return 0


def try_import_faiss():
    """Return the faiss module, or ``None`` with an explanation printed.

    ADR-007: FAISS is allowed *only* here, as a benchmark baseline. It is never
    imported from inside ``pyvec/``.
    """
    try:
        import faiss  # noqa: PLC0415

        return faiss
    except ImportError:
        return None


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: write files, never open a window
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None
