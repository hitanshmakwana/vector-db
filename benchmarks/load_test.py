"""Load test: what does the HTTP surface do under concurrent traffic?

Distinct from the ANN sweeps in ``sift_1m.py``, which measure the *index* in
isolation on one thread. This measures the **deployed system**: FastAPI, JSON
serialisation, the reader-writer lock, and the thread pool, all under N concurrent
clients. Those are the numbers that matter for the "p95 latency for 128-dim search"
kind of claim, because they include everything a real caller pays for.

    python -m benchmarks.load_test                             # spawns a server
    python -m benchmarks.load_test --url http://host:8080      # hit a live one
    python -m benchmarks.load_test --concurrency 1 2 4 8 16 32

What it reports per concurrency level: throughput, latency percentiles, error
count, and the scaling efficiency against the single-client baseline.

**Why perfect scaling is not expected.** Search holds the GIL for the Python parts
of the graph walk and releases it inside NumPy's BLAS calls, so concurrency helps
only in proportion to the time spent in C. Writes take the collection's write lock
exclusively and serialise against everything. A mixed read/write run should show
reads degrading once writers appear — that is the RW lock working, not a defect.
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness import BenchmarkRun  # noqa: E402
from pyvec.client import PyVecClient  # noqa: E402

DIM = 128
COLLECTION = "loadtest"
DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32]


@dataclass(slots=True)
class Outcome:
    latencies: list[float] = field(default_factory=list)
    errors: int = 0
    http_errors: int = 0
    transport_errors: int = 0
    error_samples: list[str] = field(default_factory=list)
    bytes_in: int = 0


class KeepAliveClient:
    """One persistent HTTP/1.1 connection, reused across requests.

    **This is not an optimisation, it is a correctness requirement for the test.**
    Opening a fresh TCP connection per request (which is what ``urllib`` does) left
    tens of thousands of sockets in ``TIME_WAIT`` and exhausted the ephemeral port
    range: on Windows that surfaces as ``WinError 10048`` and throughput collapsed
    to ~1 rps once BM25 got fast enough to push past ~300 rps. Those failures were
    the load generator hitting an OS limit, not the server failing — and a load
    test that reports client-side artifacts as server errors is worse than no load
    test.

    Reusing one connection per worker also means the measured latency is request
    handling rather than TCP setup, which is what a real client with a connection
    pool would see.
    """

    __slots__ = ("host", "port", "timeout", "_conn")

    def __init__(self, base_url: str, timeout: float) -> None:
        import http.client
        import urllib.parse

        parsed = urllib.parse.urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.timeout = timeout
        self._conn: "http.client.HTTPConnection | None" = None

    def _connect(self):
        import http.client

        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def post(self, path: str, payload: dict) -> int:
        """POST JSON. Returns bytes read. Raises on HTTP >= 400 or transport error."""
        import http.client

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        # One transparent retry: a keep-alive connection can be closed by the peer
        # between requests, which is normal and not a failure.
        for attempt in (0, 1):
            try:
                conn = self._connect()
                conn.request("POST", path, body=body, headers=headers)
                response = conn.getresponse()
                # The body must be drained or the connection cannot be reused.
                data = response.read()
                if response.status >= 400:
                    raise HTTPStatusError(response.status, data[:200].decode(
                        "utf-8", "replace"))
                return len(data)
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt:
                    raise
        return 0


class HTTPStatusError(RuntimeError):
    """A >= 400 response. Distinct from a transport failure, and counted apart."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100 * len(s) + 0.5)) - 1))
    return s[k]


def worker(
    stop: threading.Event,
    work: "queue.Queue[dict]",
    base_url: str,
    path: str,
    outcome: Outcome,
    timeout: float,
) -> None:
    client = KeepAliveClient(base_url, timeout)
    try:
        while not stop.is_set():
            try:
                payload = work.get(timeout=0.05)
            except queue.Empty:
                continue
            start = time.perf_counter()
            try:
                outcome.bytes_in += client.post(path, payload)
                outcome.latencies.append(time.perf_counter() - start)
            except HTTPStatusError as exc:
                outcome.errors += 1
                outcome.http_errors += 1
                if len(outcome.error_samples) < 3:
                    outcome.error_samples.append(str(exc))
            except (OSError, RuntimeError) as exc:
                outcome.errors += 1
                outcome.transport_errors += 1
                if len(outcome.error_samples) < 3:
                    outcome.error_samples.append(f"{type(exc).__name__}: {exc}")
            finally:
                work.task_done()
    finally:
        client.close()


def run_level(
    url: str,
    endpoint: str,
    payloads: list[dict],
    concurrency: int,
    duration: float,
    timeout: float,
) -> tuple[dict, list[Outcome]]:
    """Drive ``concurrency`` client threads at one endpoint for ``duration``."""
    path = f"/collections/{COLLECTION}/{endpoint}"
    work: "queue.Queue[dict]" = queue.Queue(maxsize=concurrency * 4)
    stop = threading.Event()
    outcomes = [Outcome() for _ in range(concurrency)]
    threads = [
        threading.Thread(
            target=worker,
            args=(stop, work, url, path, outcomes[i], timeout),
            daemon=True,
        )
        for i in range(concurrency)
    ]
    for t in threads:
        t.start()

    # Warm up so the first requests (import-time caches, mmap page faults, JIT of
    # nothing in particular) do not land in the measurement.
    for i in range(min(len(payloads), concurrency * 2)):
        work.put(payloads[i % len(payloads)])
    work.join()
    for o in outcomes:
        o.latencies.clear()
        o.errors = 0
        o.bytes_in = 0

    started = time.perf_counter()
    sent = 0
    while time.perf_counter() - started < duration:
        work.put(payloads[sent % len(payloads)])
        sent += 1
    work.join()
    elapsed = time.perf_counter() - started
    stop.set()
    for t in threads:
        t.join(timeout=5)

    latencies = [x for o in outcomes for x in o.latencies]
    errors = sum(o.errors for o in outcomes)
    http_errors = sum(o.http_errors for o in outcomes)
    transport_errors = sum(o.transport_errors for o in outcomes)
    samples = [s for o in outcomes for s in o.error_samples][:3]
    completed = len(latencies)

    return (
        {
            "endpoint": endpoint,
            "concurrency": concurrency,
            "duration_s": round(elapsed, 3),
            "requests": completed,
            "errors": errors,
            # Split apart deliberately: an HTTP 5xx is the server failing, a
            # transport error is usually the client or the OS running out of
            # something. Reporting one number conflates a real defect with an
            # artifact of the harness.
            "http_errors": http_errors,
            "transport_errors": transport_errors,
            "error_rate": round(errors / max(completed + errors, 1), 5),
            "rps": round(completed / elapsed, 1) if elapsed > 0 else 0.0,
            "mean_ms": round(1000 * statistics.fmean(latencies), 3) if latencies else 0,
            "p50_ms": round(1000 * _percentile(latencies, 50), 3),
            "p95_ms": round(1000 * _percentile(latencies, 95), 3),
            "p99_ms": round(1000 * _percentile(latencies, 99), 3),
            "max_ms": round(1000 * max(latencies), 3) if latencies else 0,
            "error_samples": "; ".join(samples),
        },
        outcomes,
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LocalServer:
    def __init__(self, data_dir: Path, port: int) -> None:
        import uvicorn

        from pyvec.api.server import create_app
        from pyvec.core.collection_manager import CollectionManager

        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(manager=CollectionManager(data_dir)),
                host="127.0.0.1", port=port, log_level="error",
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *_exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=15)


#: Vocabulary size for the generated text field.
#:
#: This matters more than it looks. BM25 cost is proportional to the *length of the
#: posting lists the query terms touch*, so a tiny vocabulary is pathological: with
#: 10 words over 20k documents every posting list holds ~12k entries and every BM25
#: query scans nearly the whole corpus. An early version of this file did exactly
#: that and reported 25 rps for BM25, which said nothing about BM25 and everything
#: about the fixture.
#:
#: Real text is Zipfian, so terms are drawn from a Zipf distribution over a
#: realistically sized vocabulary. Head terms still get long posting lists — which
#: is the honest hard case — but the average query looks like a real one.
VOCAB_SIZE = 5000
WORDS_PER_DOC = 24
ZIPF_A = 1.3


def _vocabulary(size: int = VOCAB_SIZE) -> list[str]:
    return [f"term{i:05d}" for i in range(size)]


def _sample_terms(rng: np.random.Generator, vocab: list[str], count: int) -> list[str]:
    # Zipf over vocabulary rank, clipped to the vocabulary size.
    idx = np.clip(rng.zipf(ZIPF_A, size=count) - 1, 0, len(vocab) - 1)
    return [vocab[i] for i in idx]


def seed_collection(
    client: PyVecClient, n: int, index: str, seed: int
) -> tuple[np.ndarray, list[str]]:
    from pyvec.client import PyVecHTTPError

    try:
        client.drop_collection(COLLECTION)
    except PyVecHTTPError:
        pass

    client.create_collection(
        COLLECTION, dimension=DIM, metric="cosine", index=index,
        index_params={"M": 16, "ef_construction": 200} if index == "hnsw" else {},
        text_field="content", capacity=n * 2 + 1024,
    )
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    vocab = _vocabulary()
    print(f"  seeding {n:,} vectors ({index}, vocab={len(vocab):,}) ...",
          file=sys.stderr, flush=True)
    started = time.perf_counter()
    for offset in range(0, n, 500):
        chunk = vectors[offset : offset + 500]
        client.insert(
            COLLECTION,
            [
                {
                    "id": f"v{offset + i}",
                    "vector": chunk[i],
                    "metadata": {
                        "content": " ".join(
                            _sample_terms(rng, vocab, WORDS_PER_DOC)
                        ),
                        "bucket": (offset + i) % 4,
                    },
                }
                for i in range(len(chunk))
            ],
        )
    print(f"  seeded in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    return vectors, vocab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=None)
    parser.add_argument("--n", type=int, default=20_000, help="vectors to index")
    parser.add_argument("--index", default="hnsw", choices=["hnsw", "ivf", "flat"])
    parser.add_argument("--concurrency", type=int, nargs="+",
                        default=DEFAULT_CONCURRENCY)
    parser.add_argument("--duration", type=float, default=8.0,
                        help="seconds of measured load per level")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = BenchmarkRun("load_test", seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    queries = rng.normal(size=(256, DIM)).astype(np.float32)

    def payloads_for(endpoint: str, vocab: list[str]) -> list[dict]:
        # Query terms drawn from the same Zipf distribution as the corpus, so a
        # query looks like something a user would actually type against it.
        term_rng = np.random.default_rng(args.seed + 2)
        out = []
        for i, q in enumerate(queries):
            vec = [float(x) for x in q]
            text = " ".join(_sample_terms(term_rng, vocab, 3))
            if endpoint == "query":
                out.append({"vector": vec, "k": args.k,
                            "params": {"ef_search": args.ef_search}})
            elif endpoint == "query/text":
                out.append({"text": text, "k": args.k})
            elif endpoint == "query/hybrid":
                out.append({"vector": vec, "text": text, "k": args.k,
                            "params": {"ef_search": args.ef_search}})
            else:  # filtered dense
                out.append({"vector": vec, "k": args.k, "filter": {"bucket": i % 4},
                            "params": {"ef_search": args.ef_search}})
        return out

    endpoints = ["query", "query/text", "query/hybrid", "query-filtered"]

    def drive(url: str) -> None:
        client = PyVecClient(url, timeout=args.timeout)
        health = client.health()
        print(f"  server: pyvec {health.get('version')}", file=sys.stderr)
        _, vocab = seed_collection(client, args.n, args.index, args.seed)

        baselines: dict[str, float] = {}
        for endpoint in endpoints:
            api_endpoint = "query" if endpoint == "query-filtered" else endpoint
            payloads = payloads_for(endpoint, vocab)
            print(f"\n  --- {endpoint} ---", file=sys.stderr)
            for concurrency in args.concurrency:
                row, _ = run_level(
                    url, api_endpoint, payloads, concurrency,
                    args.duration, args.timeout,
                )
                row["endpoint"] = endpoint
                row["index"] = args.index
                row["n"] = args.n
                row["k"] = args.k
                if concurrency == args.concurrency[0]:
                    baselines[endpoint] = row["rps"]
                base = baselines.get(endpoint, 0) or 1
                # Scaling efficiency: how much of the ideal linear speedup we got.
                ideal = base * concurrency / max(args.concurrency[0], 1)
                row["scaling_efficiency"] = round(row["rps"] / ideal, 3) if ideal else 0
                run.report({}, row)
                print(
                    f"    c={concurrency:3}  {row['rps']:8.1f} rps  "
                    f"p50={row['p50_ms']:7.2f}ms  p95={row['p95_ms']:7.2f}ms  "
                    f"p99={row['p99_ms']:7.2f}ms  errors={row['errors']}  "
                    f"eff={row['scaling_efficiency']:.2f}",
                    file=sys.stderr,
                )
                if row["errors"]:
                    run.note(
                        f"{endpoint} at c={concurrency}: {row['errors']} errors "
                        f"({row['error_samples']})"
                    )

        # Mixed read/write: the RW lock's behaviour is the interesting part.
        print("\n  --- mixed read + write (writer contention) ---", file=sys.stderr)
        _mixed_load(url, args, run, payloads_for("query", vocab), vocab)

    print("=" * 70)
    print("PyVec load test")
    print("=" * 70)
    if args.url:
        print(f"target: {args.url} (external)")
        drive(args.url)
    else:
        import shutil
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="pyvec_load_"))
        port = free_port()
        print(f"target: spawned server on port {port}")
        try:
            with LocalServer(root, port) as url:
                drive(url)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    path = run.save(args.out or "benchmarks/results/load_test.csv")
    print(f"\n-> {path}", file=sys.stderr)
    print()
    run.print_table(
        ["endpoint", "concurrency", "rps", "p50_ms", "p95_ms", "p99_ms",
         "errors", "scaling_efficiency"]
    )
    _summarise(run, args)
    return 0


def _mixed_load(
    url: str, args, run: BenchmarkRun, read_payloads: list[dict], vocab: list[str]
) -> None:
    """Readers under sustained load while a writer inserts continuously.

    Writes take the collection's write lock exclusively, so this quantifies what a
    background ingest costs concurrent search — the question anyone running this in
    production asks first.
    """
    concurrency = max(args.concurrency)
    stop = threading.Event()
    writes = {"count": 0, "errors": 0}

    def writer() -> None:
        i = 0
        rng = np.random.default_rng(999)
        client = KeepAliveClient(url, args.timeout)
        path = f"/collections/{COLLECTION}/insert"
        while not stop.is_set():
            payload = {
                "items": [
                    {
                        "id": f"w{i}-{j}",
                        "vector": [float(x) for x in rng.normal(size=DIM)],
                        "metadata": {
                            "content": " ".join(_sample_terms(rng, vocab, WORDS_PER_DOC)),
                            "bucket": 0,
                        },
                    }
                    for j in range(10)
                ]
            }
            try:
                client.post(path, payload)
                writes["count"] += 10
            except Exception:
                writes["errors"] += 1
            i += 1
        client.close()

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        row, _ = run_level(
            url, "query", read_payloads, concurrency, args.duration, args.timeout
        )
    finally:
        stop.set()
        thread.join(timeout=10)

    row["endpoint"] = "query (with concurrent writer)"
    row["index"] = args.index
    row["n"] = args.n
    row["k"] = args.k
    row["writes_completed"] = writes["count"]
    row["write_errors"] = writes["errors"]
    run.report({}, row)
    print(
        f"    c={concurrency:3}  {row['rps']:8.1f} rps  p50={row['p50_ms']:7.2f}ms  "
        f"p95={row['p95_ms']:7.2f}ms  errors={row['errors']}  "
        f"(writer inserted {writes['count']:,} vectors)",
        file=sys.stderr,
    )


def _summarise(run: BenchmarkRun, args) -> None:
    rows = run.results
    if not rows:
        return
    print()
    for endpoint in dict.fromkeys(r["endpoint"] for r in rows):
        subset = [r for r in rows if r["endpoint"] == endpoint]
        best = max(subset, key=lambda r: r["rps"])
        print(
            f"{endpoint:32} peak {best['rps']:8.1f} rps at c={best['concurrency']}  "
            f"(p95 {best['p95_ms']:.2f}ms)"
        )
    total_errors = sum(r["errors"] for r in rows)
    print(f"\ntotal errors across all levels: {total_errors}")

    plain = [r for r in rows if r["endpoint"] == "query"]
    contended = [r for r in rows if "concurrent writer" in str(r["endpoint"])]
    if plain and contended:
        peak = max(plain, key=lambda r: r["rps"])
        under_write = contended[0]
        if peak["rps"]:
            print(
                f"read throughput under a concurrent writer: "
                f"{under_write['rps']:.1f} rps vs {peak['rps']:.1f} rps idle "
                f"({100 * under_write['rps'] / peak['rps']:.0f}% retained) — "
                f"the write lock serialising against readers"
            )


if __name__ == "__main__":
    raise SystemExit(main())
