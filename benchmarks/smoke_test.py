"""Smoke test: does a deployment actually work, end to end?

Not a substitute for the unit suite — this is the "is the thing alive and correct
in the shape a user meets it" check you run against a freshly started server, or
after a deploy, or in CI before a release. It exercises the documented happy path
plus the error contract, over real HTTP, and exits non-zero on the first failure.

    python -m benchmarks.smoke_test                       # spawns its own server
    python -m benchmarks.smoke_test --url http://host:8080  # test a live one

Every check prints PASS/FAIL with what it verified, so a failure tells you which
guarantee broke rather than just that something did.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyvec.client import PyVecClient, PyVecHTTPError  # noqa: E402

DIM = 32
COLLECTION = "smoke"

DOCS = [
    ("s0", "the spacecraft achieved orbit around a distant planet", "space"),
    ("s1", "rocket engines burn liquid hydrogen and oxygen", "space"),
    ("s2", "astronauts train for months before a mission launch", "space"),
    ("s3", "coral reefs support a quarter of all marine species", "ocean"),
    ("s4", "deep ocean trenches remain largely unexplored", "ocean"),
    ("s5", "whale migration shifts with ocean temperature", "ocean"),
]


class Checker:
    """Accumulates pass/fail results so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append(name)
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))
        return bool(condition)

    def expect_error(self, name: str, fn, code: str, status: int) -> bool:
        try:
            fn()
        except PyVecHTTPError as exc:
            return self.check(
                name,
                exc.code == code and exc.status == status,
                f"got {exc.status} {exc.code}, wanted {status} {code}",
            )
        return self.check(name, False, "no error raised")

    def report(self) -> int:
        total = self.passed + len(self.failed)
        print(f"\n{self.passed}/{total} checks passed")
        if self.failed:
            print("failed: " + ", ".join(self.failed))
            return 1
        return 0


def embed(text: str, dim: int = DIM) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        rng = np.random.default_rng(abs(hash(token)) % (2**32))
        vec += rng.normal(size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm else vec).tolist()


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


def run_checks(client: PyVecClient, c: Checker) -> None:
    # -- liveness ---------------------------------------------------------- #
    print("\n[1] liveness")
    health = client.health()
    c.check("GET /health returns ok", health.get("status") == "ok",
            f"version {health.get('version')}")
    c.check("GET /metrics is prometheus text",
            "pyvec_uptime_seconds" in client.metrics())

    # Clean up a previous run so the smoke test is repeatable against a live host.
    try:
        client.drop_collection(COLLECTION)
    except PyVecHTTPError:
        pass

    # -- create ------------------------------------------------------------ #
    print("\n[2] collection lifecycle")
    created = client.create_collection(
        COLLECTION, dimension=DIM, metric="cosine", index="hnsw",
        index_params={"M": 16, "ef_construction": 200}, text_field="content",
    )
    c.check("POST /collections returns 201 body", created.get("name") == COLLECTION)
    c.check("collection appears in list",
            any(r["name"] == COLLECTION for r in client.list_collections()))
    detail = client.describe(COLLECTION)
    c.check("describe reports the configuration",
            detail["dimension"] == DIM and detail["metric"] == "cosine"
            and detail["index"]["type"] == "hnsw",
            f"index={detail['index']}")

    # -- insert ------------------------------------------------------------ #
    print("\n[3] insert")
    result = client.insert(
        COLLECTION,
        [
            {"id": doc_id, "vector": embed(text),
             "metadata": {"content": text, "topic": topic}}
            for doc_id, text, topic in DOCS
        ],
    )
    c.check("insert reports the right count", result["inserted"] == len(DOCS),
            f"{result}")
    c.check("num_vectors reflects the insert",
            client.describe(COLLECTION)["num_vectors"] == len(DOCS))

    # -- retrieval --------------------------------------------------------- #
    print("\n[4] the three query paths")
    query_text = "deep ocean exploration"
    query_vector = embed(query_text)

    dense = client.query(COLLECTION, query_vector, k=3)
    c.check("dense search returns k results", len(dense) == 3)
    c.check("dense scores descend for cosine",
            [h["score"] for h in dense] == sorted((h["score"] for h in dense),
                                                  reverse=True))
    c.check("dense results carry metadata",
            all("content" in h["metadata"] for h in dense))

    exact = client.query(COLLECTION, embed(DOCS[4][1]), k=1)
    c.check("a stored vector is its own nearest neighbour",
            exact[0]["id"] == "s4" and exact[0]["score"] > 0.99,
            f"{exact[0]['id']} @ {exact[0]['score']:.4f}")

    sparse = client.query_text(COLLECTION, query_text, k=3)
    c.check("BM25 search returns results", len(sparse) > 0)
    c.check("BM25 ranks the lexical match first", sparse[0]["id"] == "s4",
            f"got {sparse[0]['id']}")

    hybrid = client.hybrid(COLLECTION, query_vector, query_text, k=3)
    c.check("hybrid returns results", len(hybrid) == 3)
    c.check("hybrid exposes per-retriever ranks",
            all("dense_rank" in h and "sparse_rank" in h for h in hybrid))
    c.check("hybrid returns rrf_score, not a similarity",
            "rrf_score" in hybrid[0] and "score" not in hybrid[0])
    c.check("a doc both retrievers found ranks first",
            hybrid[0]["dense_rank"] is not None
            and hybrid[0]["sparse_rank"] is not None,
            f"{hybrid[0]['id']} ranks d={hybrid[0]['dense_rank']} "
            f"s={hybrid[0]['sparse_rank']}")

    # -- filters ----------------------------------------------------------- #
    print("\n[5] metadata filtering")
    filtered = client.query(COLLECTION, query_vector, k=5, filter={"topic": "space"})
    c.check("filter restricts results",
            filtered and all(h["metadata"]["topic"] == "space" for h in filtered),
            f"{[h['id'] for h in filtered]}")
    c.check("unmatchable filter returns empty",
            client.query(COLLECTION, query_vector, k=5,
                         filter={"topic": "nonexistent"}) == [])

    # -- point ops --------------------------------------------------------- #
    print("\n[6] get and delete")
    got = client.get(COLLECTION, "s0")
    c.check("get returns the vector and metadata",
            got["id"] == "s0" and len(got["vector"]) == DIM
            and got["metadata"]["topic"] == "space")
    client.delete(COLLECTION, "s0")
    c.check("delete removes it from the count",
            client.describe(COLLECTION)["num_vectors"] == len(DOCS) - 1)
    c.check("deleted id is absent from results",
            "s0" not in {h["id"] for h in client.query(COLLECTION, embed(DOCS[0][1]), k=6)})

    # -- error contract ---------------------------------------------------- #
    print("\n[7] error contract (API_SPEC codes)")
    c.expect_error("unknown collection -> 404 COLLECTION_NOT_FOUND",
                   lambda: client.describe("no-such-collection"),
                   "COLLECTION_NOT_FOUND", 404)
    c.expect_error("unknown id -> 404 ID_NOT_FOUND",
                   lambda: client.get(COLLECTION, "no-such-id"),
                   "ID_NOT_FOUND", 404)
    c.expect_error("deleted id -> 404 ID_NOT_FOUND",
                   lambda: client.get(COLLECTION, "s0"),
                   "ID_NOT_FOUND", 404)
    c.expect_error("duplicate id -> 409 ID_EXISTS",
                   lambda: client.insert(COLLECTION,
                                         [{"id": "s1", "vector": embed("x")}]),
                   "ID_EXISTS", 409)
    c.expect_error("wrong dimension -> 400 INVALID_DIMENSION",
                   lambda: client.insert(COLLECTION,
                                         [{"id": "bad", "vector": [1.0, 2.0]}]),
                   "INVALID_DIMENSION", 400)

    # -- maintenance ------------------------------------------------------- #
    print("\n[8] maintenance")
    optimised = client.optimize(COLLECTION)
    c.check("optimize compacts the tombstone", optimised["compacted"] == 1,
            f"{optimised}")
    c.check("data is queryable after optimize",
            len(client.query(COLLECTION, query_vector, k=3)) == 3)
    c.check("bm25 works after optimize",
            len(client.query_text(COLLECTION, "ocean", k=3)) > 0)
    snap = client.snapshot(COLLECTION)
    c.check("snapshot returns an id and path",
            bool(snap.get("snapshot_id")) and bool(snap.get("path")))

    # -- upsert ------------------------------------------------------------ #
    print("\n[9] upsert")
    client.insert(COLLECTION, [{"id": "s1", "vector": embed("replaced text"),
                                "metadata": {"content": "replaced text"}}],
                  upsert=True)
    c.check("upsert overwrote the metadata",
            client.get(COLLECTION, "s1")["metadata"]["content"] == "replaced text")
    c.check("upsert did not change the count",
            client.describe(COLLECTION)["num_vectors"] == len(DOCS) - 1,
            f"{client.describe(COLLECTION)['num_vectors']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=None,
                        help="test a running server instead of spawning one")
    parser.add_argument("--keep", action="store_true",
                        help="do not drop the smoke collection at the end")
    args = parser.parse_args(argv)

    checker = Checker()
    started = time.time()
    print("=" * 66)
    print("PyVec smoke test")
    print("=" * 66)

    if args.url:
        print(f"target: {args.url} (external)")
        client = PyVecClient(args.url)
        try:
            run_checks(client, checker)
        finally:
            if not args.keep:
                try:
                    client.drop_collection(COLLECTION)
                except PyVecHTTPError:
                    pass
    else:
        import shutil
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="pyvec_smoke_"))
        port = free_port()
        print(f"target: spawned server on port {port}, data in {root}")
        try:
            with LocalServer(root, port) as url:
                client = PyVecClient(url)
                run_checks(client, checker)

                # Restart: the durability guarantee is part of the happy path.
                print("\n[10] restart (PRD UC4)")
                before = [h["id"] for h in client.query(
                    COLLECTION, embed("deep ocean exploration"), k=3)]
                count = client.describe(COLLECTION)["num_vectors"]

            with LocalServer(root, free_port()) as url:
                client = PyVecClient(url)
                detail = client.describe(COLLECTION)
                checker.check("collection reopened after restart",
                              detail["num_vectors"] == count,
                              f"{detail['num_vectors']} vectors")
                after = [h["id"] for h in client.query(
                    COLLECTION, embed("deep ocean exploration"), k=3)]
                checker.check("same results after restart", before == after)
                checker.check("bm25 index survived restart",
                              len(client.query_text(COLLECTION, "ocean", k=3)) > 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    print(f"\nelapsed {time.time() - started:.1f}s")
    return checker.report()


if __name__ == "__main__":
    raise SystemExit(main())
