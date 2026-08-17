"""The same flow as ``quickstart.py``, but over HTTP.

Starts a real server on a free port, drives it with the client SDK, restarts it,
and shows the data is still there. This is what a downstream RAG app would
actually do (PRD G4).

    python examples/http_demo.py
"""

from __future__ import annotations

import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.quickstart import DIM, DOCS, fake_embed  # noqa: E402
from pyvec.api.server import create_app  # noqa: E402
from pyvec.client import PyVecClient  # noqa: E402
from pyvec.core.collection_manager import CollectionManager  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    """A uvicorn server in a background thread, as a context manager."""

    def __init__(self, data_dir: Path, port: int) -> None:
        import uvicorn

        self.port = port
        app = create_app(manager=CollectionManager(data_dir))
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server failed to start")
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *_exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=15)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="pyvec_http_demo_"))
    port = free_port()
    query_text = "deep ocean exploration"
    query_vector = fake_embed(query_text)

    try:
        with Server(root, port) as url:
            print(f"server up at {url}")
            client = PyVecClient(url)
            print(f"  health: {client.health()}")

            print("\n=== create ===")
            print(
                f"  {client.create_collection('docs', dimension=DIM, metric='cosine', index='hnsw', text_field='content')}"
            )

            print("\n=== insert ===")
            items = [
                {
                    "id": doc_id,
                    "vector": fake_embed(text),
                    "metadata": {"content": text,
                                 "topic": "space" if i < 4 else "ocean"},
                }
                for i, (doc_id, text) in enumerate(DOCS)
            ]
            print(f"  {client.insert('docs', items)}")

            print("\n=== query (dense) ===")
            for hit in client.query("docs", query_vector, k=3):
                print(f"  {hit['id']}  score={hit['score']:.4f}")

            print("\n=== query/text (BM25) ===")
            for hit in client.query_text("docs", query_text, k=3):
                print(f"  {hit['id']}  score={hit['score']:.4f}")

            print("\n=== query/hybrid (RRF) ===")
            for hit in client.hybrid("docs", query_vector, query_text, k=3):
                print(
                    f"  {hit['id']}  rrf={hit['rrf_score']:.5f}  "
                    f"dense_rank={hit['dense_rank']}  sparse_rank={hit['sparse_rank']}"
                )

            print("\n=== filter ===")
            hits = client.query("docs", query_vector, k=3, filter={"topic": "ocean"})
            print(f"  topic=ocean -> {[h['id'] for h in hits]}")

            print("\n=== delete ===")
            client.delete("docs", "d5")
            print(f"  d5 deleted; {client.describe('docs')['num_vectors']} vectors left")

            before = [h["id"] for h in client.query("docs", query_vector, k=3)]
            print("\nshutting the server down (no explicit checkpoint call)")

        # A second server over the same directory: a genuine process-level restart
        # as far as the collection is concerned.
        print("\n=== restart ===")
        with Server(root, free_port()) as url:
            client = PyVecClient(url)
            detail = client.describe("docs")
            print(f"  reopened: {detail['num_vectors']} vectors, index={detail['index']}")
            after = [h["id"] for h in client.query("docs", query_vector, k=3)]
            print(f"  same top-3 as before restart? {before == after}")
            print(f"  BM25 still works? {bool(client.query_text('docs', 'ocean', k=1))}")
            print(f"  optimize: {client.optimize('docs')}")
            print(f"  snapshot: {client.snapshot('docs')['snapshot_id']}")

        print("\nDone.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
