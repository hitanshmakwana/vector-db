"""End-to-end demo: insert, query, hybrid search, restart, still works.

This is the script behind the three-minute demo video in the PRD's success
metrics. It runs entirely offline against the embedded API — no server, no
network, no embedding model — so it works on a clean checkout:

    python examples/quickstart.py

For the HTTP version of the same flow, see ``examples/http_demo.py``.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the demo work on a fresh clone, before `pip install -e .`. Running a script
# inside examples/ puts examples/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyvec import Collection  # noqa: E402

# A tiny corpus with deliberate structure: two clear topics, and one document
# ("d7") whose wording overlaps the space topic while its subject is oceans.
DOCS = [
    ("d0", "the spacecraft achieved orbit around the distant planet"),
    ("d1", "rocket engines burn liquid hydrogen and oxygen"),
    ("d2", "astronauts train for months before a mission launch"),
    ("d3", "the telescope observed a supernova in a nearby galaxy"),
    ("d4", "coral reefs support a quarter of all marine species"),
    ("d5", "deep ocean trenches remain largely unexplored"),
    ("d6", "whale migration patterns shift with ocean temperature"),
    ("d7", "the submarine launched from its ocean platform into the deep"),
]

DIM = 32


def fake_embed(text: str, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """A stand-in for a real embedding model.

    Hashes tokens into a bag-of-words vector, so documents sharing vocabulary end
    up near each other. Crude, deterministic, and enough to demonstrate the
    difference between semantic and lexical retrieval without downloading
    ``sentence-transformers`` (which the DB never touches anyway — ADR-008: PyVec
    stores whatever float32 vectors you hand it and generates none of its own).
    """
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        rng = np.random.default_rng(abs(hash(token)) % (2**32) + seed)
        vec += rng.normal(size=dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def show(title: str, hits) -> None:
    print(f"\n{title}")
    for rank, hit in enumerate(hits, start=1):
        ranks = f"   ranks={hit.ranks}" if hit.ranks else ""
        text = hit.metadata.get("content", "")
        print(f"  {rank}. {hit.id}  score={hit.score:.4f}{ranks}\n      {text}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="pyvec_quickstart_"))
    print(f"data directory: {root}")

    try:
        # ---------------------------------------------------------------- #
        # 1. Create a collection
        # ---------------------------------------------------------------- #
        print("\n=== 1. create a collection ===")
        collection = Collection.create(
            "docs",
            root,
            dimension=DIM,
            metric="cosine",          # ADR-009: fixed at create time
            index_type="hnsw",        # or "ivf" / "flat"
            index_params={"M": 16, "ef_construction": 200},
            text_field="content",     # ADR-011: enables BM25 in the same collection
        )
        print(f"  {collection!r}")

        # ---------------------------------------------------------------- #
        # 2. Insert
        # ---------------------------------------------------------------- #
        print("\n=== 2. insert 8 documents ===")
        result = collection.insert(
            [
                {
                    "id": doc_id,
                    "vector": fake_embed(text),
                    "metadata": {
                        "content": text,
                        "topic": "space" if i < 4 else "ocean",
                    },
                }
                for i, (doc_id, text) in enumerate(DOCS)
            ]
        )
        print(f"  {result}  ->  {len(collection)} vectors")

        # ---------------------------------------------------------------- #
        # 3. Dense, sparse, hybrid
        # ---------------------------------------------------------------- #
        print("\n=== 3. the three query paths ===")
        query_text = "deep ocean exploration"
        query_vector = fake_embed(query_text)

        show(
            f"dense (HNSW, cosine) for {query_text!r}:",
            collection.search(query_vector, k=3),
        )
        show(
            f"sparse (BM25) for {query_text!r}:",
            collection.search_text(query_text, k=3),
        )
        show(
            f"hybrid (RRF k=60) for {query_text!r}:",
            collection.search_hybrid(query_vector, query_text, k=3),
        )
        print(
            "\n  Note the per-retriever ranks on the hybrid results: RRF returns no\n"
            "  comparable similarity score, so the ranks are how you explain the\n"
            "  ordering. A document both retrievers liked outranks one that only\n"
            "  either found."
        )

        # ---------------------------------------------------------------- #
        # 4. Metadata filtering
        # ---------------------------------------------------------------- #
        print("\n=== 4. metadata filter ===")
        show(
            "dense search restricted to topic=space:",
            collection.search(query_vector, k=3, filter={"topic": "space"}),
        )

        # ---------------------------------------------------------------- #
        # 5. Delete
        # ---------------------------------------------------------------- #
        print("\n=== 5. delete (tombstone) ===")
        collection.delete("d5")
        print(f"  deleted d5 -> {len(collection)} live, {collection.num_deleted} tombstoned")
        remaining = {h.id for h in collection.search(query_vector, k=8)}
        print(f"  d5 in results? {'d5' in remaining}")

        # ---------------------------------------------------------------- #
        # 6. Restart
        # ---------------------------------------------------------------- #
        print("\n=== 6. close and reopen (PRD UC4) ===")
        before = [h.id for h in collection.search(query_vector, k=3)]
        collection.close()
        print("  closed: checkpointed to disk and WAL truncated")
        print(f"  files: {sorted(p.name for p in (root / 'docs').iterdir())}")

        reopened = Collection.open(root / "docs")
        try:
            after = [h.id for h in reopened.search(query_vector, k=3)]
            print(f"  reopened: {len(reopened)} vectors, recovery={reopened.recovery_report}")
            print(f"  same results as before restart? {before == after}")
            print(f"  d5 still deleted? {not reopened.contains('d5')}")
            print(f"  BM25 still works? {bool(reopened.search_text('ocean', k=1))}")

            # ------------------------------------------------------------ #
            # 7. Compaction
            # ------------------------------------------------------------ #
            print("\n=== 7. optimize (compact tombstones) ===")
            print(f"  {reopened.optimize()}")
            print(f"  {len(reopened)} vectors, {reopened.num_deleted} tombstoned")

            stats = reopened.stats()
            print("\n=== stats ===")
            for key in ("num_vectors", "num_deleted", "memory_bytes", "disk_bytes"):
                print(f"  {key}: {stats[key]:,}")
            print(f"  index: {stats['index']}")
        finally:
            reopened.close()

        print("\nDone.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
