"""Benchmark 3: hybrid vs dense-only vs BM25-only.

BENCHMARKS.md setup: 100k MS MARCO passages embedded with ``all-MiniLM-L6-v2``,
the dev query set, comparing

* pure dense (HNSW, ``ef_search=64``)
* pure BM25
* hybrid (RRF, ``k=60``, 50 candidates from each side)

reported as nDCG@10, MRR@10 and Recall@100.

**Expected result:** hybrid > dense > BM25 on nDCG, with hybrid typically 3-8
nDCG points above dense-only. If there is no lift, something is wired wrong —
BM25 tokenisation or the RRF plumbing are the usual suspects. That failure mode is
worth more than the number itself, so this script prints the per-system breakdown
rather than just the winner.

    python -m benchmarks.hybrid_msmarco --synthetic
    python -m benchmarks.hybrid_msmarco --passages 100000
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.datasets import load_msmarco
from benchmarks.harness import BenchmarkRun, mrr_at_k, ndcg_at_k
from pyvec.core.collection import MAX_BATCH, Collection

K = 10
RECALL_K = 100
DENSE_CANDIDATES = 50
SPARSE_CANDIDATES = 50
RRF_K = 60


def evaluate(
    name: str,
    search,
    queries: list[dict],
    run: BenchmarkRun,
    extra: dict | None = None,
) -> dict:
    """Score one retrieval system across all queries."""
    ndcgs, mrrs, recalls = [], [], []
    latency, results = run.measure_each(
        search, queries, warmup=min(20, len(queries)), collect=True
    )
    for query, hits in zip(queries, results):
        ids = [h.id for h in hits]
        relevance = query["relevant"]
        ndcgs.append(ndcg_at_k(ids, relevance, K))
        mrrs.append(mrr_at_k(ids, set(relevance), K))
        found = len(set(ids[:RECALL_K]) & set(relevance))
        recalls.append(found / len(relevance) if relevance else 0.0)

    metrics = {
        f"ndcg@{K}": round(float(np.mean(ndcgs)), 4),
        f"mrr@{K}": round(float(np.mean(mrrs)), 4),
        f"recall@{RECALL_K}": round(float(np.mean(recalls)), 4),
        **latency,
    }
    row = run.report({"system": name, "num_queries": len(queries), **(extra or {})}, metrics)
    print(
        f"  {name:16} nDCG@{K}={metrics[f'ndcg@{K}']:.4f}  "
        f"MRR@{K}={metrics[f'mrr@{K}']:.4f}  "
        f"Recall@{RECALL_K}={metrics[f'recall@{RECALL_K}']:.4f}  "
        f"p95={metrics['p95_ms']:.2f}ms",
        file=sys.stderr,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--synthetic", action="store_true",
                        help="use the synthetic text corpus, no model download")
    parser.add_argument("--passages", type=int, default=100_000)
    parser.add_argument("--index", default="hnsw", choices=["hnsw", "ivf", "flat"])
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--dense-candidates", type=int, default=DENSE_CANDIDATES)
    parser.add_argument("--sparse-candidates", type=int, default=SPARSE_CANDIDATES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries", type=int, default=None,
                        help="evaluate only the first N queries")
    parser.add_argument("--sweep", action="store_true",
                        help="diagnose fusion: sweep rrf_k and candidate depth, "
                             "and report what weighted fusion would achieve")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = BenchmarkRun("hybrid_msmarco", seed=args.seed)
    passages, queries = load_msmarco(
        n_passages=args.passages, synthetic=args.synthetic, seed=args.seed
    )
    if args.synthetic:
        run.note(
            "SYNTHETIC text corpus: this validates the evaluation pipeline and "
            "demonstrates the hybrid-over-dense effect, but it is NOT an MS MARCO "
            "number and must not be reported as one."
        )
    if not queries:
        print("no queries with relevance judgements", file=sys.stderr)
        return 1
    if args.queries:
        queries = queries[: args.queries]

    dim = int(np.asarray(passages[0]["vector"]).shape[0])
    print(
        f"{len(passages):,} passages, {len(queries):,} queries, dim={dim}",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="pyvec_hybrid_") as tmp:
        collection = Collection.create(
            "msmarco",
            Path(tmp),
            dimension=dim,
            metric="cosine",
            index_type=args.index,
            text_field="content",
            capacity=len(passages) + 16,
            # The corpus is loaded once and never mutated, so per-insert fsync
            # would dominate the setup for no benefit to the measurement.
            fsync_policy="batch",
        )
        try:
            with run.time_block(f"index {len(passages):,} passages ({args.index})"):
                for start in range(0, len(passages), MAX_BATCH):
                    collection.insert(
                        [
                            {
                                "id": p["id"],
                                "vector": p["vector"],
                                "metadata": {"content": p["text"]},
                            }
                            for p in passages[start : start + MAX_BATCH]
                        ]
                    )

            print("\n=== Benchmark 3: hybrid vs dense vs BM25 ===", file=sys.stderr)
            fetch = max(RECALL_K, K)

            evaluate(
                "dense",
                lambda q: collection.search(
                    q["vector"], k=fetch, params={"ef_search": args.ef_search}
                ),
                queries, run,
                {"index": args.index, "ef_search": args.ef_search},
            )
            evaluate(
                "bm25",
                lambda q: collection.search_text(q["text"], k=fetch),
                queries, run, {"index": "bm25"},
            )
            evaluate(
                "hybrid-rrf",
                lambda q: collection.search_hybrid(
                    q["vector"], q["text"], k=fetch,
                    params={
                        "ef_search": args.ef_search,
                        "dense_candidates": max(args.dense_candidates, fetch),
                        "sparse_candidates": max(args.sparse_candidates, fetch),
                        "rrf_k": args.rrf_k,
                    },
                ),
                queries, run,
                {"index": args.index, "rrf_k": args.rrf_k,
                 "dense_candidates": args.dense_candidates,
                 "sparse_candidates": args.sparse_candidates},
            )

            if args.sweep:
                _diagnose_fusion(collection, queries, run, args, fetch)
        finally:
            collection.close()

    path = run.save(args.out)
    print(f"\n-> {path}", file=sys.stderr)
    print()
    run.print_table(
        ["system", f"ndcg@{K}", f"mrr@{K}", f"recall@{RECALL_K}",
         "qps", "p50_ms", "p95_ms"]
    )
    _summarise(run, synthetic=args.synthetic)
    return 0


def _diagnose_fusion(collection, queries, run: BenchmarkRun, args, fetch: int) -> None:
    """Why did fusion lose? Sweep the knobs, then test the weighting hypothesis.

    Run when the headline hybrid number comes in *below* a single retriever. Three
    things could explain that, and they need separating:

    1. **``rrf_k`` is wrong.** It controls how sharply rank 1 is favoured over the
       tail. Too large flattens the curve so a mediocre hit from either side scores
       nearly as much as a top hit.
    2. **The candidate pools are too shallow**, so fusion never sees the documents
       that would rescue it.
    3. **The retrievers are of unequal strength**, and unweighted RRF averages the
       strong one down toward the weak one. This is the interesting case, because it
       is a property of RRF rather than of the configuration.

    For (3) the fusion is recomputed here with a weight on each side. That is
    deliberately *not* a PyVec feature — ADR-003 chose plain RRF and explicitly
    deferred a weighting knob — so it is computed in the benchmark, from the same
    two ranked lists the engine returns, as evidence about whether that deferral was
    the right call.
    """
    from pyvec.fusion.rrf import reciprocal_rank_fusion

    print("\n=== Fusion diagnosis ===", file=sys.stderr)

    # Pull each side's ranking once; every variant below re-fuses the same lists, so
    # the comparison isolates fusion from retrieval.
    dense_lists, sparse_lists, relevances = [], [], []
    for query in queries:
        dense = collection.search(
            query["vector"], k=fetch, params={"ef_search": args.ef_search}
        )
        sparse = collection.search_text(query["text"], k=fetch)
        dense_lists.append([h.id for h in dense])
        sparse_lists.append([h.id for h in sparse])
        relevances.append(query["relevant"])

    def score(fused_lists) -> tuple[float, float]:
        ndcgs = [ndcg_at_k(ids, rel, K) for ids, rel in zip(fused_lists, relevances)]
        mrrs = [mrr_at_k(ids, set(rel), K) for ids, rel in zip(fused_lists, relevances)]
        return float(np.mean(ndcgs)), float(np.mean(mrrs))

    baseline_dense = score(dense_lists)
    baseline_sparse = score(sparse_lists)
    print(
        f"  dense-only  nDCG@{K}={baseline_dense[0]:.4f}   "
        f"BM25-only nDCG@{K}={baseline_sparse[0]:.4f}",
        file=sys.stderr,
    )

    # --- (1) rrf_k sweep ------------------------------------------------- #
    for rrf_k in (1, 10, 60, 200, 1000):
        fused = [
            [r.id for r in reciprocal_rank_fusion(
                {"dense": d, "sparse": s}, k=rrf_k, top_k=fetch)]
            for d, s in zip(dense_lists, sparse_lists)
        ]
        ndcg, mrr = score(fused)
        run.report(
            {"system": f"hybrid-rrf-k{rrf_k}", "variant": "rrf_k sweep",
             "rrf_k": rrf_k, "num_queries": len(queries)},
            {f"ndcg@{K}": round(ndcg, 4), f"mrr@{K}": round(mrr, 4)},
        )
        delta = (ndcg - baseline_dense[0]) * 100
        print(
            f"  rrf_k={rrf_k:5}  nDCG@{K}={ndcg:.4f}  MRR@{K}={mrr:.4f}  "
            f"({delta:+.1f} vs dense)",
            file=sys.stderr,
        )

    # --- (2) candidate depth --------------------------------------------- #
    for depth in (10, 25, 50, 100):
        if depth > fetch:
            continue
        fused = [
            [r.id for r in reciprocal_rank_fusion(
                {"dense": d[:depth], "sparse": s[:depth]}, k=args.rrf_k, top_k=fetch)]
            for d, s in zip(dense_lists, sparse_lists)
        ]
        ndcg, mrr = score(fused)
        run.report(
            {"system": f"hybrid-depth{depth}", "variant": "candidate depth",
             "dense_candidates": depth, "sparse_candidates": depth,
             "num_queries": len(queries)},
            {f"ndcg@{K}": round(ndcg, 4), f"mrr@{K}": round(mrr, 4)},
        )
        print(
            f"  depth={depth:5}  nDCG@{K}={ndcg:.4f}  "
            f"({(ndcg - baseline_dense[0]) * 100:+.1f} vs dense)",
            file=sys.stderr,
        )

    # --- (3) weighted fusion --------------------------------------------- #
    print("  weighted RRF (not a PyVec feature — see ADR-003):", file=sys.stderr)
    best = (None, -1.0)
    for weight in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        fused = []
        for d, s in zip(dense_lists, sparse_lists):
            scores: dict[str, float] = {}
            for rank, doc_id in enumerate(d, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (args.rrf_k + rank)
            for rank, doc_id in enumerate(s, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1 - weight) / (
                    args.rrf_k + rank
                )
            fused.append(
                [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
            )
        ndcg, mrr = score(fused)
        run.report(
            {"system": f"hybrid-weighted-{weight:g}", "variant": "weighted fusion",
             "dense_weight": weight, "num_queries": len(queries)},
            {f"ndcg@{K}": round(ndcg, 4), f"mrr@{K}": round(mrr, 4)},
        )
        if ndcg > best[1]:
            best = (weight, ndcg)
        print(
            f"    dense weight={weight:.1f}  nDCG@{K}={ndcg:.4f}  "
            f"({(ndcg - baseline_dense[0]) * 100:+.1f} vs dense)",
            file=sys.stderr,
        )
    print(
        f"  best weighting: dense={best[0]:.1f} at nDCG@{K}={best[1]:.4f} "
        f"({(best[1] - baseline_dense[0]) * 100:+.1f} vs dense-only)",
        file=sys.stderr,
    )


def _summarise(run: BenchmarkRun, synthetic: bool = False) -> None:
    by_system = {r["system"]: r for r in run.results}
    dense = by_system.get("dense")
    hybrid = by_system.get("hybrid-rrf")
    bm25 = by_system.get("bm25")
    if not (dense and hybrid):
        return

    lift = (hybrid[f"ndcg@{K}"] - dense[f"ndcg@{K}"]) * 100
    print()
    print(
        f"hybrid nDCG@{K} = {hybrid[f'ndcg@{K}']:.4f} vs dense "
        f"{dense[f'ndcg@{K}']:.4f}  ->  {lift:+.1f} nDCG points"
    )
    if bm25:
        print(f"BM25-only nDCG@{K} = {bm25[f'ndcg@{K}']:.4f}")

        # The claim that generalises: fusion should beat *either* retriever alone.
        best_single = max(dense[f"ndcg@{K}"], bm25[f"ndcg@{K}"])
        beats_both = hybrid[f"ndcg@{K}"] >= best_single
        print(
            "hybrid beats both single retrievers: "
            + ("yes" if beats_both else "NO — fusion is losing, investigate")
        )

        # "hybrid > dense > BM25" is specifically the MS MARCO expectation from
        # BENCHMARKS.md, where the embedding model is strong. The synthetic corpus
        # is built with a deliberately weak semantic signal, so dense placing last
        # there is by construction, not a defect.
        if not synthetic:
            ordered = hybrid[f"ndcg@{K}"] >= dense[f"ndcg@{K}"] >= bm25[f"ndcg@{K}"]
            print(
                "ordering hybrid > dense > BM25: "
                + ("as expected" if ordered else "NOT as expected — investigate")
            )
        else:
            print(
                "(synthetic corpus: dense is weak by construction, so the "
                "MS MARCO 'dense > BM25' expectation does not apply)"
            )
    if lift <= 0:
        print(
            "\nNo lift from fusion. Before trusting this, check: (1) BM25 is "
            "actually matching — is text_field populated? (2) both retrievers "
            "return enough candidates, (3) the query text and vector describe the "
            "same information need.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
