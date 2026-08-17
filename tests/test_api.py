"""HTTP API — every endpoint and every error code in API_SPEC.md."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pyvec.api.server import create_app
from pyvec.core.collection_manager import CollectionManager
from tests.conftest import TEXT_CORPUS

DIM = 8


@pytest.fixture
def client(data_root):
    app = create_app(manager=CollectionManager(data_root))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def docs(client):
    """A ``docs`` collection with a small text+vector corpus loaded."""
    client.post(
        "/collections",
        json={
            "name": "docs",
            "dimension": DIM,
            "metric": "cosine",
            "index": {"type": "flat"},
            "text_field": "content",
        },
    )
    rng = np.random.default_rng(3)
    items = [
        {
            "id": f"d{i}",
            "vector": rng.normal(size=DIM).astype(np.float32).tolist(),
            "metadata": {
                "content": TEXT_CORPUS[i % len(TEXT_CORPUS)],
                "category": "animals" if i % 2 == 0 else "tech",
                "n": i,
            },
        }
        for i in range(16)
    ]
    client.post("/collections/docs/insert", json={"items": items})
    return items


class TestCreateCollection:
    def test_returns_201_with_the_documented_body(self, client):
        r = client.post(
            "/collections",
            json={"name": "c1", "dimension": 4, "metric": "cosine",
                  "index": {"type": "hnsw", "params": {"M": 16, "ef_construction": 200}}},
        )
        assert r.status_code == 201
        assert r.json()["name"] == "c1"
        assert r.json()["created_at"].endswith("Z")

    def test_defaults_to_cosine_and_hnsw(self, client):
        client.post("/collections", json={"name": "c1", "dimension": 4})
        detail = client.get("/collections/c1").json()
        assert detail["metric"] == "cosine"
        assert detail["index"]["type"] == "hnsw"

    @pytest.mark.parametrize("index_type", ["hnsw", "ivf", "flat"])
    def test_every_index_type_is_accepted(self, client, index_type):
        r = client.post(
            "/collections",
            json={"name": f"c-{index_type}", "dimension": 4,
                  "index": {"type": index_type}},
        )
        assert r.status_code == 201

    def test_unknown_metric_is_rejected(self, client):
        r = client.post(
            "/collections", json={"name": "c", "dimension": 4, "metric": "manhattan"}
        )
        assert r.status_code == 400
        assert r.json()["code"] == "INVALID_REQUEST"

    def test_unknown_index_type_is_rejected(self, client):
        r = client.post(
            "/collections",
            json={"name": "c", "dimension": 4, "index": {"type": "magic"}},
        )
        assert r.status_code == 400

    def test_non_positive_dimension_is_rejected(self, client):
        assert client.post("/collections", json={"name": "c", "dimension": 0}).status_code == 400

    def test_duplicate_name_conflicts(self, client):
        client.post("/collections", json={"name": "c", "dimension": 4})
        r = client.post("/collections", json={"name": "c", "dimension": 4})
        assert r.status_code == 409
        assert r.json()["code"] == "COLLECTION_EXISTS"

    def test_unsafe_name_is_rejected(self, client):
        r = client.post("/collections", json={"name": "../escape", "dimension": 4})
        assert r.status_code == 400

    def test_unknown_field_is_rejected(self, client):
        r = client.post(
            "/collections", json={"name": "c", "dimension": 4, "typo": True}
        )
        assert r.status_code == 400


class TestListAndDescribe:
    def test_list_is_empty_initially(self, client):
        assert client.get("/collections").json() == {"collections": []}

    def test_list_reports_the_documented_summary(self, client, docs):
        rows = client.get("/collections").json()["collections"]
        assert len(rows) == 1
        assert rows[0]["name"] == "docs"
        assert rows[0]["num_vectors"] == 16
        assert rows[0]["dimension"] == DIM

    def test_describe_reports_the_documented_fields(self, client, docs):
        body = client.get("/collections/docs").json()
        for key in (
            "name", "dimension", "metric", "index", "text_field",
            "num_vectors", "num_deleted", "memory_bytes", "disk_bytes",
        ):
            assert key in body
        assert body["num_vectors"] == 16
        assert body["text_field"] == "content"

    def test_describe_unknown_collection_is_404(self, client):
        r = client.get("/collections/nope")
        assert r.status_code == 404
        assert r.json()["code"] == "COLLECTION_NOT_FOUND"

    def test_drop_returns_204_and_removes_it(self, client, docs):
        assert client.delete("/collections/docs").status_code == 204
        assert client.get("/collections/docs").status_code == 404

    def test_drop_unknown_collection_is_404(self, client):
        assert client.delete("/collections/nope").status_code == 404


class TestInsert:
    def test_returns_201_with_counts(self, client, docs):
        r = client.post(
            "/collections/docs/insert",
            json={"items": [{"id": "new", "vector": [1.0] * DIM}]},
        )
        assert r.status_code == 201
        assert r.json() == {"inserted": 1, "duplicates_skipped": 0}

    def test_batch_insert(self, client):
        client.post("/collections", json={"name": "c", "dimension": DIM})
        items = [{"id": f"b{i}", "vector": [float(i)] * DIM} for i in range(100)]
        r = client.post("/collections/c/insert", json={"items": items})
        assert r.status_code == 201 and r.json()["inserted"] == 100

    def test_wrong_dimension_is_400_with_the_documented_code(self, client, docs):
        r = client.post(
            "/collections/docs/insert",
            json={"items": [{"id": "bad", "vector": [1.0, 2.0]}]},
        )
        assert r.status_code == 400
        assert r.json()["code"] == "INVALID_DIMENSION"

    def test_duplicate_id_is_409(self, client, docs):
        r = client.post(
            "/collections/docs/insert",
            json={"items": [{"id": "d0", "vector": [1.0] * DIM}]},
        )
        assert r.status_code == 409
        assert r.json()["code"] == "ID_EXISTS"

    def test_upsert_overwrites_instead_of_conflicting(self, client, docs):
        r = client.post(
            "/collections/docs/insert",
            json={"items": [{"id": "d0", "vector": [1.0] * DIM,
                             "metadata": {"content": "replaced"}}],
                  "upsert": True},
        )
        assert r.status_code == 201
        assert client.get("/collections/docs/vectors/d0").json()["metadata"] == {
            "content": "replaced"
        }

    def test_oversized_batch_is_413(self, client, docs):
        items = [{"id": f"x{i}", "vector": [1.0] * DIM} for i in range(1001)]
        r = client.post("/collections/docs/insert", json={"items": items})
        assert r.status_code == 413
        assert r.json()["code"] == "PAYLOAD_TOO_LARGE"

    def test_empty_batch_is_rejected(self, client, docs):
        assert client.post("/collections/docs/insert", json={"items": []}).status_code == 400

    def test_insert_into_unknown_collection_is_404(self, client):
        r = client.post(
            "/collections/nope/insert",
            json={"items": [{"id": "a", "vector": [1.0]}]},
        )
        assert r.status_code == 404

    def test_malformed_body_is_400(self, client, docs):
        assert client.post("/collections/docs/insert", json={"nope": 1}).status_code == 400


class TestQuery:
    def test_returns_the_documented_shape(self, client, docs):
        r = client.post(
            "/collections/docs/query", json={"vector": docs[0]["vector"], "k": 3}
        )
        assert r.status_code == 200
        body = r.json()
        assert "took_ms" in body and isinstance(body["took_ms"], float)
        assert len(body["results"]) == 3
        for hit in body["results"]:
            assert set(hit) == {"id", "score", "metadata"}

    def test_nearest_is_the_query_itself(self, client, docs):
        body = client.post(
            "/collections/docs/query", json={"vector": docs[5]["vector"], "k": 1}
        ).json()
        assert body["results"][0]["id"] == "d5"
        assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_k_defaults_to_ten(self, client, docs):
        body = client.post(
            "/collections/docs/query", json={"vector": docs[0]["vector"]}
        ).json()
        assert len(body["results"]) == 10

    def test_filter_is_applied(self, client, docs):
        body = client.post(
            "/collections/docs/query",
            json={"vector": docs[0]["vector"], "k": 5,
                  "filter": {"category": "animals"}},
        ).json()
        assert body["results"]
        assert all(h["metadata"]["category"] == "animals" for h in body["results"])

    def test_index_params_are_forwarded(self, client):
        client.post(
            "/collections",
            json={"name": "h", "dimension": DIM, "index": {"type": "hnsw"}},
        )
        items = [{"id": f"h{i}", "vector": [float(i % 7), 1.0] + [0.0] * (DIM - 2)}
                 for i in range(50)]
        client.post("/collections/h/insert", json={"items": items})
        r = client.post(
            "/collections/h/query",
            json={"vector": items[0]["vector"], "k": 5, "params": {"ef_search": 128}},
        )
        assert r.status_code == 200 and len(r.json()["results"]) == 5

    def test_wrong_query_dimension_is_400(self, client, docs):
        r = client.post("/collections/docs/query", json={"vector": [1.0, 2.0]})
        assert r.status_code == 400
        assert r.json()["code"] == "INVALID_DIMENSION"

    def test_k_zero_is_rejected(self, client, docs):
        r = client.post(
            "/collections/docs/query", json={"vector": docs[0]["vector"], "k": 0}
        )
        assert r.status_code == 400

    def test_query_unknown_collection_is_404(self, client):
        r = client.post("/collections/nope/query", json={"vector": [1.0]})
        assert r.status_code == 404


class TestTextQuery:
    def test_returns_bm25_ranked_results(self, client, docs):
        body = client.post(
            "/collections/docs/query/text", json={"text": "quick brown fox", "k": 3}
        ).json()
        assert body["results"]
        scores = [h["score"] for h in body["results"]]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0 for s in scores)

    def test_no_match_returns_an_empty_list(self, client, docs):
        body = client.post(
            "/collections/docs/query/text", json={"text": "zzzznotpresent"}
        ).json()
        assert body["results"] == []

    def test_filter_is_applied(self, client, docs):
        body = client.post(
            "/collections/docs/query/text",
            json={"text": "the", "k": 10, "filter": {"category": "tech"}},
        ).json()
        assert all(h["metadata"]["category"] == "tech" for h in body["results"])

    def test_collection_without_a_text_field_is_rejected(self, client):
        client.post("/collections", json={"name": "novec", "dimension": DIM})
        client.post(
            "/collections/novec/insert",
            json={"items": [{"id": "a", "vector": [1.0] * DIM}]},
        )
        r = client.post("/collections/novec/query/text", json={"text": "x"})
        assert r.status_code == 400
        assert r.json()["code"] == "NO_TEXT_FIELD"

    def test_empty_text_is_rejected(self, client, docs):
        assert client.post(
            "/collections/docs/query/text", json={"text": ""}
        ).status_code == 400


class TestHybridQuery:
    def test_returns_the_documented_shape_with_ranks(self, client, docs):
        r = client.post(
            "/collections/docs/query/hybrid",
            json={"vector": docs[0]["vector"], "text": "quick brown fox", "k": 5},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 5
        for hit in body["results"]:
            assert set(hit) == {
                "id", "rrf_score", "dense_rank", "sparse_rank", "metadata"
            }
            assert "score" not in hit, "RRF returns no comparable similarity score"
        assert any(h["dense_rank"] for h in body["results"])
        assert any(h["sparse_rank"] for h in body["results"])

    def test_rrf_scores_descend(self, client, docs):
        body = client.post(
            "/collections/docs/query/hybrid",
            json={"vector": docs[0]["vector"], "text": "the quick dog", "k": 8},
        ).json()
        scores = [h["rrf_score"] for h in body["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_missing_from_one_retriever_gives_a_null_rank(self, client, docs):
        body = client.post(
            "/collections/docs/query/hybrid",
            json={"vector": docs[0]["vector"], "text": "consectetur", "k": 10},
        ).json()
        assert any(h["sparse_rank"] is None for h in body["results"])

    def test_all_documented_params_are_accepted(self, client, docs):
        r = client.post(
            "/collections/docs/query/hybrid",
            json={
                "vector": docs[0]["vector"],
                "text": "quick fox",
                "k": 5,
                "filter": {"category": "animals"},
                "params": {
                    "ef_search": 64,
                    "dense_candidates": 50,
                    "sparse_candidates": 50,
                    "rrf_k": 60,
                },
            },
        )
        assert r.status_code == 200
        assert all(
            h["metadata"]["category"] == "animals" for h in r.json()["results"]
        )

    def test_collection_without_a_text_field_is_rejected(self, client):
        client.post("/collections", json={"name": "novec", "dimension": DIM})
        r = client.post(
            "/collections/novec/query/hybrid",
            json={"vector": [1.0] * DIM, "text": "x"},
        )
        assert r.status_code == 400
        assert r.json()["code"] == "NO_TEXT_FIELD"


class TestVectorEndpoints:
    def test_get_returns_the_vector_and_metadata(self, client, docs):
        body = client.get("/collections/docs/vectors/d3").json()
        assert body["id"] == "d3"
        assert len(body["vector"]) == DIM
        assert body["metadata"]["n"] == 3

    def test_get_unknown_id_is_404(self, client, docs):
        r = client.get("/collections/docs/vectors/nope")
        assert r.status_code == 404
        assert r.json()["code"] == "ID_NOT_FOUND"

    def test_delete_returns_204(self, client, docs):
        assert client.delete("/collections/docs/vectors/d3").status_code == 204

    def test_deleted_id_reads_as_404(self, client, docs):
        client.delete("/collections/docs/vectors/d3")
        r = client.get("/collections/docs/vectors/d3")
        assert r.status_code == 404
        assert r.json()["code"] == "ID_NOT_FOUND"

    def test_deleted_id_disappears_from_results(self, client, docs):
        client.delete("/collections/docs/vectors/d3")
        body = client.post(
            "/collections/docs/query", json={"vector": docs[3]["vector"], "k": 16}
        ).json()
        assert "d3" not in {h["id"] for h in body["results"]}

    def test_delete_unknown_id_is_404(self, client, docs):
        assert client.delete("/collections/docs/vectors/nope").status_code == 404

    def test_delete_twice_is_404(self, client, docs):
        client.delete("/collections/docs/vectors/d3")
        assert client.delete("/collections/docs/vectors/d3").status_code == 404


class TestMaintenance:
    def test_optimize_returns_202_and_compacts(self, client, docs):
        client.delete("/collections/docs/vectors/d1")
        client.delete("/collections/docs/vectors/d2")
        r = client.post("/collections/docs/optimize")
        assert r.status_code == 202
        body = r.json()
        assert body["compacted"] == 2
        assert body["num_vectors"] == 14
        assert body["job_id"].startswith("opt-")
        assert client.get("/collections/docs").json()["num_deleted"] == 0

    def test_data_is_still_queryable_after_optimize(self, client, docs):
        client.delete("/collections/docs/vectors/d1")
        client.post("/collections/docs/optimize")
        body = client.post(
            "/collections/docs/query", json={"vector": docs[5]["vector"], "k": 1}
        ).json()
        assert body["results"][0]["id"] == "d5"

    def test_snapshot_returns_201_with_an_id_and_path(self, client, docs):
        r = client.post("/collections/docs/snapshot")
        assert r.status_code == 201
        body = r.json()
        assert body["snapshot_id"]
        assert "snapshots" in body["path"]

    def test_maintenance_on_unknown_collection_is_404(self, client):
        assert client.post("/collections/nope/optimize").status_code == 404
        assert client.post("/collections/nope/snapshot").status_code == 404


class TestHealthAndMeta:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["uptime_s"] >= 0
        assert body["version"]

    def test_metrics_is_prometheus_text(self, client, docs):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        text = r.text
        assert "# TYPE pyvec_vectors gauge" in text
        assert 'pyvec_vectors{collection="docs"} 16' in text
        assert 'op="inserts"' in text

    def test_metrics_tracks_queries(self, client, docs):
        client.post("/collections/docs/query", json={"vector": docs[0]["vector"]})
        assert 'op="queries_dense"} 1' in client.get("/metrics").text

    def test_root_lists_the_endpoints(self, client):
        body = client.get("/").json()
        assert body["name"] == "pyvec"
        assert any("query/hybrid" in e for e in body["endpoints"])

    def test_openapi_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestPersistenceAcrossRestart:
    def test_data_survives_an_app_restart(self, data_root):
        """PRD UC4 through the HTTP surface: restart the server, data is there."""
        app = create_app(manager=CollectionManager(data_root))
        items = [{"id": f"d{i}", "vector": [float(i), 1.0] + [0.0] * (DIM - 2),
                  "metadata": {"content": f"doc {i}"}} for i in range(10)]
        with TestClient(app) as c:
            c.post(
                "/collections",
                json={"name": "docs", "dimension": DIM, "index": {"type": "hnsw"},
                      "text_field": "content"},
            )
            c.post("/collections/docs/insert", json={"items": items})
            before = c.post(
                "/collections/docs/query", json={"vector": items[0]["vector"], "k": 5}
            ).json()["results"]

        # A brand new app over the same directory: this is a process restart.
        restarted = create_app(manager=CollectionManager(data_root))
        with TestClient(restarted) as c:
            assert c.get("/collections/docs").json()["num_vectors"] == 10
            after = c.post(
                "/collections/docs/query", json={"vector": items[0]["vector"], "k": 5}
            ).json()["results"]
            assert [h["id"] for h in after] == [h["id"] for h in before]
            assert c.post(
                "/collections/docs/query/text", json={"text": "doc 3"}
            ).json()["results"]
