"""CLI and client SDK.

The client talks HTTP, so these tests stand up a real server on a real socket in
a background thread rather than using an in-process test client. That is the only
way to exercise the ``urllib`` transport, the error-payload parsing, and the
argument plumbing the way a user would hit them.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from pyvec.api.server import create_app
from pyvec.cli import build_parser, main
from pyvec.client import PyVecClient, PyVecHTTPError
from pyvec.core.collection_manager import CollectionManager

DIM = 8


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A real uvicorn server in a thread, shared by this module's tests."""
    root = tmp_path_factory.mktemp("cli_data")
    port = _free_port()
    app = create_app(manager=CollectionManager(root))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.02)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=15)


@pytest.fixture
def client(server):
    return PyVecClient(server)


def _items(n, seed=0, prefix="d"):
    rng = np.random.default_rng(seed)
    return [
        {
            "id": f"{prefix}{i}",
            "vector": rng.normal(size=DIM).astype(np.float32).tolist(),
            "metadata": {"content": f"document number {i} about vectors",
                         "n": i, "category": "a" if i % 2 else "b"},
        }
        for i in range(n)
    ]


class TestClientSDK:
    def test_the_documented_usage_sketch_works(self, client):
        """Exactly the flow from API_SPEC.md's "Client SDK sketch"."""
        client.create_collection(
            "sdk", dimension=DIM, metric="cosine", index="hnsw", text_field="content"
        )
        items = _items(20)
        assert client.insert("sdk", items)["inserted"] == 20
        results = client.hybrid("sdk", vector=items[0]["vector"], text="vectors", k=10)
        assert results
        assert "rrf_score" in results[0]

    def test_round_trip_of_every_operation(self, client):
        client.create_collection("rt", dimension=DIM, text_field="content", index="flat")
        items = _items(10, seed=1)
        client.insert("rt", items)

        assert client.describe("rt")["num_vectors"] == 10
        assert any(c["name"] == "rt" for c in client.list_collections())

        got = client.get("rt", "d3")
        assert got["id"] == "d3" and len(got["vector"]) == DIM

        dense = client.query("rt", items[3]["vector"], k=3)
        assert dense[0]["id"] == "d3"

        text = client.query_text("rt", "document number 3", k=3)
        assert text

        client.delete("rt", "d3")
        with pytest.raises(PyVecHTTPError) as exc:
            client.get("rt", "d3")
        assert exc.value.status == 404 and exc.value.code == "ID_NOT_FOUND"

        assert client.optimize("rt")["compacted"] == 1
        assert client.snapshot("rt")["snapshot_id"]
        assert client.health()["status"] == "ok"
        assert "pyvec_vectors" in client.metrics()

        client.drop_collection("rt")
        assert not any(c["name"] == "rt" for c in client.list_collections())

    def test_numpy_arrays_are_accepted_without_manual_conversion(self, client):
        client.create_collection("np", dimension=DIM, index="flat")
        vec = np.random.default_rng(0).normal(size=DIM).astype(np.float32)
        client.insert("np", [{"id": "a", "vector": vec}])
        assert client.query("np", vec, k=1)[0]["id"] == "a"
        client.drop_collection("np")

    def test_large_inserts_are_chunked_past_the_server_limit(self, client):
        """The server caps a batch at 1000; the client should not make the caller
        care about that."""
        client.create_collection("chunk", dimension=DIM, index="flat")
        items = [{"id": f"c{i}", "vector": [float(i % 13)] * DIM} for i in range(2500)]
        assert client.insert("chunk", items)["inserted"] == 2500
        assert client.describe("chunk")["num_vectors"] == 2500
        client.drop_collection("chunk")

    def test_filters_and_params_are_passed_through(self, client):
        client.create_collection("fp", dimension=DIM, index="hnsw", text_field="content")
        client.insert("fp", _items(30, seed=2))
        hits = client.query(
            "fp", _items(1, seed=2)[0]["vector"], k=5,
            filter={"category": "a"}, params={"ef_search": 100},
        )
        assert hits and all(h["metadata"]["category"] == "a" for h in hits)
        client.drop_collection("fp")

    def test_server_errors_surface_the_documented_code(self, client):
        with pytest.raises(PyVecHTTPError) as exc:
            client.describe("does-not-exist")
        assert exc.value.status == 404
        assert exc.value.code == "COLLECTION_NOT_FOUND"
        assert "not found" in exc.value.message

    def test_dimension_error_surfaces(self, client):
        client.create_collection("dim", dimension=DIM, index="flat")
        with pytest.raises(PyVecHTTPError) as exc:
            client.insert("dim", [{"id": "a", "vector": [1.0, 2.0]}])
        assert exc.value.code == "INVALID_DIMENSION"
        client.drop_collection("dim")

    def test_unreachable_server_reports_a_connection_failure(self):
        broken = PyVecClient(f"http://127.0.0.1:{_free_port()}", timeout=2.0)
        with pytest.raises(PyVecHTTPError) as exc:
            broken.health()
        assert exc.value.status == 0
        assert exc.value.code == "CONNECTION_FAILED"


class TestCLIParser:
    def test_every_documented_subcommand_parses(self):
        parser = build_parser()
        for argv in (
            ["serve", "--port", "9999"],
            ["create", "docs", "--dimension", "384", "--metric", "cosine",
             "--index", "hnsw", "--text-field", "content"],
            ["insert", "docs", "--file", "x.jsonl"],
            ["query", "docs", "--vector", "1,2,3", "-k", "5"],
            ["search-text", "docs", "quick fox"],
            ["hybrid", "docs", "--vector", "1,2", "--text", "fox"],
            ["get", "docs", "d1"],
            ["delete", "docs", "d1"],
            ["ls"],
            ["describe", "docs"],
            ["rm", "docs", "--yes"],
            ["optimize", "docs"],
            ["snapshot", "docs"],
            ["health"],
            ["metrics"],
        ):
            assert parser.parse_args(argv).command == argv[0]

    def test_a_missing_subcommand_is_an_error(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_invalid_metric_choice_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["create", "d", "--dimension", "4", "--metric", "manhattan"]
            )


class TestCLICommands:
    def test_create_insert_query_lifecycle(self, server, tmp_path, capsys):
        url = ["--url", server]
        assert main(url + ["create", "cli", "--dimension", str(DIM),
                           "--index", "flat", "--text-field", "content"]) == 0

        records = tmp_path / "items.jsonl"
        records.write_text(
            "\n".join(json.dumps(i) for i in _items(12, seed=5)), encoding="utf-8"
        )
        assert main(url + ["insert", "cli", "--file", str(records)]) == 0
        assert "12" in capsys.readouterr().out

        vector = ",".join(str(v) for v in _items(12, seed=5)[0]["vector"])
        assert main(url + ["query", "cli", "--vector", vector, "-k", "3"]) == 0
        out = capsys.readouterr().out
        assert "d0" in out and "score" in out

        assert main(url + ["search-text", "cli", "document number 3"]) == 0
        assert "id" in capsys.readouterr().out

        assert main(url + ["hybrid", "cli", "--vector", vector, "--text", "vectors"]) == 0
        assert "rrf_score" in capsys.readouterr().out

        assert main(url + ["describe", "cli"]) == 0
        assert "num_vectors" in capsys.readouterr().out

        assert main(url + ["ls"]) == 0
        assert "cli" in capsys.readouterr().out

        assert main(url + ["rm", "cli", "--yes"]) == 0
        assert "dropped" in capsys.readouterr().out

    def test_json_array_input_is_accepted(self, server, tmp_path, capsys):
        url = ["--url", server]
        main(url + ["create", "jsonarr", "--dimension", str(DIM), "--index", "flat"])
        path = tmp_path / "items.json"
        path.write_text(json.dumps(_items(5, seed=6)), encoding="utf-8")
        assert main(url + ["insert", "jsonarr", "--file", str(path)]) == 0
        assert "5" in capsys.readouterr().out
        main(url + ["rm", "jsonarr", "--yes"])

    def test_vector_file_input(self, server, tmp_path, capsys):
        url = ["--url", server]
        main(url + ["create", "vf", "--dimension", str(DIM), "--index", "flat"])
        items = _items(5, seed=7)
        recs = tmp_path / "i.jsonl"
        recs.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
        main(url + ["insert", "vf", "--file", str(recs)])
        capsys.readouterr()

        qfile = tmp_path / "q.json"
        qfile.write_text(json.dumps(items[0]["vector"]), encoding="utf-8")
        assert main(url + ["query", "vf", "--vector-file", str(qfile), "-k", "2"]) == 0
        assert "d0" in capsys.readouterr().out
        main(url + ["rm", "vf", "--yes"])

    def test_json_output_mode_emits_parseable_json(self, server, capsys):
        url = ["--url", server]
        assert main(url + ["--json", "health"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "ok"

    def test_metrics_command(self, server, capsys):
        assert main(["--url", server, "metrics"]) == 0
        assert "pyvec_uptime_seconds" in capsys.readouterr().out

    def test_filter_and_params_json_arguments(self, server, tmp_path, capsys):
        url = ["--url", server]
        main(url + ["create", "flt", "--dimension", str(DIM), "--index", "hnsw"])
        items = _items(20, seed=8)
        recs = tmp_path / "i.jsonl"
        recs.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
        main(url + ["insert", "flt", "--file", str(recs)])
        capsys.readouterr()
        vector = ",".join(str(v) for v in items[0]["vector"])
        assert main(url + ["query", "flt", "--vector", vector,
                           "--filter", '{"category": "a"}',
                           "--params", '{"ef_search": 64}']) == 0
        assert capsys.readouterr().out
        main(url + ["rm", "flt", "--yes"])

    def test_index_params_on_create(self, server, capsys):
        url = ["--url", server]
        assert main(url + ["create", "ip", "--dimension", str(DIM),
                           "--index", "hnsw",
                           "--index-params", '{"M": 8, "ef_construction": 50}']) == 0
        capsys.readouterr()
        main(url + ["--json", "describe", "ip"])
        detail = json.loads(capsys.readouterr().out)
        assert detail["index"]["params"]["M"] == 8
        main(url + ["rm", "ip", "--yes"])

    def test_optimize_and_snapshot(self, server, tmp_path, capsys):
        url = ["--url", server]
        main(url + ["create", "maint", "--dimension", str(DIM), "--index", "flat"])
        items = _items(6, seed=9)
        recs = tmp_path / "i.jsonl"
        recs.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
        main(url + ["insert", "maint", "--file", str(recs)])
        main(url + ["delete", "maint", "d0"])
        capsys.readouterr()
        assert main(url + ["optimize", "maint"]) == 0
        assert "compacted" in capsys.readouterr().out
        assert main(url + ["snapshot", "maint"]) == 0
        assert "snapshot_id" in capsys.readouterr().out
        main(url + ["rm", "maint", "--yes"])


class TestCLIErrors:
    def test_unreachable_server_exits_nonzero_with_a_hint(self, capsys):
        port = _free_port()
        assert main(["--url", f"http://127.0.0.1:{port}", "health"]) == 1
        err = capsys.readouterr().err
        assert "cannot reach a PyVec server" in err
        assert "pyvec serve" in err

    def test_server_error_exits_nonzero(self, server, capsys):
        assert main(["--url", server, "describe", "no-such-collection"]) == 1
        assert "COLLECTION_NOT_FOUND" in capsys.readouterr().err

    def test_malformed_filter_json_is_reported(self, server):
        with pytest.raises(SystemExit, match="not valid JSON"):
            main(["--url", server, "query", "docs", "--vector", "1,2",
                  "--filter", "{not json"])

    def test_both_vector_flags_is_an_error(self, server):
        with pytest.raises(SystemExit, match="not both"):
            main(["--url", server, "query", "d", "--vector", "1,2",
                  "--vector-file", "x.json"])

    def test_neither_vector_flag_is_an_error(self, server):
        with pytest.raises(SystemExit, match="required"):
            main(["--url", server, "query", "d"])

    def test_non_numeric_vector_is_reported(self, server):
        with pytest.raises(SystemExit, match="comma-separated floats"):
            main(["--url", server, "query", "d", "--vector", "1,two,3"])

    def test_missing_input_file_is_reported(self, server):
        with pytest.raises((SystemExit, FileNotFoundError)):
            main(["--url", server, "insert", "d", "--file", "no-such-file.jsonl"])

    def test_empty_input_file_is_reported(self, server, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit, match="no records"):
            main(["--url", server, "insert", "d", "--file", str(empty)])
