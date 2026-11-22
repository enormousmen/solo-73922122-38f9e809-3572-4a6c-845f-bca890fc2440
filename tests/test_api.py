import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main_module  # noqa: E402
from app.adapters import D_IN, PRESETS  # noqa: E402
from app.scheduler import Scheduler  # noqa: E402

LEGAL = next(s for s in PRESETS if s.key == "legal")


def wait_until(fn, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached before timeout")


@pytest.fixture()
def client():
    main_module.scheduler = Scheduler(budget_bytes=30_000, specs=PRESETS)
    main_module.demo_runner.scheduler = main_module.scheduler
    # A context-managed client keeps one event loop alive for background tasks.
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_index_serves_the_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "LoRA" in res.text
    assert client.get("/static/app.js").status_code == 200


def test_presets_expose_arrays_sizes_and_budget(client):
    data = client.get("/api/presets").json()
    keys = {p["key"] for p in data["presets"]}
    assert keys == {s.key for s in PRESETS}
    assert data["input_dim"] == D_IN
    assert data["budget_bytes"] == 30_000
    assert all(p["bytes"] > 0 for p in data["presets"])
    assert LEGAL.nbytes == next(p["bytes"] for p in data["presets"] if p["key"] == "legal")


def test_submit_with_sample_vector_and_fetch_result(client):
    res = client.post("/api/requests", json={"adapter": "legal", "compute_ms": 20})
    assert res.status_code == 201
    request_id = res.json()["id"]
    detail = client.get(f"/api/requests/{request_id}").json()
    assert detail["adapter"] == "legal"
    assert detail["state"] in ("queued", "loading", "ready", "running")
    detail = wait_until(
        lambda: (lambda d: d if d["state"] in ("done", "error") else None)(
            client.get(f"/api/requests/{request_id}").json()
        )
    )
    assert detail["state"] == "done", detail
    assert len(detail["result"]) == 96
    assert np.all(np.isfinite(detail["result"]))


def test_submit_with_custom_vector(client):
    vector = [0.1] * D_IN
    res = client.post("/api/requests", json={"adapter": "support", "vector": vector})
    assert res.status_code == 201
    request_id = res.json()["id"]
    detail = wait_until(
        lambda: (lambda d: d if d["state"] in ("done", "error") else None)(
            client.get(f"/api/requests/{request_id}").json()
        )
    )
    assert detail["state"] == "done"
    # determinism: same vector + adapter yields the same output
    again = client.post("/api/requests", json={"adapter": "support", "vector": vector})
    other = again.json()["id"]
    d2 = wait_until(
        lambda: (lambda d: d if d["state"] in ("done", "error") else None)(
            client.get(f"/api/requests/{other}").json()
        )
    )
    assert d2["result"] == detail["result"]


def test_invalid_inputs_are_rejected(client):
    assert client.post("/api/requests", json={"adapter": "nope"}).status_code == 422
    short = client.post("/api/requests", json={"adapter": "legal", "vector": [0.0]})
    assert short.status_code == 422
    nan_body = '{"adapter": "legal", "vector": [' + ", ".join(["NaN"] * D_IN) + "]}"
    bad = client.post(
        "/api/requests", content=nan_body, headers={"content-type": "application/json"}
    )
    assert bad.status_code == 422
    assert client.get("/api/requests/doesnotexist").status_code == 404


def test_state_snapshot_tracks_budget_and_residents(client):
    client.post("/api/requests", json={"adapter": "legal", "compute_ms": 10})
    state = client.get("/api/state").json()
    assert state["budget_bytes"] == 30_000
    assert state["used_bytes"] == LEGAL.nbytes
    assert {r["key"] for r in state["resident"]} == {"legal"}
    assert state["free_bytes"] == 30_000 - LEGAL.nbytes
    assert "events" in state and state["demo"]["phase"] == "idle"


def test_load_failure_is_reported_without_killing_other_requests(client):
    bad = client.post(
        "/api/requests",
        json={"adapter": "brief", "simulate_load_failure": True, "compute_ms": 10},
    ).json()["id"]
    good = client.post(
        "/api/requests", json={"adapter": "legal", "compute_ms": 400}
    ).json()["id"]
    final = wait_until(
        lambda: (lambda b, g: {"bad": b, "good": g}
                 if b["state"] == "error" and g["state"] == "done" else None)(
            client.get(f"/api/requests/{bad}").json(),
            client.get(f"/api/requests/{good}").json(),
        )
    )
    assert final["bad"]["state"] == "error"
    assert "加载失败" in final["bad"]["error"]
    assert final["good"]["state"] == "done"
    assert final["good"]["result"] is not None
    state = client.get("/api/state").json()
    assert state["used_bytes"] == LEGAL.nbytes


def test_demo_refuses_to_start_while_requests_are_active(client):
    client.post("/api/requests", json={"adapter": "legal", "compute_ms": 1500})
    res = client.post("/api/demo")
    assert res.status_code == 409
    assert "处理请求" in res.json()["detail"]


def test_demo_endpoint_runs_and_reset_clears_state(client):
    res = client.post("/api/demo")
    assert res.status_code == 200
    assert res.json()["status"] == "running"
    # wait for completion
    state = wait_until(
        lambda: (lambda s: s if s["demo"]["phase"] in ("finished", "error") else None)(
            client.get("/api/state").json()
        )
    )
    assert state["demo"]["phase"] == "finished", state["demo"]
    kinds = [e["kind"] for e in state["events"]]
    assert "head_bypassed" in kinds and "head_holds_line" in kinds

    reset = client.post("/api/reset")
    assert reset.status_code == 200
    state = client.get("/api/state").json()
    assert state["used_bytes"] == 0
    assert state["queue"] == [] and state["running"] == [] and state["resident"] == []
    assert state["demo"]["phase"] == "idle"
