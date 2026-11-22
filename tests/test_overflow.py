"""Overflow handling: float64 recompute, recovery, and per-request failure."""

import asyncio
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main_module  # noqa: E402
from app.adapters import (  # noqa: E402
    D_IN,
    D_OUT,
    FLOAT32_MAX,
    PRESET_BY_KEY,
    PRESETS,
    OutputOutOfRangeError,
    apply_linear,
    load_arrays,
)
from app.scheduler import Scheduler  # noqa: E402

SUPPORT = PRESET_BY_KEY["support"]


def vec(seed=0):
    return np.random.default_rng(seed).standard_normal(D_IN, dtype=np.float32)


def reference_float64(spec, weights, vector):
    hidden = vector.astype(np.float64) @ weights["a"].astype(np.float64).T
    return hidden @ weights["b"].astype(np.float64).T * spec.scale + weights[
        "bias"
    ].astype(np.float64)


async def wait_for(predicate, timeout=4.0, interval=0.005):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    raise AssertionError("condition not reached before timeout")


# ------------------------------------------------------------------ numerics


def test_finite_results_keep_the_original_float32_output():
    weights = load_arrays(SUPPORT)
    v = vec(5)
    out = apply_linear(weights, SUPPORT, v)
    expected = (v @ weights["a"].T) @ weights["b"].T * SUPPORT.scale + weights["bias"]
    assert out.dtype == np.float32
    # Bit-identical to the original pipeline, not merely close.
    np.testing.assert_array_equal(out, expected.astype(np.float32, copy=False))


def test_float32_overflow_recovers_via_float64_recompute():
    weights = load_arrays(SUPPORT)
    v = np.full(D_IN, 1e37, dtype=np.float32)
    out = apply_linear(weights, SUPPORT, v)
    assert out.dtype == np.float32
    assert out.shape == (D_OUT,)
    assert np.all(np.isfinite(out))
    ref = reference_float64(SUPPORT, weights, v)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-30)


def test_float64_recompute_shares_resident_weights():
    # The recovery path must not copy the weight matrices to float64.
    spec = PRESET_BY_KEY["legal"]  # legal + 1e37 overflows float32, then recovers
    weights = load_arrays(spec)
    v = np.full(D_IN, 1e37, dtype=np.float32)
    tracemalloc.start()
    try:
        apply_linear(weights, spec, v)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    weights_f64_bytes = (spec.rank * D_IN + D_OUT * spec.rank + D_OUT) * 8
    # Peak temporaries stay far below a float64 copy of the resident weights.
    assert peak < weights_f64_bytes // 4


def test_out_of_range_result_raises_with_reason():
    weights = load_arrays(SUPPORT)
    v = np.full(D_IN, 1e38, dtype=np.float32)
    with pytest.raises(OutputOutOfRangeError, match="float32"):
        apply_linear(weights, SUPPORT, v)


def test_float64_recompute_never_produces_non_finite_for_float32_inputs():
    # Extreme but finite float32 inputs stay finite in float64.
    weights = load_arrays(SUPPORT)
    v = np.full(D_IN, FLOAT32_MAX, dtype=np.float32)
    with pytest.raises(OutputOutOfRangeError):
        apply_linear(weights, SUPPORT, v)


# ----------------------------------------------------------------- scheduler


@pytest.mark.asyncio
async def test_failed_overflow_releases_slot_and_weights_others_dispatch():
    sch = Scheduler(budget_bytes=30_000, specs=PRESETS)
    ok = sch.submit(
        SUPPORT, np.full(D_IN, 1e37, dtype=np.float32), compute_ms=10, load_delay=0.01
    )
    bad = sch.submit(SUPPORT, np.full(D_IN, 1e38, dtype=np.float32), compute_ms=10)
    after = sch.submit(SUPPORT, vec(3), compute_ms=10)
    await wait_for(
        lambda: all(r.state in ("done", "error") for r in (ok, bad, after))
    )

    # Recoverable overflow: full finite result.
    assert ok.state == "done"
    assert ok.result is not None and np.all(np.isfinite(ok.result))

    # Out of range: only this request fails, result left empty.
    assert bad.state == "error"
    assert bad.result is None
    assert "计算失败" in bad.error and "float32" in bad.error

    # Compute slot and weight reference released; later requests dispatch.
    assert after.state == "done"
    assert sch.running == []
    assert sch.resident["support"].refcount == 0
    assert sch.used_bytes == SUPPORT.nbytes  # idle weights stay cached
    kinds = [e["kind"] for e in sch.events]
    assert "compute_failed" in kinds and "compute_done" in kinds


@pytest.mark.asyncio
async def test_overflow_failure_does_not_disturb_a_running_compute():
    sch = Scheduler(budget_bytes=30_000, specs=PRESETS)
    slow = sch.submit(SUPPORT, vec(1), compute_ms=600, load_delay=0.01)
    await wait_for(lambda: slow.state == "running")
    bad = sch.submit(SUPPORT, np.full(D_IN, 1e38, dtype=np.float32), compute_ms=10)
    await wait_for(lambda: bad.state == "error")
    assert slow.state == "running"  # untouched by the sibling failure
    await wait_for(lambda: slow.state == "done")
    assert slow.result is not None
    assert sch.resident["support"].refcount == 0


# ----------------------------------------------------------------------- API


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
    with TestClient(main_module.app) as test_client:
        yield test_client


def submit_and_finish(client, value):
    res = client.post(
        "/api/requests",
        json={"adapter": "support", "compute_ms": 10, "vector": [value] * D_IN},
    )
    assert res.status_code == 201, res.text
    request_id = res.json()["id"]
    return wait_until(
        lambda: (lambda d: d if d["state"] in ("done", "error") else None)(
            client.get(f"/api/requests/{request_id}").json()
        )
    )


def test_api_recovers_overflow_with_complete_finite_result(client):
    detail = submit_and_finish(client, 1e37)
    assert detail["state"] == "done", detail
    assert detail["error"] is None
    result = detail["result"]
    assert len(result) == D_OUT
    assert all(x is not None for x in result)  # no null holes like before
    assert np.all(np.isfinite(result))
    # Same values as the float64 reference cast back to float32.
    weights = load_arrays(SUPPORT)
    ref = reference_float64(SUPPORT, weights, np.full(D_IN, 1e37, np.float32))
    np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-30)


def test_api_marks_out_of_range_request_failed_with_empty_result(client):
    detail = submit_and_finish(client, 1e38)
    assert detail["state"] == "error", detail
    assert detail["result"] is None
    assert "计算失败" in detail["error"]
    assert "float32" in detail["error"]
    # The failure is visible in the finished list too.
    state = client.get("/api/state").json()
    entry = next(r for r in state["finished"] if r["id"] == detail["id"])
    assert entry["state"] == "error"
    assert entry["result_preview"] is None
    # Slot and budget released for the next requests.
    assert state["running"] == []
    assert state["used_bytes"] == SUPPORT.nbytes
