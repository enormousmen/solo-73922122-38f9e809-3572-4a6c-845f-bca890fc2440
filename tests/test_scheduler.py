import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters import D_IN, PRESET_BY_KEY  # noqa: E402
from app.scheduler import Scheduler  # noqa: E402

BUDGET = 30_000
LEGAL = PRESET_BY_KEY["legal"].nbytes
SUPPORT = PRESET_BY_KEY["support"].nbytes
BRIEF = PRESET_BY_KEY["brief"].nbytes


def vec(seed=0):
    return np.random.default_rng(seed).standard_normal(D_IN, dtype=np.float32)


def make_scheduler(**kw):
    return Scheduler(budget_bytes=kw.pop("budget_bytes", BUDGET), **kw)


async def wait_for(predicate, timeout=4.0, interval=0.005):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    raise AssertionError("condition not reached before timeout")


def events(sch, kind):
    return [e for e in sch.events if e["kind"] == kind]


def test_budget_is_smaller_than_two_large_adapters():
    # The premise of the exercise: two 32-rank adapters cannot coexist.
    assert 2 * BRIEF > BUDGET
    assert LEGAL + SUPPORT <= BUDGET


@pytest.mark.asyncio
async def test_same_adapter_shares_one_copy():
    sch = make_scheduler()
    a = sch.submit(PRESET_BY_KEY["legal"], vec(1), compute_ms=400)
    b = sch.submit(PRESET_BY_KEY["legal"], vec(2), compute_ms=400)
    await wait_for(lambda: a.state == b.state == "running")
    assert len(sch.resident) == 1
    assert sch.used_bytes == LEGAL
    assert sch.resident["legal"].refcount == 2
    assert b.load_attempts == 0  # b never loads; it joins the resident copy
    await wait_for(lambda: a.state == b.state == "done")


@pytest.mark.asyncio
async def test_memory_charged_while_loading_and_freed_after_last_user():
    sch = make_scheduler()
    req = sch.submit(PRESET_BY_KEY["legal"], vec(), compute_ms=100, load_delay=0.1)
    assert sch.used_bytes == LEGAL  # charged at load start, not at completion
    assert sch.resident["legal"].loading is True
    await wait_for(lambda: req.state == "done")
    assert sch.resident["legal"].refcount == 0
    assert sch.used_bytes == LEGAL  # idle weights stay cached
    # A new request for the same adapter reuses the idle copy without a load.
    again = sch.submit(PRESET_BY_KEY["legal"], vec(), compute_ms=100)
    await wait_for(lambda: again.state == "done")
    assert again.load_attempts == 0


@pytest.mark.asyncio
async def test_failed_load_errors_only_that_request_and_returns_budget():
    sch = make_scheduler()
    bad = sch.submit(
        PRESET_BY_KEY["brief"], vec(), compute_ms=100, simulate_load_failure=True
    )
    ok = sch.submit(PRESET_BY_KEY["legal"], vec(), compute_ms=300)
    await wait_for(lambda: bad.state == "error")
    assert "加载失败" in bad.error
    assert "brief" not in sch.resident  # failed reservation released at once
    assert bad not in sch.queue
    await wait_for(lambda: ok.state == "done")
    assert ok.result is not None
    assert sch.used_bytes == LEGAL  # only the successful legal copy remains


@pytest.mark.asyncio
async def test_failed_load_does_not_disturb_running_compute():
    sch = make_scheduler()
    running = sch.submit(PRESET_BY_KEY["legal"], vec(), compute_ms=800)
    await wait_for(lambda: running.state == "running")
    # support fits next to the running legal copy, so this load really starts.
    bad = sch.submit(
        PRESET_BY_KEY["support"],
        vec(),
        compute_ms=100,
        simulate_load_failure=True,
        load_delay=0.05,
    )
    await wait_for(lambda: bad.state == "error")
    assert running.state == "running"
    await wait_for(lambda: running.state == "done")
    assert running.result is not None
    assert sch.used_bytes == LEGAL


@pytest.mark.asyncio
async def test_head_waits_for_memory_then_bypass_lets_one_later_request_through():
    sch = make_scheduler()
    a = sch.submit(PRESET_BY_KEY["legal"], vec(1), compute_ms=700)
    b = sch.submit(PRESET_BY_KEY["support"], vec(2), compute_ms=700)
    await wait_for(lambda: a.state == b.state == "running")
    assert sch.running == [a, b]

    head = sch.submit(PRESET_BY_KEY["brief"], vec(3), compute_ms=300)
    await asyncio.sleep(0)
    assert head.state == "queued"
    assert head.blocked_on_memory is True
    assert sch.queue[0] is head
    assert events(sch, "waiting_memory")

    overtaker = sch.submit(PRESET_BY_KEY["legal"], vec(4), compute_ms=150)
    await wait_for(lambda: overtaker.state == "running", timeout=1)
    assert events(sch, "head_bypassed")
    assert overtaker.started_at < (head.started_at or float("inf"))
    await wait_for(lambda: overtaker.state == "done")
    # The head keeps its place in the queue while the bypass runs.
    assert sch.queue[0] is head
    assert head.state == "queued"


@pytest.mark.asyncio
async def test_head_is_bypassed_at_most_once():
    sch = make_scheduler()
    a = sch.submit(PRESET_BY_KEY["legal"], vec(1), compute_ms=1200)
    b = sch.submit(PRESET_BY_KEY["support"], vec(2), compute_ms=1200)
    await wait_for(lambda: a.state == b.state == "running")

    head = sch.submit(PRESET_BY_KEY["brief"], vec(3), compute_ms=100)
    await asyncio.sleep(0)
    first = sch.submit(PRESET_BY_KEY["legal"], vec(4), compute_ms=200)
    second = sch.submit(PRESET_BY_KEY["legal"], vec(5), compute_ms=200)
    await wait_for(lambda: first.state == "running", timeout=2)
    await asyncio.sleep(0.03)
    # The second later request must not run while the head still waits.
    assert second.state == "queued"
    assert sch.queue.index(head) == 0
    assert len(events(sch, "head_bypassed")) == 1

    await wait_for(lambda: first.state == "done", timeout=2)
    await asyncio.sleep(0.02)
    assert events(sch, "head_holds_line")  # head refuses a second bypass
    assert second.state == "queued"

    await wait_for(lambda: head.state in ("running", "done"), timeout=4)
    assert head.started_at is not None
    await wait_for(lambda: second.state in ("running", "done"), timeout=4)
    assert second.started_at >= head.started_at


@pytest.mark.asyncio
async def test_head_wins_when_idle_cache_can_be_evicted():
    sch = make_scheduler()
    a = sch.submit(PRESET_BY_KEY["legal"], vec(), compute_ms=50)
    b = sch.submit(PRESET_BY_KEY["support"], vec(), compute_ms=50)
    await wait_for(lambda: a.state == b.state == "done")
    # Both weights are idle; a brief request needs both slots and must evict
    # rather than bypass.
    head = sch.submit(PRESET_BY_KEY["brief"], vec(), compute_ms=100)
    assert head.state == "loading"  # no bypass: memory could be reclaimed
    assert sch.used_bytes == BRIEF
    assert not events(sch, "head_bypassed")
    assert events(sch, "evicted")
    await wait_for(lambda: head.state == "done")


@pytest.mark.asyncio
async def test_compute_slots_are_limited_to_two():
    sch = make_scheduler()
    reqs = [
        sch.submit(PRESET_BY_KEY["support"], vec(i), compute_ms=300) for i in range(4)
    ]
    await wait_for(lambda: len(sch.running) == 2)
    await asyncio.sleep(0.05)
    assert len(sch.running) == 2
    assert sum(1 for r in reqs if r.state == "queued") == 2
    await wait_for(lambda: all(r.state == "done" for r in reqs), timeout=5)


@pytest.mark.asyncio
async def test_follower_waits_for_head_to_start_then_shares_weights():
    sch = make_scheduler()
    a = sch.submit(PRESET_BY_KEY["legal"], vec(1), compute_ms=700)
    b = sch.submit(PRESET_BY_KEY["support"], vec(2), compute_ms=700)
    await wait_for(lambda: a.state == b.state == "running")

    head = sch.submit(PRESET_BY_KEY["brief"], vec(3), compute_ms=200)
    await asyncio.sleep(0)
    overtaker = sch.submit(PRESET_BY_KEY["legal"], vec(4), compute_ms=1200)
    follower = sch.submit(PRESET_BY_KEY["brief"], vec(5), compute_ms=200)
    await wait_for(lambda: overtaker.state == "running", timeout=1)
    await wait_for(lambda: overtaker.state == "done", timeout=3)
    # Still waiting: the same head must not be bypassed twice.
    assert follower.state == "queued"
    assert follower.load_attempts == 0
    assert sch.queue[:2] == [head, follower]

    await wait_for(lambda: head.state == "running", timeout=3)
    # Follower is queued behind the head; it dispatches once the head starts.
    await wait_for(lambda: follower.state == "running", timeout=1.5)
    assert follower.started_at >= head.started_at
    assert follower.load_attempts == 0  # joined the head's resident copy
    assert sch.resident["brief"].refcount == 2
    await wait_for(lambda: follower.state == "done")


@pytest.mark.asyncio
async def test_follower_survives_head_load_failure_by_retrying():
    sch = make_scheduler()
    head = sch.submit(
        PRESET_BY_KEY["brief"], vec(1), compute_ms=50, simulate_load_failure=True
    )
    follower = sch.submit(PRESET_BY_KEY["brief"], vec(2), compute_ms=50)
    await wait_for(lambda: head.state == "error" and follower.state in ("done", "error"))
    assert head.state == "error"
    assert follower.state == "done"  # it became head and retried the load
    assert follower.load_attempts == 1
    assert sch.used_bytes == BRIEF
    await wait_for(lambda: follower.state == "done")


@pytest.mark.asyncio
async def test_shared_request_does_not_trigger_load_failure_flag():
    # A request that joins an already-loaded adapter never loads, so its
    # failure flag is irrelevant and it must still succeed.
    sch = make_scheduler()
    first = sch.submit(PRESET_BY_KEY["legal"], vec(1), compute_ms=300, load_delay=0.05)
    await wait_for(lambda: first.state == "running")
    joined = sch.submit(
        PRESET_BY_KEY["legal"], vec(2), compute_ms=50, simulate_load_failure=True
    )
    await wait_for(lambda: joined.state == "done")
    assert joined.result is not None
    assert joined.load_attempts == 0


@pytest.mark.asyncio
async def test_used_bytes_returns_to_zero_after_reset():
    sch = make_scheduler()
    sch.submit(PRESET_BY_KEY["legal"], vec(), compute_ms=500, load_delay=0.05)
    await asyncio.sleep(0.02)
    sch.reset()
    assert sch.used_bytes == 0
    assert sch.resident == {}
    assert sch.queue == []
    assert sch.running == []


@pytest.mark.asyncio
async def test_numerics_match_reference_linear_layer():
    sch = make_scheduler()
    v = vec(9)
    req = sch.submit(PRESET_BY_KEY["legal"], v, compute_ms=10, load_delay=0.01)
    await wait_for(lambda: req.state == "done")
    spec = PRESET_BY_KEY["legal"]
    from app.adapters import load_arrays

    w = load_arrays(spec)
    expected = (v @ w["a"].T) @ w["b"].T * spec.scale + w["bias"]
    np.testing.assert_allclose(req.result, expected, rtol=1e-5, atol=1e-5)
    assert req.result.shape == (96,)
