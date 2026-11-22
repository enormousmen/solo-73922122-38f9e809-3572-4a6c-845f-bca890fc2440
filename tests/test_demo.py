import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters import PRESETS  # noqa: E402
from app.demo import DemoRunner  # noqa: E402
from app.scheduler import Scheduler  # noqa: E402

BUDGET = 30_000


async def wait_for(predicate, timeout=30.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("demo did not reach expected state")


async def test_demo_shows_shared_weights_head_wait_and_single_bypass():
    scheduler = Scheduler(budget_bytes=BUDGET, specs=PRESETS)
    runner = DemoRunner(scheduler)

    phases = []
    started = await runner.start()
    assert started is True
    assert await runner.start() is False  # already running
    try:
        while runner.phase not in ("finished", "error"):
            if runner.phase not in phases:
                phases.append(runner.phase)
            await asyncio.sleep(0.02)
        assert runner.phase == "finished", runner.error
    finally:
        await runner.cancel()

    kinds = [e["kind"] for e in scheduler.events]
    assert "waiting_memory" in kinds
    assert kinds.count("head_bypassed") == 1
    assert "bypass_dispatched" in kinds
    assert "head_holds_line" in kinds
    assert kinds.count("head_bypassed") == kinds.count("bypass_dispatched")

    # The scripted demo left legal + support resident, finished everything.
    assert scheduler.queue == []
    assert scheduler.running == []
    assert scheduler.used_bytes == sum(r.nbytes for r in scheduler.resident.values())
    assert scheduler.used_bytes <= BUDGET
    done = [r for r in scheduler.finished if r.state == "done"]
    assert len(done) == 6
    assert all(r.result is not None for r in done)

    # Ordering: the overtaker started before the head; the head before the
    # second later request.
    steps = {s["request_id"]: s for s in runner.steps}
    assert len(steps) == 6
    briefs = [
        s["request_id"] for s in runner.steps
        if scheduler.requests[s["request_id"]].spec.key == "brief"
    ]
    assert len(briefs) == 2  # head + follower sharing its weights
    head = scheduler.requests[briefs[0]]
    head_pos = next(i for i, s in enumerate(runner.steps) if s["request_id"] == head.id)
    overtaker = scheduler.requests[runner.steps[head_pos + 1]["request_id"]]
    assert overtaker.spec.key == "legal"
    later = scheduler.requests[runner.steps[head_pos + 2]["request_id"]]
    assert later.spec.key == "brief"

    assert overtaker.started_at < head.started_at
    # The follower never overtakes: dispatched only once the head has started,
    # then shares the head's resident weights.
    assert head.started_at <= later.started_at
    assert later.load_attempts == 0
    assert scheduler.requests[briefs[1]] is later
    assert later.started_at - head.started_at < 1.0

    assert "head_waiting" in phases or "bypassing" in phases
