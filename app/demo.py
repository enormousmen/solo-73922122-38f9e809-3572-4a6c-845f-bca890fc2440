"""Scripted demo of weight sharing, the head-memory wait, and the single bypass.

Timeline with the default 30 000 B budget:

* legal (21 888 B) and support (7 552 B) fit together; two legal requests
  share a single copy of the legal weights.
* brief (29 056 B) arrives while both residents are in use, so it cannot fit.
  It becomes the head and waits for memory.
* A later legal request is let through *once* to use the resident legal
  weights while the head keeps waiting.
* A further later request (for the same `brief` adapter) must not overtake the
  same head.  It stays queued until the head has started; then it is dispatched
  and shares the head's resident brief weights.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from .adapters import D_IN, PRESET_BY_KEY
from .scheduler import Scheduler

SAMPLE_SEED = 7


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(D_IN, dtype=np.float32)


class DemoRunner:
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.task: asyncio.Task | None = None
        self.phase = "idle"
        self.error: str | None = None
        self.steps: list[dict[str, Any]] = []

    def plan(self) -> list[dict[str, Any]]:
        return [
            {"at": 0.0, "action": "submit", "adapter": "legal", "note": "两份请求共用一份权重"},
            {"at": 0.0, "action": "submit", "adapter": "legal", "note": "同一适配器，共享驻留权重"},
            {"at": 0.7, "action": "submit", "adapter": "support", "note": "填满预算"},
            {"at": 3.6, "action": "submit", "adapter": "brief", "note": "新队首，内存不足，开始等待"},
            {"at": 3.8, "action": "submit", "adapter": "legal", "note": "后到请求复用驻留权重，越过队首一次"},
            {"at": 3.9, "action": "submit", "adapter": "brief", "note": "再次后到：必须等队首启动后才派发，之后共享队首权重"},
        ]

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def status(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "running": self.running,
            "error": self.error,
            "steps": self.steps,
        }

    async def start(self) -> bool:
        if self.running:
            return False
        self.steps = []
        self.error = None
        self.phase = "starting"
        self.task = asyncio.create_task(self._run())
        return True

    async def cancel(self) -> None:
        self.phase = "idle"
        task, self.task = self.task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _wait(self, predicate, *, timeout: float, interval: float = 0.02) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if predicate():
                return
            await asyncio.sleep(interval)
        raise TimeoutError("demo phase timed out waiting for scheduler state")

    def _record(self, action: str, adapter: str, request_id: str | None, note: str) -> None:
        self.steps.append(
            {
                "at": round(asyncio.get_running_loop().time(), 2),
                "action": action,
                "adapter": adapter,
                "request_id": request_id,
                "note": note,
            }
        )

    async def _run(self) -> None:
        try:
            sch = self.scheduler
            self.phase = "sharing_weights"
            first = sch.submit(
                PRESET_BY_KEY["legal"], _vector(SAMPLE_SEED), compute_ms=2400, source="demo"
            )
            second = sch.submit(
                PRESET_BY_KEY["legal"], _vector(SAMPLE_SEED + 1), compute_ms=2600, source="demo"
            )
            self._record("submit", "legal", first.id, "第一单：加载 legal 权重")
            self._record("submit", "legal", second.id, "第二单：共享同一份 legal 权重")

            # Let both legal requests get onto the compute slots first.
            await self._wait(lambda: len(sch.running) == 2, timeout=5)
            self.phase = "filling_budget"
            third = sch.submit(
                PRESET_BY_KEY["support"], _vector(SAMPLE_SEED + 2), compute_ms=3000, source="demo"
            )
            self._record("submit", "support", third.id, "support 权重填满预算；等第一个计算位释放")

            # Both legal requests must still hold their weights, so the third
            # (support) computes alone; wait until the second legal compute is
            # under way before filling memory with brief.
            await self._wait(
                lambda: third.started_at is not None and sch.running == [third], timeout=8
            )
            await asyncio.sleep(0.3)
            self.phase = "head_waiting"
            head = sch.submit(
                PRESET_BY_KEY["brief"], _vector(SAMPLE_SEED + 3), compute_ms=1600, source="demo"
            )
            self._record("submit", "brief", head.id, "队首需要 29056 B，预算已满，开始等内存")

            await asyncio.sleep(0.2)
            self.phase = "bypassing"
            overtaker = sch.submit(
                PRESET_BY_KEY["legal"], _vector(SAMPLE_SEED + 4), compute_ms=1800, source="demo"
            )
            self._record("submit", "legal", overtaker.id, "后到请求使用驻留 legal 权重，越过队首一次")

            await asyncio.sleep(0.1)
            waiter = sch.submit(
                PRESET_BY_KEY["brief"], _vector(SAMPLE_SEED + 5), compute_ms=1400, source="demo"
            )
            self._record(
                "submit", "brief", waiter.id,
                "队首已越过一次：此单必须等队首启动后派发，随后共享 brief 权重",
            )

            self.phase = "running"
            await self._wait(
                lambda: not sch.queue and not sch.running
                and sch.requests[waiter.id].state in ("done", "error"),
                timeout=40,
                interval=0.2,
            )
            self.phase = "finished"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface demo failures to the UI
            self.error = str(exc)
            self.phase = "error"
