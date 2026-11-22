"""Memory-budgeted scheduler for a two-slot CPU LoRA node.

Rules implemented here:

* one resident copy per adapter, shared by every request that uses it;
* memory is charged by the exact array byte count from the moment a load
  starts until the adapter is evicted (only possible when no request uses it);
* at most two compute slots;
* when the queue head cannot be loaded for lack of memory and a compute slot
  is free, one later request whose adapter is already resident may overtake
  the head.  Each head may be overtaken at most once;
* if idle cached weights can be evicted to fit the head, they are evicted
  first, so the head always wins over a bypass;
* a failed load fails only that request, releases its bytes, and leaves
  running computes alone.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .adapters import (
    D_IN,
    OUTPUT_DIM,
    PRESETS,
    AdapterSpec,
    apply_linear,
    load_arrays,
)

MAX_CONCURRENT_COMPUTES = 2


@dataclass
class Request:
    id: str
    spec: AdapterSpec
    vector: np.ndarray
    compute_delay: float
    simulate_load_failure: bool
    source: str = "manual"
    state: str = "queued"  # queued | loading | ready | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    skipped_head: bool = False
    blocked_on_memory: bool = False
    waiting_reported: bool = False
    held_reported: bool = False
    load_attempts: int = 0
    load_delay: float | None = None  # optional override (demo/tests)
    shared_with: int = 0
    result: np.ndarray | None = None
    error: str | None = None
    task: asyncio.Task | None = None


@dataclass
class Resident:
    key: str
    spec: AdapterSpec
    nbytes: int
    refcount: int = 0
    loading: bool = True
    weights: dict[str, np.ndarray] | None = None
    loaded_at: float | None = None
    last_used: float = field(default_factory=time.time)


class Scheduler:
    def __init__(
        self,
        *,
        budget_bytes: int = 30_000,
        specs: Iterable[AdapterSpec] = PRESETS,
        max_concurrent: int = MAX_CONCURRENT_COMPUTES,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.budget_bytes = budget_bytes
        self.max_concurrent = max_concurrent
        self.specs: dict[str, AdapterSpec] = {spec.key: spec for spec in specs}
        self.queue: list[Request] = []
        self.running: list[Request] = []
        self.resident: dict[str, Resident] = {}
        self.used_bytes = 0
        self.requests: dict[str, Request] = {}
        self.finished: list[Request] = []
        self.finished_total = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=400)
        self._event_seq = 0
        self._generation = 0
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ API

    def submit(
        self,
        spec: AdapterSpec,
        vector: np.ndarray,
        *,
        compute_ms: int = 600,
        simulate_load_failure: bool = False,
        source: str = "manual",
        load_delay: float | None = None,
    ) -> Request:
        if spec.key not in self.specs:
            raise ValueError(f"unknown adapter {spec.key!r}")
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (D_IN,):
            raise ValueError(f"vector must contain {D_IN} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("vector contains non-finite values")
        req = Request(
            id=uuid.uuid4().hex[:8],
            spec=self.specs[spec.key],
            vector=vector,
            compute_delay=max(0, compute_ms) / 1000.0,
            simulate_load_failure=simulate_load_failure,
            source=source,
            load_delay=load_delay,
        )
        self.requests[req.id] = req
        self.queue.append(req)
        self._log("submitted", req, f"请求进入队列，目标适配器 {req.spec.name}")
        self._pump()
        return req

    def reset(self) -> None:
        """Cancel all background work and return to an empty node."""
        self._generation += 1
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        self.queue.clear()
        self.running.clear()
        self.resident.clear()
        self.used_bytes = 0
        self.requests.clear()
        self.finished.clear()
        self.finished_total = 0
        self.events.clear()
        self._event_seq = 0
        self._log_raw("reset", None, "节点已重置，内存与队列清空")

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        queue_view = []
        for pos, req in enumerate(self.queue, start=1):
            queue_view.append(self._request_view(req, now, position=pos))
        return {
            "now": now,
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.budget_bytes - self.used_bytes,
            "max_concurrent": self.max_concurrent,
            "queue": queue_view,
            "running": [self._request_view(r, now) for r in self.running],
            "finished": [
                self._request_view(r, now) for r in self.finished[-12:]
            ],
            "finished_total": self.finished_total,
            "finished_window": 12,
            "resident": [
                {
                    "key": r.key,
                    "name": r.spec.name,
                    "bytes": r.nbytes,
                    "refcount": r.refcount,
                    "loading": r.loading,
                    "in_use": r.refcount > 0,
                    "loaded_at": r.loaded_at,
                    "idle_seconds": (
                        None if r.refcount > 0 or r.loading else round(now - r.last_used, 2)
                    ),
                }
                for r in self.resident.values()
            ],
            "events": list(self.events),
        }

    def request_detail(self, request_id: str) -> dict[str, Any] | None:
        req = self.requests.get(request_id)
        if req is None:
            return None
        return self._request_view(req, time.time(), include_result=True)

    # ------------------------------------------------------------- internals

    def _request_view(
        self, req: Request, now: float, *, position: int | None = None, include_result=False
    ) -> dict[str, Any]:
        is_head = bool(self.queue) and self.queue[0] is req
        started = req.started_at
        view = {
            "id": req.id,
            "adapter": req.spec.key,
            "adapter_name": req.spec.name,
            "state": req.state,
            "source": req.source,
            "position": position,
            "is_head": is_head,
            "skipped_head": req.skipped_head,
            "blocked_on_memory": req.blocked_on_memory and is_head,
            "compute_ms": round(req.compute_delay * 1000),
            "shared_with": req.shared_with,
            "load_attempts": req.load_attempts,
            "created_at": req.created_at,
            "started_at": started,
            "finished_at": req.finished_at,
            "queue_seconds": round((started or now) - req.created_at, 2),
            "run_seconds": (
                round((req.finished_at or now) - started, 2) if started else None
            ),
            "error": req.error,
            "result_preview": (
                [round(float(x), 5) for x in req.result[:8]] if req.result is not None else None
            ),
        }
        if include_result or req.state in ("done", "error"):
            view["result"] = (
                [round(float(x), 5) for x in req.result] if req.result is not None else None
            )
        return view

    def _log_raw(self, kind: str, req: Request | None, message: str, **extra: Any) -> None:
        self._event_seq += 1
        entry: dict[str, Any] = {
            "seq": self._event_seq,
            "ts": time.time(),
            "kind": kind,
            "request_id": req.id if req else None,
            "adapter": req.spec.key if req else None,
            "message": message,
        }
        entry.update(extra)
        self.events.append(entry)

    def _log(self, kind: str, req: Request, message: str, **extra: Any) -> None:
        self._log_raw(kind, req, message, **extra)

    def _spawn_load(self, req: Request) -> None:
        req.task = asyncio.create_task(self._load(req, self._generation))
        self._tasks.add(req.task)
        req.task.add_done_callback(self._tasks.discard)

    def _pump(self) -> None:
        """Advance the state machine. Runs synchronously on the event loop."""
        for _ in range(200):  # defensive loop bound
            if not self.queue:
                return
            head = self.queue[0]

            if head.state == "loading":
                return  # later requests deliberately wait for the head
            if head.state == "ready":
                if len(self.running) >= self.max_concurrent:
                    return
                self._start_compute(head)
                continue
            if head.state != "queued":  # stale entry, should not happen
                self.queue.pop(0)
                continue

            resident = self.resident.get(head.spec.key)
            if resident is not None and resident.loading:
                # Head's adapter is being loaded by an earlier reservation.
                return
            if resident is not None and resident.weights is not None:
                if len(self.running) >= self.max_concurrent:
                    return
                resident.refcount += 1
                resident.last_used = time.time()
                head.state = "ready"
                continue

            # Whether the head could ever fit, after evicting idle weights.
            if not self._could_fit(head.spec.nbytes):
                head.blocked_on_memory = True
                if not head.waiting_reported:
                    head.waiting_reported = True
                    idle_bytes = sum(
                        r.nbytes
                        for r in self.resident.values()
                        if r.refcount == 0 and not r.loading
                    )
                    self._log(
                        "waiting_memory",
                        head,
                        f"队首缺内存：需要 {head.spec.nbytes} B，"
                        f"当前占用 {self.used_bytes} B，可回收空闲 {idle_bytes} B；"
                        "等待在用权重释放",
                    )

                # Only the current head gets one bypass opportunity.
                if head.skipped_head:
                    if not head.held_reported:
                        head.held_reported = True
                        self._log(
                            "head_holds_line",
                            head,
                            f"队首 {head.id} 已越过一次，本次不再放行任何后到请求；"
                            "等待在用权重计算结束并释放内存",
                        )
                    return
                if len(self.running) >= self.max_concurrent:
                    return  # bypass needs a free compute slot
                candidate = self._find_bypass_candidate()
                if candidate is None:
                    return
                head.skipped_head = True
                head.blocked_on_memory = True
                cand_resident = self.resident[candidate.spec.key]
                cand_resident.refcount += 1
                cand_resident.last_used = time.time()
                candidate.state = "ready"
                self._log(
                    "head_bypassed",
                    head,
                    f"队首 {head.id} 继续等内存，先放行后到请求 {candidate.id} "
                    f"使用已驻留的 {candidate.spec.name}（共享权重）",
                    bypassed_request_id=candidate.id,
                )
                self._log(
                    "bypass_dispatched",
                    candidate,
                    f"后到请求 {candidate.id} 越过队首开始使用驻留权重",
                )
                self._start_compute(candidate)
                # The head has used its single bypass; stop dispatching behind it.
                return

            # Space can be made: the head always wins over any bypass.
            head.blocked_on_memory = False
            if len(self.running) >= self.max_concurrent:
                return
            if self._try_make_space(head.spec.nbytes):
                self._begin_load(head)
            return

    def _could_fit(self, nbytes: int) -> bool:
        if self.used_bytes + nbytes <= self.budget_bytes:
            return True
        idle_bytes = sum(
            r.nbytes for r in self.resident.values() if r.refcount == 0 and not r.loading
        )
        return self.used_bytes - idle_bytes + nbytes <= self.budget_bytes

    def _try_make_space(self, nbytes: int) -> bool:
        if self.used_bytes + nbytes <= self.budget_bytes:
            return True
        idle = sorted(
            (r for r in self.resident.values() if r.refcount == 0 and not r.loading),
            key=lambda r: r.last_used,
        )
        freed = 0
        chosen: list[Resident] = []
        for r in idle:
            chosen.append(r)
            freed += r.nbytes
            if self.used_bytes - freed + nbytes <= self.budget_bytes:
                for victim in chosen:
                    self._evict(victim)
                return True
        return False  # do not evict anything if it still would not fit

    def _evict(self, resident: Resident) -> None:
        self.resident.pop(resident.key, None)
        self.used_bytes -= resident.nbytes
        self._log_raw(
            "evicted",
            None,
            f"卸载空闲权重 {resident.spec.name}，归还 {resident.nbytes} B",
            adapter=resident.key,
        )

    def _find_bypass_candidate(self) -> Request | None:
        for req in self.queue[1:]:
            if req.state != "queued":
                continue
            resident = self.resident.get(req.spec.key)
            if resident is not None and resident.weights is not None and not resident.loading:
                return req
        return None

    def _begin_load(self, req: Request) -> None:
        resident = Resident(
            key=req.spec.key,
            spec=req.spec,
            nbytes=req.spec.nbytes,
            refcount=1,  # reservation for this request
            loading=True,
        )
        self.resident[req.spec.key] = resident
        self.used_bytes += resident.nbytes
        req.state = "loading"
        req.load_attempts += 1
        req.blocked_on_memory = False
        self._log(
            "load_start",
            req,
            f"开始加载 {req.spec.name}，立即计入 {resident.nbytes} B 预算"
            f"（已用 {self.used_bytes}/{self.budget_bytes} B）",
            bytes=resident.nbytes,
        )
        self._spawn_load(req)

    async def _load(self, req: Request, generation: int) -> None:
        try:
            await asyncio.sleep(
                req.spec.load_delay if req.load_delay is None else req.load_delay
            )
            if generation != self._generation or req.state != "loading":
                return
            try:
                weights = load_arrays(req.spec, fail=req.simulate_load_failure)
            except Exception as exc:  # simulated/external load error
                self._load_failed(req, str(exc))
                return
            resident = self.resident.get(req.spec.key)
            if resident is None:
                return
            resident.weights = weights
            resident.loading = False
            resident.loaded_at = time.time()
            resident.last_used = time.time()
            req.state = "ready"
            self._log(
                "load_done",
                req,
                f"{req.spec.name} 加载完成，权重驻留，{req.spec.nbytes} B",
            )
        except asyncio.CancelledError:
            return
        self._pump()

    def _load_failed(self, req: Request, message: str) -> None:
        resident = self.resident.get(req.spec.key)
        if req in self.queue:
            self.queue.remove(req)
        req.state = "error"
        req.error = f"加载失败: {message}"
        req.finished_at = time.time()
        req.blocked_on_memory = False
        if resident is not None and resident.loading:
            resident.refcount = max(0, resident.refcount - 1)
            if resident.refcount == 0:
                self.resident.pop(req.spec.key, None)
                self.used_bytes -= resident.nbytes
                returned = resident.nbytes
            else:
                returned = 0
        else:
            returned = 0
        self.finished.append(req)
        self.finished_total += 1
        self._log(
            "load_failed",
            req,
            f"{req.spec.name} 加载失败，请求报错；归还 {returned} B 预算",
            bytes=returned,
        )
        self._pump()

    def _start_compute(self, req: Request) -> None:
        if req in self.queue:
            self.queue.remove(req)
        resident = self.resident.get(req.spec.key)
        req.state = "running"
        req.started_at = time.time()
        req.blocked_on_memory = False
        req.shared_with = resident.refcount if resident else 0
        if resident is not None:
            resident.last_used = time.time()
        self.running.append(req)
        self._log(
            "compute_start",
            req,
            f"开始计算 {req.id}（{req.spec.name}），"
            f"权重引用 {req.shared_with} 份请求共享，占用计算位 {len(self.running)}/{self.max_concurrent}",
        )
        req.task = asyncio.create_task(self._compute(req, self._generation))
        self._tasks.add(req.task)
        req.task.add_done_callback(self._tasks.discard)

    async def _compute(self, req: Request, generation: int) -> None:
        error: str | None = None
        result: np.ndarray | None = None
        try:
            await asyncio.sleep(req.compute_delay)
            if generation != self._generation:
                return
            resident = self.resident.get(req.spec.key)
            if resident is None or resident.weights is None:
                raise RuntimeError("resident weights disappeared before compute")
            result = apply_linear(resident.weights, req.spec, req.vector)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            error = f"计算失败: {exc}"
        if generation != self._generation:
            return

        if req in self.running:
            self.running.remove(req)
        resident = self.resident.get(req.spec.key)
        if resident is not None:
            resident.refcount = max(0, resident.refcount - 1)
            resident.last_used = time.time()
        req.finished_at = time.time()
        if error is not None:
            req.state = "error"
            req.error = error
            self._log("compute_failed", req, error)
        else:
            req.state = "done"
            req.result = result
            self._log(
                "compute_done",
                req,
                f"请求 {req.id} 完成，输出 {OUTPUT_DIM} 维；"
                f"权重引用降为 {resident.refcount if resident else 0}",
            )
        self.finished.append(req)
        self.finished_total += 1
        self._pump()
