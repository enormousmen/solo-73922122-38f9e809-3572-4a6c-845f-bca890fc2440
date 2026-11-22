"""FastAPI app for the LoRA inference node demo."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters import D_IN, INPUT_DIM, OUTPUT_DIM, PRESETS, PRESET_BY_KEY
from .demo import DemoRunner
from .scheduler import Scheduler

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


def _budget_from_env() -> int:
    raw = os.environ.get("NODE_MEMORY_BYTES", "30000")
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"NODE_MEMORY_BYTES must be an integer, got {raw!r}")
    if value <= 0:
        raise SystemExit("NODE_MEMORY_BYTES must be positive")
    return value


async def _lifespan(_: FastAPI):
    yield
    # Cancel in-flight loads/computes and the demo so shutdown is quiet.
    await demo_runner.cancel()
    scheduler.reset()


scheduler = Scheduler(budget_bytes=_budget_from_env(), specs=PRESETS)
demo_runner = DemoRunner(scheduler)
app = FastAPI(title="LoRA 小内存推理节点", version="1.0.0", lifespan=_lifespan)


class SubmitRequest(BaseModel):
    adapter: str = Field(..., description="preset adapter key")
    vector: list[float] | None = Field(
        None, description=f"input vector of {D_IN} floats; omitted = sample"
    )
    compute_ms: int = Field(600, ge=0, le=60_000)
    simulate_load_failure: bool = False


def _sample_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(D_IN, dtype=np.float32)


def _parse_vector(raw: list[float] | None, seed: int) -> np.ndarray:
    if raw is None:
        return _sample_vector(seed)
    if len(raw) != D_IN:
        raise HTTPException(status_code=422, detail=f"vector must contain {D_IN} values")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise HTTPException(status_code=422, detail="vector must contain finite values")
    return vector.astype(np.float32)


@app.get("/api/presets")
async def presets() -> dict[str, Any]:
    return {
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "budget_bytes": scheduler.budget_bytes,
        "max_concurrent": scheduler.max_concurrent,
        "presets": [
            {
                "key": spec.key,
                "name": spec.name,
                "rank": spec.rank,
                "bytes": spec.nbytes,
                "load_delay": spec.load_delay,
            }
            for spec in PRESETS
        ],
    }


@app.get("/api/state")
async def state() -> dict[str, Any]:
    snap = scheduler.snapshot()
    snap["demo"] = demo_runner.status()
    return snap


@app.get("/api/requests/{request_id}")
async def request_detail(request_id: str) -> dict[str, Any]:
    detail = scheduler.request_detail(request_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="request not found")
    return detail


@app.post("/api/requests", status_code=201)
async def submit(payload: SubmitRequest) -> dict[str, Any]:
    spec = PRESET_BY_KEY.get(payload.adapter)
    if spec is None:
        raise HTTPException(status_code=422, detail=f"unknown adapter {payload.adapter!r}")
    vector = _parse_vector(payload.vector, seed=len(scheduler.requests) + 1)
    try:
        req = scheduler.submit(
            spec,
            vector,
            compute_ms=payload.compute_ms,
            simulate_load_failure=payload.simulate_load_failure,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    position = scheduler.queue.index(req) + 1 if req in scheduler.queue else None
    return {"id": req.id, "position": position}


@app.post("/api/demo")
async def start_demo() -> dict[str, Any]:
    if demo_runner.running:
        raise HTTPException(status_code=409, detail="样例已在运行")
    if scheduler.queue or scheduler.running:
        raise HTTPException(
            status_code=409, detail="节点正在处理请求，请先重置或等待队列清空"
        )
    started = await demo_runner.start()
    if not started:
        raise HTTPException(status_code=409, detail="样例已在运行")
    return {"status": "running", "plan": demo_runner.plan()}


@app.post("/api/reset")
async def reset() -> dict[str, str]:
    await demo_runner.cancel()
    scheduler.reset()
    return {"status": "reset"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
