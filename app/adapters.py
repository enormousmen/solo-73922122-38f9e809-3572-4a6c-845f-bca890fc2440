"""Deterministic mock LoRA artifacts generated on the CPU.

Each preset adapter is a pair of low-rank matrices (A: r x d_in, B: d_out x r)
plus a float32 bias.  The "size" charged to the memory budget is the exact
number of bytes those arrays occupy, so the scheduler and the numerical
pipeline agree on what is resident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

D_IN = 128
D_OUT = 96


@dataclass(frozen=True)
class AdapterSpec:
    key: str
    name: str
    rank: int
    scale: float
    seed: int
    load_delay: float = 0.35  # simulated disk/validation latency

    @property
    def nbytes(self) -> int:
        """Bytes of the resident arrays (A, B, bias), matching `load_arrays`."""
        elements = self.rank * D_IN + D_OUT * self.rank + D_OUT
        return elements * np.dtype(np.float32).itemsize


PRESETS: tuple[AdapterSpec, ...] = (
    AdapterSpec("support", "支持语气", 8, 0.30, 11, load_delay=0.4),
    AdapterSpec("finance", "财务口吻", 16, 0.35, 23, load_delay=0.5),
    AdapterSpec("legal", "法律措辞", 24, 0.25, 37, load_delay=0.6),
    AdapterSpec("brief", "极简摘要", 32, 0.20, 41, load_delay=0.7),
)
PRESET_BY_KEY = {spec.key: spec for spec in PRESETS}
INPUT_DIM = D_IN
OUTPUT_DIM = D_OUT


def load_arrays(spec: AdapterSpec, fail: bool = False) -> dict[str, np.ndarray]:
    """Materialise an adapter's weights. Raises RuntimeError when `fail`."""
    if fail:
        raise RuntimeError(f"adapter {spec.key!r} failed integrity validation")
    rng = np.random.default_rng(spec.seed)
    a = rng.standard_normal((spec.rank, D_IN), dtype=np.float32)
    b = rng.standard_normal((D_OUT, spec.rank), dtype=np.float32)
    bias = rng.standard_normal((D_OUT,), dtype=np.float32) * 0.1
    return {"a": a, "b": b, "bias": bias}


def apply_linear(
    weights: dict[str, np.ndarray],
    spec: AdapterSpec,
    vector: np.ndarray,
    delay: Callable[[], None] | None = None,
) -> np.ndarray:
    """Run y = (x @ A^T) @ B^T * scale + bias on one input vector."""
    if vector.shape != (D_IN,):
        raise ValueError(f"expected vector of shape ({D_IN},), got {vector.shape}")
    hidden = vector @ weights["a"].T
    out = (hidden @ weights["b"].T) * spec.scale + weights["bias"]
    if delay is not None:
        delay()
    return out.astype(np.float32, copy=False)
