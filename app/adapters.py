"""Deterministic mock LoRA artifacts generated on the CPU.

Each preset adapter is a pair of low-rank matrices (A: r x d_in, B: d_out x r)
plus a float32 bias.  The "size" charged to the memory budget is the exact
number of bytes those arrays occupy, so the scheduler and the numerical
pipeline agree on what is resident.

Numerical overflow policy
-------------------------
The layer is evaluated in float32, exactly as the resident weights are
stored.  If an intermediate or final value overflows float32 (inf/nan), the
same layer is recomputed in float64 using only O(d_in + rank + d_out)
temporary space — the resident float32 weights are shared read-only and
cast one row at a time, never copied wholesale.  When every final value
then fits the finite float32 range it is cast back and returned; otherwise
:class:`OutputOutOfRangeError` is raised so the scheduler can fail just
that one request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

D_IN = 128
D_OUT = 96

FLOAT32_MAX = float(np.finfo(np.float32).max)


class OutputOutOfRangeError(RuntimeError):
    """Final layer values cannot be represented as finite float32."""


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
    """Run y = (x @ A^T) @ B^T * scale + bias on one input vector.

    The float32 fast path is bit-identical to the original pipeline, so any
    request that produced a finite result before keeps its exact output.  If
    an intermediate or final value overflows float32, the same layer is
    recomputed in float64 (see :func:`_apply_linear_float64`); results that
    then fit the finite float32 range are cast back and returned in full,
    otherwise :class:`OutputOutOfRangeError` is raised.
    """
    if vector.shape != (D_IN,):
        raise ValueError(f"expected vector of shape ({D_IN},), got {vector.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        hidden = vector @ weights["a"].T
        out = (hidden @ weights["b"].T) * spec.scale + weights["bias"]
    if not (np.all(np.isfinite(hidden)) and np.all(np.isfinite(out))):
        out64 = _apply_linear_float64(weights, spec, vector)
        bad = ~np.isfinite(out64) | (np.abs(out64) > FLOAT32_MAX)
        if np.any(bad):
            raise OutputOutOfRangeError(
                f"输出 {int(np.sum(bad))}/{D_OUT} 个值超出 float32 有限范围"
                f"（|y| 峰值 {float(np.abs(out64).max()):.3e}，上限 {FLOAT32_MAX:.3e}），"
                "float64 重算后仍无法转回 float32，结果留空"
            )
        out = out64.astype(np.float32)
    if delay is not None:
        delay()
    return out.astype(np.float32, copy=False)


def _apply_linear_float64(
    weights: dict[str, np.ndarray],
    spec: AdapterSpec,
    vector: np.ndarray,
) -> np.ndarray:
    """Recompute the same linear layer in float64 after a float32 overflow.

    Temporary space grows only linearly with the vector lengths
    (D_IN + rank + D_OUT): the resident float32 weights are shared
    read-only and cast one row at a time, so no float64 copy of the weight
    matrices is ever materialised.
    """
    x = vector.astype(np.float64)  # (D_IN,) temporary
    hidden = np.empty(spec.rank, dtype=np.float64)  # (rank,) intermediate
    a = weights["a"]  # (rank, D_IN) float32, shared resident weights
    for j in range(spec.rank):
        hidden[j] = np.dot(x, a[j].astype(np.float64))  # one (D_IN,) row cast
    out = np.empty(D_OUT, dtype=np.float64)  # (D_OUT,) output
    b = weights["b"]  # (D_OUT, rank) float32, shared resident weights
    for k in range(D_OUT):
        out[k] = np.dot(hidden, b[k].astype(np.float64))  # one (rank,) row cast
    out *= float(spec.scale)
    out += weights["bias"].astype(np.float64)  # one (D_OUT,) cast
    return out
