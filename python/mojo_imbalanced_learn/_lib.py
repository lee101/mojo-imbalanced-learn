"""ctypes bridge for the Mojo resampling kernels."""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_IMBALANCED_LEARN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-imbalanced-learn.so"
)

I = ctypes.c_int64
STATUS = ctypes.c_int32
_SIGNATURES = {
    "mil_smote_neighbors": ([I, I, I, I, I, I], STATUS),
    "mil_smote_neighbors_range": ([I, I, I, I, I, I, I, I], STATUS),
    "mil_smote_generate": ([I, I, I, I, I, I, I, I, I], STATUS),
    "mil_gather_f64": ([I, I, I, I, I, I], STATUS),
}

# A brute-force k-nearest-neighbour sweep is n*n*d fused multiply-adds. Measured
# on this box the fan-out only wins above roughly eight million of them, and it
# reaches 2.8x at eight workers, so smaller sweeps stay on one core.
NEIGHBOR_PARALLEL_WORK = 1 << 23
NEIGHBOR_MAX_WORKERS = 8


def _worker_count(items: int) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(items, NEIGHBOR_MAX_WORKERS, available))


def _spans(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``total`` rows into ``parts`` contiguous, near-equal spans."""
    step = -(-total // parts)
    return [
        (lo, min(lo + step, total)) for lo in range(0, total, step) if lo < total
    ]

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("ABI buffers must be C-contiguous NumPy arrays")
    if array.size and array.ctypes.data == 0:
        raise ValueError("ABI buffers must have non-null storage")
    return array.ctypes.data


def f64(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)

def _check_status(name: str, status: int) -> None:
    messages = {
        1: "received a null buffer",
        2: "received invalid dimensions",
        3: "received an out-of-range index",
    }
    if status:
        raise RuntimeError(f"{name} failed: {messages.get(status, f'error {status}')}")


def smote_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    if X.dtype != np.float64 or X.ndim != 2 or not X.flags.c_contiguous:
        raise TypeError("SMOTE neighbor input must be a C-contiguous float64 matrix")
    n, d = X.shape
    if n == 0 or d == 0 or k < 1 or k >= n:
        raise ValueError("invalid matrix shape or neighbor count")
    indices = np.empty((n, k + 1), dtype=np.int64)
    distances = np.empty((n, k + 1), dtype=np.float64)
    if n * n * d < NEIGHBOR_PARALLEL_WORK:
        status = lib().mil_smote_neighbors(
            addr(X), addr(indices), addr(distances), n, d, k
        )
        _check_status("mil_smote_neighbors", status)
        return indices

    workers = _worker_count(n)
    function = lib().mil_smote_neighbors_range
    arguments = (addr(X), addr(indices), addr(distances), n, d, k + 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(
            pool.map(
                lambda span: function(*arguments, span[0], span[1]),
                _spans(n, workers),
            )
        )
    _check_status("mil_smote_neighbors_range", max(statuses))
    return indices


def smote_generate(
    X: np.ndarray,
    neighbors: np.ndarray,
    selections: np.ndarray,
    steps: np.ndarray,
    k: int,
) -> np.ndarray:
    n, d = X.shape
    selections_i64 = np.ascontiguousarray(selections, dtype=np.int64)
    steps_f64 = np.ascontiguousarray(steps, dtype=np.float64)
    if (
        X.dtype != np.float64
        or not X.flags.c_contiguous
        or neighbors.dtype != np.int64
        or not neighbors.flags.c_contiguous
        or neighbors.shape != (n, k + 1)
        or selections_i64.ndim != 1
        or steps_f64.shape != selections_i64.shape
    ):
        raise TypeError("invalid SMOTE generation buffers")
    result = np.empty((selections_i64.size, d), dtype=np.float64)
    if result.shape[0] == 0:
        return result
    status = lib().mil_smote_generate(
        addr(X),
        addr(neighbors),
        addr(selections_i64),
        addr(steps_f64),
        addr(result),
        n,
        d,
        k,
        result.shape[0],
    )
    _check_status("mil_smote_generate", status)
    return result


def gather_f64(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if X.dtype != np.float64 or X.ndim != 2 or not X.flags.c_contiguous:
        raise TypeError("gather input must be a C-contiguous float64 matrix")
    indices_i64 = np.ascontiguousarray(indices, dtype=np.int64)
    if indices_i64.ndim != 1:
        raise ValueError("gather indices must be one-dimensional")
    if np.any(indices_i64 < 0) or np.any(indices_i64 >= X.shape[0]):
        raise IndexError("gather index is out of bounds")
    result = np.empty((indices_i64.size, X.shape[1]), dtype=np.float64)
    if result.shape[0] == 0:
        return result
    status = lib().mil_gather_f64(
        addr(X),
        addr(indices_i64),
        addr(result),
        X.shape[0],
        indices_i64.size,
        X.shape[1],
    )
    _check_status("mil_gather_f64", status)
    return result
