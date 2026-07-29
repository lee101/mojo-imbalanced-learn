"""End-to-end benchmarks against imbalanced-learn on identical arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from imblearn.over_sampling import RandomOverSampler as UpstreamOver  # noqa: E402
from imblearn.over_sampling import SMOTE as UpstreamSMOTE  # noqa: E402
from imblearn.under_sampling import RandomUnderSampler as UpstreamUnder  # noqa: E402
from mojo_imbalanced_learn import RandomOverSampler, RandomUnderSampler, SMOTE  # noqa: E402


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def data(count, features, minority, seed):
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.normal(size=(count, features)))
    y = np.zeros(count, dtype=np.int64)
    y[minority:] = 1
    return X, y


def cases():
    X_smote, y_smote = data(10_000, 20, 1_000, 1)
    X_wide, y_wide = data(4_000, 100, 600, 2)
    X_random, y_random = data(500_000, 8, 50_000, 3)
    return [
        (
            "SMOTE (10k x 20, 10% minority)",
            lambda: SMOTE(random_state=0).fit_resample(X_smote, y_smote),
            lambda: UpstreamSMOTE(random_state=0).fit_resample(X_smote, y_smote),
        ),
        (
            "SMOTE (4k x 100, 15% minority)",
            lambda: SMOTE(random_state=0).fit_resample(X_wide, y_wide),
            lambda: UpstreamSMOTE(random_state=0).fit_resample(X_wide, y_wide),
        ),
        (
            "RandomOverSampler (500k x 8)",
            lambda: RandomOverSampler(random_state=0).fit_resample(X_random, y_random),
            lambda: UpstreamOver(random_state=0).fit_resample(X_random, y_random),
        ),
        (
            "RandomUnderSampler (500k x 8)",
            lambda: RandomUnderSampler(random_state=0).fit_resample(X_random, y_random),
            lambda: UpstreamUnder(random_state=0).fit_resample(X_random, y_random),
        ),
    ]


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| case | Mojo port | imbalanced-learn | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_seconds = best_time(ours)
        upstream_seconds = best_time(upstream)
        ratio = upstream_seconds / mojo_seconds
        outcome = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {outcome} |"
        )


if __name__ == "__main__":
    main()
