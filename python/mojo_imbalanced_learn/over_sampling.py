"""Over-sampling estimators compatible with imbalanced-learn's dense API."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from ._lib import f64, gather_f64, smote_generate, smote_neighbors
from .base import BaseSampler


class SMOTE(BaseSampler):
    _sampling_kind = "over"

    def __init__(self, *, sampling_strategy="auto", random_state=None, k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _fit_resample(self, X, y):
        if sparse.issparse(X):
            raise TypeError("SMOTE currently supports dense arrays only.")
        if not isinstance(self.k_neighbors, (int, np.integer)):
            raise TypeError("k_neighbors must be an integer in this port.")
        k = int(self.k_neighbors)
        if k < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if X.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("SMOTE supports only float32 and float64 input.")
        source_dtype = X.dtype
        X_f64 = f64(X)
        generated = []
        labels = []
        for class_sample, n_samples in self.sampling_strategy_.items():
            n_samples = int(n_samples)
            if n_samples == 0:
                continue
            X_class = np.ascontiguousarray(X_f64[y == class_sample])
            if X_class.shape[0] <= k:
                raise ValueError(
                    f"Expected at least {k + 1} samples in class {class_sample!r}, "
                    f"but found {X_class.shape[0]}."
                )
            neighbors = smote_neighbors(X_class, k)
            random_state = check_random_state(self.random_state)
            selections = np.ascontiguousarray(
                random_state.randint(0, X_class.shape[0] * k, size=n_samples),
                dtype=np.int64,
            )
            steps = np.ascontiguousarray(
                random_state.uniform(size=n_samples), dtype=np.float64
            )
            X_new = smote_generate(X_class, neighbors, selections, steps, k)
            generated.append(X_new.astype(source_dtype, copy=False))
            labels.append(np.full(n_samples, class_sample, dtype=y.dtype))
        if not generated:
            return X.copy(), y.copy()
        return np.vstack([X, *generated]), np.hstack([y, *labels])


class RandomOverSampler(BaseSampler):
    _sampling_kind = "over"

    def __init__(self, *, sampling_strategy="auto", random_state=None, shrinkage=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.shrinkage = shrinkage

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        if isinstance(self.shrinkage, Real):
            self.shrinkage_ = {
                label: self.shrinkage for label in self.sampling_strategy_
            }
        elif self.shrinkage is None or isinstance(self.shrinkage, Mapping):
            self.shrinkage_ = self.shrinkage
        else:
            raise ValueError("shrinkage must be None, a non-negative float, or a dict.")
        if self.shrinkage_ is not None:
            missing = self.sampling_strategy_.keys() - self.shrinkage_.keys()
            if missing:
                raise ValueError(f"shrinkage is missing target classes: {missing}")
            if any(value < 0 for value in self.shrinkage_.values()):
                raise ValueError("shrinkage factors must be non-negative.")

        if self.shrinkage_ is None:
            all_indices = [np.arange(X.shape[0])]
            for label, n_samples in self.sampling_strategy_.items():
                target_indices = np.flatnonzero(y == label)
                all_indices.append(
                    random_state.choice(
                        target_indices, size=int(n_samples), replace=True
                    )
                )
            self.sample_indices_ = np.concatenate(all_indices).astype(int, copy=False)
            if (
                isinstance(X, np.ndarray)
                and X.dtype == np.float64
                and X.flags.c_contiguous
            ):
                X_result = gather_f64(X, self.sample_indices_)
            else:
                X_result = X[self.sample_indices_]
            return X_result, y[self.sample_indices_]

        all_indices = [np.arange(X.shape[0])]
        if sparse.issparse(X):
            X_numeric = X.astype(float)
            X_parts = [X.copy()]
        else:
            X_numeric = np.asarray(X)
            if not np.issubdtype(X_numeric.dtype, np.number):
                raise ValueError("X must be numeric when shrinkage is not None.")
            X_parts = [X_numeric.copy()]
        n_total, n_features = X.shape
        smoothing = (4 / ((n_features + 2) * n_total)) ** (1 / (n_features + 4))
        y_parts = [y.copy()]
        for label, n_samples in self.sampling_strategy_.items():
            target_indices = np.flatnonzero(y == label)
            bootstrap = random_state.choice(
                target_indices, size=int(n_samples), replace=True
            )
            all_indices.append(bootstrap)
            selected_class = (
                X_numeric[target_indices].toarray()
                if sparse.issparse(X_numeric)
                else X_numeric[target_indices]
            )
            scale = np.std(np.asarray(selected_class), axis=0)
            noise = random_state.randn(bootstrap.size, n_features)
            X_new = noise * (self.shrinkage_[label] * smoothing * scale)
            selected_base = (
                X_numeric[bootstrap].toarray()
                if sparse.issparse(X_numeric)
                else X_numeric[bootstrap]
            )
            base = np.asarray(selected_base)
            X_new = X_new + base
            X_parts.append(sparse.csr_matrix(X_new) if sparse.issparse(X) else X_new)
            y_parts.append(y[bootstrap])
        self.sample_indices_ = np.concatenate(all_indices).astype(int, copy=False)
        stack = (
            sparse.vstack(X_parts, format=X.format)
            if sparse.issparse(X)
            else np.vstack(X_parts)
        )
        return stack, np.hstack(y_parts)
