"""Estimator base classes."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

from ._strategy import _stats, check_sampling_strategy


class BaseSampler(BaseEstimator):
    _sampling_kind = ""

    def fit(self, X, y, **params):
        self._validate(X, y)
        return self

    def fit_resample(self, X, y, **params):
        X_checked, y_checked = self._validate(X, y)
        return self._fit_resample(X_checked, y_checked)

    def _validate(self, X, y):
        X_checked, y_checked = check_X_y(
            X, y, accept_sparse=("csr", "csc"), dtype=None
        )
        if y_checked.ndim != 1:
            raise ValueError("Only one-dimensional classification targets are supported.")
        self.n_features_in_ = X_checked.shape[1]
        class_stats = _stats(y_checked)
        self._class_labels = tuple(class_stats)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy,
            y_checked,
            self._sampling_kind,
            stats=class_stats,
        )
        return X_checked, y_checked
