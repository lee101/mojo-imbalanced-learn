"""Random under-sampling."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from ._lib import gather_f64
from .base import BaseSampler


class RandomUnderSampler(BaseSampler):
    _sampling_kind = "under"

    def __init__(self, *, sampling_strategy="auto", random_state=None, replacement=False):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement

    def _fit_resample(self, X, y):
        if not isinstance(self.replacement, (bool, np.bool_)):
            raise TypeError("replacement must be a boolean.")
        random_state = check_random_state(self.random_state)
        selected = []
        for label in self._class_labels:
            target_indices = np.flatnonzero(y == label)
            if label in self.sampling_strategy_:
                local = random_state.choice(
                    target_indices.size,
                    size=int(self.sampling_strategy_[label]),
                    replace=self.replacement,
                )
                target_indices = target_indices[local]
            selected.append(target_indices)
        self.sample_indices_ = np.concatenate(selected).astype(int, copy=False)
        if (
            isinstance(X, np.ndarray)
            and X.dtype == np.float64
            and X.flags.c_contiguous
        ):
            X_result = gather_f64(X, self.sample_indices_)
        else:
            X_result = X[self.sample_indices_]
        return X_result, y[self.sample_indices_]
