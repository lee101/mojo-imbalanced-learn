"""Sampling-strategy validation shared by the public samplers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from sklearn.utils.multiclass import type_of_target


def _stats(y):
    classes, counts = np.unique(y, return_counts=True)
    return OrderedDict(zip(classes, counts, strict=True))


def check_sampling_strategy(strategy, y, kind: str, *, stats=None):
    if stats is None:
        stats = _stats(y)
    if len(stats) <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class.")
    if callable(strategy):
        strategy = strategy(y)
    if isinstance(strategy, Real) and not isinstance(strategy, bool):
        ratio = float(strategy)
        if not 0 < ratio <= 1:
            raise ValueError("sampling_strategy as a float must be in the range (0, 1].")
        if type_of_target(y) != "binary":
            raise ValueError("sampling_strategy as a float is only valid for binary targets.")
        if kind == "over":
            majority = max(stats, key=stats.get)
            desired = int(max(stats.values()) * ratio)
            result = {
                label: desired - count
                for label, count in stats.items()
                if label != majority
            }
            if any(value <= 0 for value in result.values()):
                raise ValueError("The ratio would require removing minority samples.")
        else:
            minority = min(stats, key=stats.get)
            desired = int(min(stats.values()) / ratio)
            result = {
                label: desired for label in stats if label != minority
            }
            if any(result[label] > stats[label] for label in result):
                raise ValueError("The ratio would require generating majority samples.")
        return OrderedDict(sorted(result.items()))
    if isinstance(strategy, Mapping):
        unknown = set(strategy) - set(stats)
        if unknown:
            raise ValueError(f"Target classes {unknown} are not present in y.")
        result = {}
        for label, desired in strategy.items():
            if not isinstance(desired, Integral) or isinstance(desired, bool):
                raise TypeError("Dictionary sample counts must be integers.")
            desired = int(desired)
            if desired < 0:
                raise ValueError("The number of samples cannot be negative.")
            if kind == "over":
                if desired < stats[label]:
                    raise ValueError("Over-sampling cannot remove samples.")
                result[label] = desired - stats[label]
            else:
                if desired > stats[label]:
                    raise ValueError("Under-sampling cannot generate samples.")
                result[label] = desired
        return OrderedDict(sorted(result.items()))
    if not isinstance(strategy, str):
        raise ValueError("sampling_strategy must be a float, str, dict, or callable.")
    if strategy == "auto":
        strategy = "not majority" if kind == "over" else "not minority"
    valid = {"minority", "majority", "not minority", "not majority", "all"}
    if strategy not in valid:
        raise ValueError(f"Unknown sampling_strategy {strategy!r}.")
    if kind == "over" and strategy == "majority":
        raise ValueError("'majority' cannot be used with an over-sampler.")
    if kind == "under" and strategy == "minority":
        raise ValueError("'minority' cannot be used with an under-sampler.")
    majority = max(stats, key=stats.get)
    minority = min(stats, key=stats.get)
    if strategy == "minority":
        selected = {minority}
    elif strategy == "majority":
        selected = {majority}
    elif strategy == "not minority":
        selected = set(stats) - {minority}
    elif strategy == "not majority":
        selected = set(stats) - {majority}
    else:
        selected = set(stats)
    target = max(stats.values()) if kind == "over" else min(stats.values())
    result = {
        label: (target - stats[label] if kind == "over" else target)
        for label in stats
        if label in selected
    }
    return OrderedDict(sorted(result.items()))
