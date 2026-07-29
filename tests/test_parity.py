from __future__ import annotations

import numpy as np
import pytest
from imblearn.over_sampling import RandomOverSampler as UpstreamOver
from imblearn.over_sampling import SMOTE as UpstreamSMOTE
from imblearn.under_sampling import RandomUnderSampler as UpstreamUnder
from scipy import sparse
from sklearn.base import clone

from mojo_imbalanced_learn import RandomOverSampler, RandomUnderSampler, SMOTE
from mojo_imbalanced_learn._lib import gather_f64, lib, smote_neighbors


def multiclass_data(dtype=np.float64):
    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(rng.normal(size=(33, 6)).astype(dtype))
    y = np.array(["small"] * 7 + ["large"] * 15 + ["middle"] * 11)
    return X, y


def assert_same_result(ours, upstream, X, y, *, atol=0):
    X_ours, y_ours = ours.fit_resample(X, y)
    X_upstream, y_upstream = upstream.fit_resample(X, y)
    if sparse.issparse(X_upstream):
        assert np.allclose(X_ours.toarray(), X_upstream.toarray(), atol=atol, rtol=0)
    elif X_upstream.dtype == object:
        assert np.array_equal(X_ours, X_upstream)
    else:
        assert np.allclose(X_ours, X_upstream, atol=atol, rtol=0)
    assert np.array_equal(y_ours, y_upstream)
    assert ours.sampling_strategy_ == upstream.sampling_strategy_
    assert ours.n_features_in_ == upstream.n_features_in_
    return X_ours, y_ours


def test_smote_binary_matches_upstream():
    rng = np.random.default_rng(1)
    X = np.ascontiguousarray(rng.normal(size=(40, 8)))
    y = np.array([0] * 13 + [1] * 27)
    assert_same_result(
        SMOTE(random_state=42, k_neighbors=5),
        UpstreamSMOTE(random_state=42, k_neighbors=5),
        X,
        y,
        atol=5e-15,
    )


def test_smote_multiclass_matches_upstream():
    X, y = multiclass_data()
    assert_same_result(
        SMOTE(random_state=9, k_neighbors=3),
        UpstreamSMOTE(random_state=9, k_neighbors=3),
        X,
        y,
        atol=5e-15,
    )


def test_smote_dict_strategy_matches_upstream():
    X, y = multiclass_data()
    strategy = {"small": 18, "middle": 16}
    assert_same_result(
        SMOTE(sampling_strategy=strategy, random_state=3, k_neighbors=2),
        UpstreamSMOTE(sampling_strategy=strategy, random_state=3, k_neighbors=2),
        X,
        y,
        atol=5e-15,
    )


def test_smote_float_strategy_matches_upstream():
    rng = np.random.default_rng(11)
    X = np.ascontiguousarray(rng.normal(size=(30, 4)))
    y = np.array([0] * 10 + [1] * 20)
    assert_same_result(
        SMOTE(sampling_strategy=0.75, random_state=4, k_neighbors=3),
        UpstreamSMOTE(sampling_strategy=0.75, random_state=4, k_neighbors=3),
        X,
        y,
        atol=5e-15,
    )


def test_smote_float32_preserves_dtype_and_values():
    X, y = multiclass_data(np.float32)
    result, labels = assert_same_result(
        SMOTE(random_state=2, k_neighbors=2),
        UpstreamSMOTE(random_state=2, k_neighbors=2),
        X,
        y,
        atol=5e-7,
    )
    assert result.dtype == np.float32
    assert labels.dtype == y.dtype


def test_smote_rejects_integer_dtype_that_would_narrow_at_the_abi():
    X, y = multiclass_data()
    with pytest.raises(TypeError, match="float32 and float64"):
        SMOTE(k_neighbors=2).fit_resample(X.astype(np.int64), y)


def test_smote_is_deterministic():
    X, y = multiclass_data()
    sampler = SMOTE(random_state=13, k_neighbors=3)
    first = sampler.fit_resample(X, y)
    second = sampler.fit_resample(X, y)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_smote_rejects_too_few_class_samples():
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="at least 4"):
        SMOTE(k_neighbors=3).fit_resample(X, y)


@pytest.mark.parametrize("n", [50, 350])
def test_smote_neighbor_kernel_matches_sklearn(n):
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(21)
    X = np.ascontiguousarray(rng.normal(size=(n, 9)))
    k = 6
    indices = smote_neighbors(X, k)
    expected = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X, return_distance=False
    )
    assert np.array_equal(indices, expected)


@pytest.mark.parametrize("rows", [37, 40_000])
def test_gather_kernel_simd_tail_serial_and_parallel(rows):
    rng = np.random.default_rng(22)
    X = np.ascontiguousarray(rng.normal(size=(5_000, 7)))
    indices = rng.integers(0, X.shape[0], size=rows, dtype=np.int64)
    assert np.array_equal(gather_f64(X, indices), X[indices])


def test_gather_rejects_invalid_buffers_and_indices():
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    with pytest.raises(TypeError, match="C-contiguous float64"):
        gather_f64(X.astype(np.float32), np.array([0]))
    with pytest.raises(IndexError, match="out of bounds"):
        gather_f64(X, np.array([4]))


def test_raw_abi_rejects_null_buffers_without_dereferencing():
    assert lib().mil_smote_neighbors(0, 0, 0, 2, 3, 1) == 1
    assert lib().mil_smote_generate(0, 0, 0, 0, 0, 2, 3, 1, 1) == 1
    assert lib().mil_gather_f64(0, 0, 0, 2, 1, 3) == 1


def test_random_over_sampler_matches_upstream():
    X, y = multiclass_data()
    ours = RandomOverSampler(random_state=5)
    upstream = UpstreamOver(random_state=5)
    assert_same_result(ours, upstream, X, y)
    assert np.array_equal(ours.sample_indices_, upstream.sample_indices_)


@pytest.mark.parametrize("shrinkage", [0.0, 0.35, {"small": 0.2, "middle": 0.4}])
def test_random_over_sampler_shrinkage_matches_upstream(shrinkage):
    X, y = multiclass_data()
    ours = RandomOverSampler(random_state=6, shrinkage=shrinkage)
    upstream = UpstreamOver(random_state=6, shrinkage=shrinkage)
    assert_same_result(ours, upstream, X, y)
    assert np.array_equal(ours.sample_indices_, upstream.sample_indices_)
    assert ours.shrinkage_ == upstream.shrinkage_


def test_random_over_sampler_object_data_matches_upstream():
    X = np.array([["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5]], dtype=object)
    y = np.array([0, 0, 1, 1, 1])
    assert_same_result(
        RandomOverSampler(random_state=2),
        UpstreamOver(random_state=2),
        X,
        y,
    )


def test_random_over_sampler_sparse_matches_upstream():
    X, y = multiclass_data()
    X = sparse.csr_matrix(X)
    assert_same_result(
        RandomOverSampler(random_state=2),
        UpstreamOver(random_state=2),
        X,
        y,
    )


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_random_over_sampler_shrinkage_does_not_narrow(dtype):
    X, y = multiclass_data(dtype)
    ours = RandomOverSampler(random_state=2, shrinkage=0.3)
    upstream = UpstreamOver(random_state=2, shrinkage=0.3)
    result, _ = assert_same_result(ours, upstream, X, y, atol=5e-15)
    assert result.dtype == np.float64


def test_random_over_sampler_sparse_shrinkage_matches_upstream():
    X, y = multiclass_data()
    X = sparse.csr_matrix(X)
    ours = RandomOverSampler(random_state=2, shrinkage=0.3)
    upstream = UpstreamOver(random_state=2, shrinkage=0.3)
    assert_same_result(ours, upstream, X, y, atol=5e-15)


def test_random_under_sampler_matches_upstream():
    X, y = multiclass_data()
    ours = RandomUnderSampler(random_state=8)
    upstream = UpstreamUnder(random_state=8)
    assert_same_result(ours, upstream, X, y)
    assert np.array_equal(ours.sample_indices_, upstream.sample_indices_)


def test_random_under_sampler_replacement_matches_upstream():
    X, y = multiclass_data()
    ours = RandomUnderSampler(
        sampling_strategy={"large": 13}, random_state=1, replacement=True
    )
    upstream = UpstreamUnder(
        sampling_strategy={"large": 13}, random_state=1, replacement=True
    )
    assert_same_result(ours, upstream, X, y)
    assert np.array_equal(ours.sample_indices_, upstream.sample_indices_)


def test_random_under_sampler_sparse_matches_upstream():
    X, y = multiclass_data()
    X = sparse.csc_matrix(X)
    assert_same_result(
        RandomUnderSampler(random_state=4),
        UpstreamUnder(random_state=4),
        X,
        y,
    )


def test_random_under_sampler_object_data_matches_upstream():
    X = np.array([["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5]], dtype=object)
    y = np.array([0, 0, 1, 1, 1])
    assert_same_result(
        RandomUnderSampler(random_state=2),
        UpstreamUnder(random_state=2),
        X,
        y,
    )


def test_callable_strategy_matches_upstream():
    X, y = multiclass_data()

    def strategy(target):
        return {"small": 13}

    assert_same_result(
        RandomOverSampler(sampling_strategy=strategy, random_state=2),
        UpstreamOver(sampling_strategy=strategy, random_state=2),
        X,
        y,
    )


def test_sklearn_get_params_and_clone():
    sampler = SMOTE(sampling_strategy="minority", random_state=9, k_neighbors=2)
    assert sampler.get_params() == {
        "k_neighbors": 2,
        "random_state": 9,
        "sampling_strategy": "minority",
    }
    assert sampler.set_params(k_neighbors=4) is sampler
    assert sampler.k_neighbors == 4
    cloned = clone(sampler)
    assert cloned.get_params() == sampler.get_params()


def test_fit_sets_attributes_and_returns_self():
    X, y = multiclass_data()
    sampler = RandomUnderSampler()
    assert sampler.fit(X, y) is sampler
    assert sampler.n_features_in_ == X.shape[1]
    assert sampler.sampling_strategy_ == {"large": 7, "middle": 7}


def test_invalid_single_class_target_is_rejected():
    X = np.ones((6, 2))
    with pytest.raises(ValueError, match="more than 1 class"):
        SMOTE().fit_resample(X, np.zeros(6))


def test_non_integer_dictionary_count_is_not_silently_truncated():
    X = np.ones((5, 2))
    y = np.array([0, 0, 1, 1, 1])
    with pytest.raises(TypeError, match="counts must be integers"):
        RandomOverSampler(sampling_strategy={0: 3.7}).fit_resample(X, y)


def test_under_sampler_replacement_must_be_boolean():
    X, y = multiclass_data()
    with pytest.raises(TypeError, match="must be a boolean"):
        RandomUnderSampler(replacement="yes").fit_resample(X, y)
