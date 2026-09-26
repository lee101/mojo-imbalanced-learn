# mojo-imbalanced-learn

`mojo-imbalanced-learn` is a standalone port of the compute-heavy core of
[imbalanced-learn](https://imbalanced-learn.org/): SMOTE neighbor search and
sample generation, plus random over- and under-sampling. The numerical kernels
are written in Mojo and exposed through a Python API with the same estimator
names, constructor signatures, and `fit_resample` contract as the covered
upstream classes.

## Coverage

| namespace | implemented |
| --- | --- |
| `over_sampling` | `SMOTE`, `RandomOverSampler`, smoothed bootstrap through `shrinkage` |
| `under_sampling` | `RandomUnderSampler`, with or without replacement |
| estimator API | `fit`, `fit_resample`, `get_params`, `set_params`, cloning, `sampling_strategy_`, `n_features_in_`, `sample_indices_` where upstream provides it |
| strategies | `float`, string, dictionary, and callable strategies |
| data | dense float32/float64 arrays for SMOTE; dense numeric/object and SciPy sparse arrays for random samplers |

SMOTE preserves float32 and float64 output dtype. Its interpolation is computed
in float64, so float32 inputs are converted once and cast back. Other SMOTE
input dtypes are rejected instead of being silently narrowed at the native
boundary. Smoothed random over-sampling follows upstream and returns float64
for integer and lower-precision floating-point input.

Not covered are `SMOTENC`, `SMOTEN`, `BorderlineSMOTE`, `SVMSMOTE`,
`KMeansSMOTE`, `ADASYN`, cleaning methods such as Tomek links, sparse SMOTE,
custom nearest-neighbor estimator objects, and pandas container metadata.
These classes involve materially different algorithms rather than thin aliases,
so they are not presented as implemented.

The parity suite compares directly with imbalanced-learn 0.14.2. It checks
generated values and labels, seeded selection order, public fitted attributes,
dtypes, sparse and object inputs, strategy forms, smoothed bootstrap, and
scikit-learn estimator cloning.

## Install

```bash
pixi install
pixi run build
pixi run test
```

Pixi installs the pinned Mojo nightly, Python, NumPy, SciPy, scikit-learn, and
imbalanced-learn used by the parity tests. The build produces
`dist/libmojo-imbalanced-learn.so`.

## Usage

```python
import numpy as np
from mojo_imbalanced_learn.over_sampling import SMOTE

X = np.array([
    [0.0, 0.0],
    [0.2, 0.1],
    [0.1, 0.3],
    [0.3, 0.2],
    [2.0, 2.0],
    [2.2, 2.1],
    [1.9, 2.3],
    [2.1, 1.9],
    [2.3, 2.2],
    [1.8, 2.0],
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_balanced, y_balanced = SMOTE(
    random_state=0,
    k_neighbors=2,
).fit_resample(X, y)

print(X_balanced.shape)                  # (12, 2)
print(np.bincount(y_balanced).tolist())  # [6, 6]
```

Run it through Pixi so the package and compiled library are on the configured
paths:

```bash
pixi run python example.py
```

## Performance

Measured on 2026-07-29 on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64.
Each entry is the best of five end-to-end calls after warming both
implementations. The comparison is against imbalanced-learn 0.14.2 on identical
contiguous float64 arrays.

| case | Mojo port | imbalanced-learn | result |
| --- | ---: | ---: | ---: |
| SMOTE (10k x 20, 10% minority) | 17.37 ms | 57.07 ms | 3.29x faster |
| SMOTE (4k x 100, 15% minority) | 19.76 ms | 78.51 ms | 3.97x faster |
| RandomOverSampler (500k x 8) | 63.25 ms | 435.43 ms | 6.88x faster |
| RandomUnderSampler (500k x 8) | 25.32 ms | 138.06 ms | 5.45x faster |

Reproduce the table with:

```bash
pixi run bench
```

The Pixi task holds a machine-wide file lock so concurrent benchmark jobs do
not run at the same time.

No GPU path is provided.

## How it works

`src/capi.mojo` is one compilation unit containing three kernels: sorted
within-class k-nearest-neighbor search, SMOTE interpolation, and indexed row
gathering. Neighbor search and gathering use SIMD with scalar remainder loops.
Neighbor search is a compute-bound `n*n*d` sweep, so the Python shim fans it out
across a `ThreadPoolExecutor` above 8.4M fused multiply-adds and reaches about
2.8x at eight workers; below that it runs on one core, where threading only adds
synchronisation. Indexed row gathering moves `d` doubles per row and performs no
arithmetic worth splitting, so it always runs serially.
`build/build.sh` emits one shared library. The Python classes validate the
estimator contract, resolve sampling strategies, reproduce upstream
random-number sequencing, allocate outputs, and call the kernels once per
operation.

Dense kernel inputs are C-contiguous, row-major float64 arrays. Python owns all
input, output, and scratch memory. Buffers cross the C ABI as integer addresses;
the exported `@export(...)` functions rebuild
`UnsafePointer` values inside Mojo. Each synchronous call retains local
references to every NumPy buffer for the full call. The Python wrappers verify
dtype, rank, contiguity, shapes, and index ranges; Mojo checks non-null
addresses, dimensions, overflow, and indices before dereferencing. Native
status codes are converted to Python exceptions. Float64 data and contiguous
int64 indices cross the boundary without conversion or copies. No allocation
or Python object crosses the ABI.

Random samplers use the Mojo gather kernel for contiguous float64 input and
fall back to native NumPy or SciPy indexing when preserving object dtypes or a
sparse format requires it.

## License

MIT
