# metal-kmeans

GPU-accelerated KMeans clustering via Apple Metal. Uses 5 kernel variants (simdgroup, split-D, tiled centroid) to run Lloyd's algorithm entirely on GPU — no CPU readback inside the loop.

```python
from metal_kmeans import metal_kmeans, MetalKMeans
import numpy as np

data = np.random.randn(10000, 32).astype(np.float32)

# Functional API
labels, centroids, n_iter, inertia = metal_kmeans(
    data.ravel().tolist(), *data.shape, n_clusters=8,
    max_iterations=20, tolerance=1e-4, seed=42,
)

# sklearn-style API
km = MetalKMeans(n_clusters=8, max_iterations=20, tolerance=1e-4)
km.fit(data, *data.shape)
print(f"{km.inertia_:.2f}  {km.n_iter_} iters")
```

## Install

```sh
pip install maturin
git clone https://github.com/YOUR_USER/metal-operators
cd metal-operators
maturin develop
```

Metal shaders compile on first call (~20 ms/kernel); subsequent calls reuse cached pipelines.

## Example

```sh
python3 examples/example.py
```

Runs a 2D smoke test (1000 pts, 3 clusters) then a 100K×32 benchmark with k=64.

## Requirements

- macOS (Apple Silicon or AMD GPU with Metal support)
- Python 3.9+
- Rust toolchain (one-time `maturin develop` only)

No Xcode required — Metal shaders compile at runtime from inline source.

## API

### Functional

```python
metal_kmeans(data, n, d, n_clusters, max_iterations=100,
             tolerance=1e-4, seed=42) -> (labels, centroids, n_iter, inertia)
```

| Returns | Type | Shape |
|---|---|---|
| `labels` | `np.ndarray[intp]` | `(n,)` |
| `centroids` | `np.ndarray[float32]` | `(n_clusters, d)` |
| `n_iter` | `int` | — |
| `inertia` | `float` | — |

### sklearn-style

```python
km = MetalKMeans(n_clusters=8, max_iterations=20, tolerance=1e-4, seed=42)
km.fit(data, n, d)          # data: list[float] or np.ndarray[float32]
km.predict(new_data, n, d)  # assign to fitted centroids
km.cluster_centers_         # (n_clusters, d) float32
km.labels_                  # per-point labels from last fit
km.inertia_                 # within-cluster sum of squared distances
km.n_iter_                  # iterations used
```

`data` is flat row-major: `data[i * d + j]` = point `i`, dimension `j`.

## Rust API (advanced)

For direct Rust integration:

```rust
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

let ctx = MetalContext::new()?;
let mut km = KMeans::new(KMeansConfig { k: 8, max_iterations: 15, .. });
km.fit(&ctx, &data, n, d)?;
println!("inertia: {}", km.inertia());
```

## Tests

```sh
cargo test                     # Rust: 25 integration tests
python3 examples/example.py    # Python smoke test
```

Rust test matrix: D = {2, 4, 8, 16, 32, 64, 128}, K = {1, 8, 16, 32, 33, 64, 256}, including adjusted Rand index validation against CPU reference, multi-simdgroup correctness, split-D, empty-cluster handling, and timing.

## Benchmarks

```sh
cargo bench --bench kmeans_benchmark
cargo bench --bench kmeans_benchmark -- "fit.*1M_D=32"
```

Results (Apple M3):

### Assign (single iteration)

| N | D | K | Metal | BLAS-CPU | Speedup |
|---|---|---|---|---|---|
| 1M | 2 | 8 | 0.89 ms | 20.7 ms | **23×** |
| 100K | 2 | 8 | 0.30 ms | 2.04 ms | **6.7×** |
| 10K | 64 | 16 | 1.78 ms | 0.57 ms | 0.3× |
| 10K | 128 | 32 | 6.6 ms | 1.05 ms | 0.16× |

### Fit (15 iterations)

| N | D | K | Metal | BLAS-CPU | sklearn | vs BLAS |
|---|---|---|---|---|---|---|
| 100K | 32 | 256 | 32 ms | 82 ms | 33 ms | **2.6×** |
| 1M | 32 | 64 | 128 ms | 211 ms | 140 ms | **1.65×** |
| 1M | 32 | 16 | 36 ms | 108 ms | 99 ms | **3.0×** |
| 3M | 32 | 16 | 98 ms | 387 ms | — | **3.9×** |
| 200K | 64 | 512 | 306 ms | 348 ms | 118 ms | **1.1×** |
| 50K | 2 | 8 | 1.2 ms | 2.5 ms | 3.7 ms | **2.0×** |

The GPU centroid update (`kmeans_centroid_tiled`) replaces the CPU bottleneck (12 ms → <1 ms warm for 1M×32×16), yielding ~3× fit speedup. 2D shapes show the largest gains from assign (23×). High-D (128D) loses — 8×8 tile size doesn't fill the GPU's matrix units.

## Kernel dispatch

| Condition | Kernel | Threads/TG | Points/TG |
|---|---|---|---|
| D < 8 \|\| D % 8 != 0 | `kmeans_assign` (Naive) | 256 | 1 |
| K ≤ 16, shared memory fits | `kmeans_assign_simdgroup_c16` (CTILE=16) | 128 | 8 |
| Shared memory fits | `kmeans_assign_simdgroup` (CTILE=8) | 128 | 8 |
| D > 0, shared memory exceeded | `kmeans_assign_splitd` | 128 | 128 |

Centroid update: GPU (`kmeans_centroid_tiled`) when `(K×D+K)×4 ≤ 32 KB`, else CPU fallback.

## Project structure

```
src/
├── lib.rs               – crate root + PyO3 pymodule entry
├── python.rs             – PyO3 bindings (MetalKMeans class, functional API)
├── metal/mod.rs          – MetalContext: device, queue, buffer helpers
└── kmeans/mod.rs         – KMeans, assign kernel picker, centroid dispatch
python/metal_kmeans/
├── __init__.py           – flashlib-style Python API
└── _native.pyi           – type stubs
shaders/
└── kmeans.metal          – 5 Metal kernels (3 assign, 1 init, 1 centroid tiled)
examples/
└── example.py            – smoke test + benchmark
docs/
└── algorithm.md          – algorithm deep-dive
pyproject.toml            – maturin build config
```

## Notes

Apple GPU bugs worked around in `shaders/kmeans.metal`:

1. `simdgroup_load(base, stride, col, row)` produces garbage when `col ≠ 0` or `row ≠ 0`. Fix: adjust base pointer instead.
2. `simdgroup_multiply(acc, A, B)` **replaces** `acc` with `A*B` rather than accumulating. Fix: store each dim-tile to a separate shared-memory slot and sum via threads.
3. The Metal implementation uses row-first addressing (`pointer[(row+i)*stride + (col+j)]`) despite the spec describing column-first.
