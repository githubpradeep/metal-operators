# metal-operators

Classical ML operators accelerated with [Apple Metal](https://developer.apple.com/metal/) GPU compute, implemented in Rust via the [`metal`](https://crates.io/crates/metal) crate.

## What's here

- **KMeans fit & predict** — GPU-accelerated assignment kernels + multi-iteration Lloyd's algorithm
  - **Naive** — per-point loop over centroids in device memory (D < 8 or D % 8 != 0)
  - **Tiled** — cooperatively loads centroid tiles into threadgroup memory
  - **Simdgroup** — 8×8 dot-product tiles via `simdgroup_load`/`multiply`/`store` (D % 8 == 0, shared memory ≤ 32 KB)
- **k-means++** initialization on GPU
- **CPU reference** — BLAS-accelerated via Accelerate `sgemm_` for benchmarking
- **sklearn comparison** — times `KMeans` via Python subprocess for validation

## Requirements

- macOS (Apple Silicon or AMD GPU with Metal support)
- Rust 2024 edition
- Python 3 + scikit-learn (for benchmarks)

No Xcode required — Metal shaders are compiled at runtime from inline source (`shaders/kmeans.metal`).

## Usage

```rust
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

let ctx = MetalContext::new()?;

let data: Vec<f32> = /* n × d row-major points */;
let n = 100_000;
let d = 32;
let k = 64;

let mut km = KMeans::new(KMeansConfig {
    k,
    max_iterations: 15,
    tolerance: 1e-4,
    seed: 42,
    init_centroids: None, // or Some(centroids)
});

km.fit(&ctx, &data, n, d)?;
println!("inertia: {}", km.inertia());
println!("iterations: {}", km.n_iter());

let labels = km.predict(&ctx, &data, n, d)?;
```

## Tests

```sh
cargo test
```

19 tests covering D = {8, 16, 32, 64, 128}, K = {1, 8, 16, 32, 33, 64, 256}, including correctness validation against a pure-CPU reference via adjusted Rand index.

## Benchmarks

```sh
cargo bench --bench kmeans_benchmark
# filter to specific shapes:
cargo bench --bench kmeans_benchmark -- "fit.*1M_D=32"
```

Assign benchmarks compare raw GPU kernel time against BLAS pairwise distances.
Fit benchmarks run full Lloyd iterations (Metal vs BLAS-CPU vs sklearn).

### Results (Apple M3 MacBook)

**Assign (single iteration)**

| N | D | K | Metal | BLAS-CPU | Speedup |
|---|---|---|---|---|---|
| 1M | 2 | 8 | 0.89 ms | 20.7 ms | **23×** |
| 100K | 2 | 8 | 0.30 ms | 2.04 ms | **6.7×** |
| 10K | 64 | 16 | 1.78 ms | 0.57 ms | 0.3× |
| 10K | 128 | 32 | 6.6 ms | 1.05 ms | 0.16× |

**Fit (15 iterations default)**

| N | D | K | Metal | BLAS-CPU | sklearn | vs BLAS |
|---|---|---|---|---|---|---|
| 100K | 32 | 256 | 32 ms | 82 ms | 33 ms | **2.6×** |
| 1M | 32 | 64 | 128 ms | 211 ms | 140 ms | **1.65×** |
| 1M | 32 | 16 | 101 ms | 108 ms | 99 ms | **1.1×** |
| 3M | 32 | 16 | 261 ms | 387 ms | — | **1.5×** |
| 200K | 64 | 512 | 306 ms | 348 ms | 118 ms | **1.1×** |
| 50K | 2 | 8 | 1.2 ms | 2.5 ms | 3.7 ms | **2.0×** |

Low-dimensional (2D) shapes show the largest gains (up to 23×). High-dimensional (128D) loses — the simdgroup 8×8 tile size doesn't effectively engage the GPU's matrix units.

## Architecture

```
src/
├── lib.rs               – crate root
├── metal/mod.rs          – MetalContext: device, queue, buffer helpers
└── kmeans/mod.rs         – KMeans struct, assign kernel dispatch, centroid update
shaders/
└── kmeans.metal          – Metal kernels (assign, simdgroup assign, min-distances)
benches/
├── kmeans_benchmark.rs   – Criterion benchmarks vs BLAS & sklearn
└── sklearn_kmeans.py     – Python subprocess for sklearn timing
tests/                    – 19 integration tests
reference/
└── flashlib/             – Python Triton/CuteDSL reference (subtree, not tracked)
```

## Notes

The simdgroup kernel works around two Apple GPU bugs discovered during development:

1. `simdgroup_load(base, stride, col, row)` produces incorrect results when `col != 0` or `row != 0`. Workaround: adjust the base pointer instead (`base + col`).
2. `simdgroup_multiply(acc, A, B)` **replaces** `acc` with `A*B` on this GPU rather than accumulating (`acc += A*B`). Workaround: store each dim-tile to a separate shared-memory slot and sum via threads.
3. The Metal implementation uses row-first addressing (`pointer[(row+i)*stride + (col+j)]`) even though the specification describes column-first.
