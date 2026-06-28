# Algorithm & Optimization Guide

This document explains the design and optimization rationale for the Metal-accelerated KMeans implementation.

## Table of Contents

1. [Lloyd's Algorithm Overview](#1-lloyds-algorithm-overview)
2. [Assign: Nearest-Center Lookup](#2-assign-nearest-center-lookup)
   - [2.1 Naive Kernel](#21-naive-kernel)
   - [2.2 Simdgroup Kernel (CTILE=8)](#22-simdgroup-kernel-ctile8)
   - [2.3 Simdgroup Kernel (CTILE=16)](#23-simdgroup-kernel-ctile16)
   - [2.4 Split-D Kernel](#24-split-d-kernel)
   - [2.5 Kernel Picker](#25-kernel-picker)
3. [Centroid Update](#3-centroid-update)
   - [3.1 CPU Baseline (the bottleneck)](#31-cpu-baseline-the-bottleneck)
   - [3.2 Naive GPU atomics (too contended)](#32-naive-gpu-atomics-too-contended)
   - [3.3 Per-cluster scatter+reduce (underutilized)](#33-per-cluster-scatterreduce-underutilized)
   - [3.4 Tiled centroid kernel (final)](#34-tiled-centroid-kernel-final)
   - [3.5 CPU fallback path](#35-cpu-fallback-path)
4. [Memory Access Analysis](#4-memory-access-analysis)
   - [4.1 Assign: coalescing patterns](#41-assign-coalescing-patterns)
   - [4.2 Centroid: coalescing patterns](#42-centroid-coalescing-patterns)
   - [4.3 Shared memory bank conflicts](#43-shared-memory-bank-conflicts)
5. [Performance Results](#5-performance-results)
   - [5.1 Assign kernel comparison](#51-assign-kernel-comparison)
   - [5.2 Centroid update comparison](#52-centroid-update-comparison)
   - [5.3 End-to-end fit](#53-end-to-end-fit)
6. [Apple GPU Bugs & Workarounds](#6-apple-gpu-bugs--workarounds)

---

## 1. Lloyd's Algorithm Overview

KMeans clustering iterates two steps until convergence:

1. **Assign**: For each point, find the nearest centroid (by Euclidean distance).
2. **Update**: Recompute each centroid as the mean of all points assigned to it.

```
for iter in 0..max_iterations:
    // Step 1: Assign
    for each point p_i:
        label[i] = argmin_c ||p_i - centroid_c||²

    // Step 2: Update
    for each cluster c:
        centroid_c = (1 / count_c) * sum_{label[i]=c} p_i

    if max_shift < tolerance: break
```

The cost profile for N=1M, D=32, K=16 on Apple M3:

| Step | Time | Bottleneck |
|---|---|---|
| Assign | ~1 ms | GPU compute (matmul bound) |
| Read labels | ~0.8 ms | GPU→CPU transfer |
| CPU centroid update | **~12 ms** | Memory bandwidth (CPU) |
| **Total** | **~14 ms** | |

The centroid update was the dominant cost — 86% of per-iteration time. The primary optimization goal was moving this step to the GPU.

## 2. Assign: Nearest-Center Lookup

### 2.1 Naive Kernel

**`kmeans_assign`** — the simplest kernel. Each thread handles one point, loops over all K centroids (device memory), computes Euclidean distance. Used when D < 8 or D % 8 != 0 (simdgroup restriction).

```
Threadgroup: 256 threads, ceil(N/256) groups
Per thread:  1 point × K centroids × D dimensions = K*D FLOPs
Memory:      centroid reads from device memory (no caching)
```

**When to use**: D < 8, D not multiple of 8, or K=0 (edge case).

### 2.2 Simdgroup Kernel (CTILE=8)

**`kmeans_assign_simdgroup`** — the primary workhorse. Uses Apple GPU's `simdgroup_float8x8` matrix-multiply primitives to compute 8×8 dot-product tiles.

**Key design:**

```
Threadgroup: 128 threads (= 4 simdgroups × 32 lanes)
Tile:        8 points × 8 centroids (PTILE=8, CTILE=8)
```

**Data flow per centroid tile:**

1. **Load points** into shared memory: `sh_pts[8 × D]`, row-major
2. **Load centroids** transposed: `sh_cent[D × 8]`, column `c` holds centroid `c`'s D values
3. **Compute dim tiles**: For each 8-dimensional chunk (DD = 0, 8, 16, ...):
   - Load `A[8×8]` = points (active dims) from `sh_pts`
   - Load `B[8×8]` = centroids (active dims) from `sh_cent`
   - `T = A × B` → `T[p][c]` = dot product over this dim chunk for point `p`, centroid `c`
   - Store `T` to `sh_dots`
4. **Reduce dot products** across dim tiles (threads sum the partial dots)
5. **Compute distance**: For each centroid `c`: `distance = ||p||² + ||c||² - 2·dot(p,c)`

**Why norms?** The identity `||p - c||² = ||p||² + ||c||² - 2·p·c` lets us precompute point and centroid norms, reducing per-centroid work from D multiplications + D additions to a single multiply-accumulate (`nx + nc - 2*dot`).

**Multi-simdgroup distribution:** All 4 simdgroups participate. Dim tiles are distributed round-robin: simdgroup 0 handles tiles 0, 4, 8, ...; simdgroup 1 handles tiles 1, 5, 9, ... After all simdgroups finish, the results sit in `sh_dots` (shared memory) for the reduction threads to read.

### 2.3 Simdgroup Kernel (CTILE=16)

**`kmeans_assign_simdgroup_c16`** — optimized variant for K ≤ 16 where all centroids fit in a single tile.

**Difference from CTILE=8:**

| Aspect | CTILE=8 | CTILE=16 |
|---|---|---|
| Centroid tile size | 8 centroids | 16 centroids |
| Centroid tiles per K=16 | 2 (c_base=0,8) | **1** (c_base=0) |
| Barriers per iteration | 2 (centroid load + matmul) | 2 (same) |
| Shared memory | 16·D + dim_tiles·64 + 16 floats | 24·D + dim_tiles·128 + 16 floats |

With CTILE=16, a single centroid tile covers all K=16 centroids. The matmul processes 2 centroid batches per dim tile (cents 0..7 and cents 8..15) to fit the 8-wide simdgroup output.

**Benefit**: Eliminates the outer centroid-tile loop for K ≤ 16, saving one centroid load and one best-dist update pass per iteration.

**When to use**: K ≤ 16 and shared memory fits: `(24·D + dim_tiles·128 + 16) × 4 ≤ 32,768`.

### 2.4 Split-D Kernel

**`kmeans_assign_splitd`** — for large D where simdgroup shared memory doesn't fit (or for non-multiple-of-8 D).

**Design:**

```
Threadgroup: 128 threads, handles 128 points (PTILE=128)
Outer loop:  centroids in CTILE=8 chunks
Inner loop:  D in BD=32 chunks
Shared:      BD × CTILE = 256 floats (minimal)
```

Each thread accumulates dot products in registers (`float dot_acc[8]`) across D chunks. After accumulating all D chunks, it computes distances and picks the best centroid. This keeps the K×D distance matrix implicit (never materialized).

**Works for any D** — no D % 8 requirement, no shared memory upper bound issue.

### 2.5 Kernel Picker

The `pick_assign_kernel` function selects the optimal kernel at runtime:

```
if D >= 8 and D % 8 == 0:
    if K <= 16 and CTILE=16 shared memory fits:
        → kmeans_assign_simdgroup_c16
    elif CTILE=8 shared memory fits:
        → kmeans_assign_simdgroup
    else:
        → kmeans_assign_splitd
else:
    → kmeans_assign (Naive)
```

Shared memory calculations:
- CTILE=16: `(24·D + dim_tiles·128 + 16) × 4` bytes
- CTILE=8:  `(16·D + dim_tiles·64 + 16) × 4` bytes
- Limit: 32,768 bytes (Apple GPU threadgroup memory limit)

## 3. Centroid Update

### 3.1 CPU Baseline (the bottleneck)

The original `compute_centroids` is a straightforward CPU loop:

```rust
for i in 0..n {
    let label = assignments[i];
    counts[label] += 1;
    for dim in 0..d {
        sums[label * d + dim] += data[i * d + dim];
    }
}
```

**Performance**: ~12 ms for N=1M, D=32, K=16.

**Why slow**: Each iteration reads one `u32` (from labels) and `D` floats (from data). For N=1M, D=32: 33M reads from main memory. CPU memory bandwidth (~40 GB/s on M3) is the bottleneck.

### 3.2 Naive GPU atomics (too contended) [dead code, reference only]

**`kmeans_centroid_accum`** (removed from shaders) — each thread handled 4 points (BATCH=4), atomically adding to device global sums.

```
Threads: N/4
Per thread: 4 points × D float atomics + 1 int atomic
Total atomics: N·D float + N int = 32M + 1M = 33M atomics
```

**Problem**: Contention. All N/K ≈ 62,500 threads targeted each of the K·D = 512 sum locations and K = 16 count locations. The GPU serialized these atomics at the memory controller. Result: **~600 ms** — 50× slower than CPU.

### 3.3 Per-cluster scatter+reduce (underutilized) [dead code, reference only]

**`kmeans_scatter` + `kmeans_centroid_sorted`** (removed from shaders) — two-pass approach inspired by flash-kmeans:

1. **Scatter**: Each point atomically writes its index to a per-cluster contiguous range in `sorted_ids`. Uses `atomic_fetch_add` on per-label counters. ~1M uint32 atomics on K bins.
2. **Per-cluster reduce**: One threadgroup per cluster. Each thread handles one dimension, loops over all points in the cluster, accumulates sums, writes final centroid.

**Problem**: Severe GPU underutilization. For K=16, only 16 threadgroups with 32 active threads each = 512 active threads. Apple M3 has 128 execution units capable of running 4096+ threads. The GPU is ~12% utilized. Result: **~38 ms warm** — still slower than CPU's 12ms.

### 3.4 Tiled centroid kernel (final)

**`kmeans_centroid_tiled`** — the solution that achieves full GPU utilization.

**Key insight**: Process points in batches across many threadgroups, NOT one threadgroup per cluster.

**Design:**

```
Threadgroup: 128 threads, handles 128 points (PTILE=128)
Groups:      ceil(N/128) → ~7,812 for N=1M
Shared:      K·D + K floats (max ~32 KB)
```

**Per-threadgroup data flow:**

1. **Zero shared memory**: Each thread zeros K entries for its assigned dimension. Thread 0 also zeros K count slots. One barrier.

2. **Process points** (loop, sequential): For each point in the batch:
   - Thread `dim` (unique, 0..D-1) reads `points[p·D + dim]`
   - Reads `label = assignments[p]`
   - Adds to `shared[label·D + dim]` (no conflict — each thread has a unique `dim`)
   - Thread 0 also increments `shared[K·D + label]` (counts — only thread 0 writes here)

3. **Barrier** (after all points processed)

4. **Flush to global**: For each label `c`:
   - Thread `dim` reads `shared[c·D + dim]`. If non-zero, CAS-loop atomic add to `centroid_sums[c·D + dim]` via `atomic_uint` (float bits reinterpretted through `as_type`).
   - Thread 0 reads `shared[K·D + c]`. If non-zero, `atomic_fetch_add` to `centroid_counts[c]`.

**Why no threadgroup atomics needed**: Each thread has a unique `dim = lid` (for `lid < D`). When multiple threads write to `shared[label·D + dim]`, the `dim` offset is different for each thread, so the addresses don't overlap. Thread 0 is the sole writer to `shared[K·D + label]` (counts). The outer `for i in 0..count` loop processes points sequentially, so within a single thread, writes to the same address happen in order (no race).

**Atomic contention analysis:**

| Metric | Old (per-point) | New (tiled) |
|---|---|---|
| Total device atomics | 33M (32M float + 1M int) | ≤ K·D·ceil(N/PTILE) = 512 × 7812 = ~4M |
| Contenders per location | N/K = 62,500 | ceil(N/PTILE) = 7,812 |
| Float atomic mechanism | `device atomic<float>` (removed in macOS 26) | `device atomic_uint` + CAS loop via `atomic_compare_exchange_weak` |
| Atomic traffic reduction | — | ~8× |

**Memory access pattern** (coalescing):

Within a simdgroup (32 threads) at iteration `i` of the point loop:
- Threads read `points[(p_start+i)·D + dim]` for `dim` = 0, 1, ..., 31
- These are **contiguous in memory** (same point row, consecutive dimensions) → one cache line fetch serves all 32 threads
- All threads read the same `assignments[p_start + i]` → broadcast from cache

**Shared memory limit**: When `(K·D + K)·4 > 32,768`, the kernel doesn't fit. This happens for large K·D products (e.g., K=256, D=32 → 33,792 bytes). In this case, the CPU fallback is used.

### 3.5 CPU fallback path

When shared memory requirements exceed 32 KB, `fit()` falls back to the CPU `compute_centroids` method:

```rust
if shared_needed <= 32_768 {
    // dispatch kmeans_centroid_tiled
} else {
    // read labels to CPU, compute centroids on CPU
    let assignments: Vec<u32> = ctx.read_buffer(&assign_buffer, n);
    let new_centroids = Self::compute_centroids(data, n, d, k, &assignments);
}
```

This preserves correctness for all shapes while providing GPU acceleration for the common case (K ≤ 64, D ≤ 128).

## 4. Memory Access Analysis

### 4.1 Assign: coalescing patterns

**Simdgroup kernel (CTILE=8/16):**

| Access | Pattern | Coalescing |
|---|---|---|
| Points → shared | Strided (po·D + pd), round-robin across threads | Poor per-thread, but all 4 simdgroups load the entire 8×D tile |
| Centroids → shared | Transposed layout, cooperative load | Sequential per thread (covers contiguous centroid data) |
| Simdgroup loads from shared | Regular 8×8 tiles | Perfect — all lanes access contiguous `sh_pts[p·D + dd·8 + lane]` |
| Norms (device) | Per-point | Random access (5th and 6th buffer) |

**Split-D kernel:**

| Access | Pattern | Coalescing |
|---|---|---|
| Centroids → shared | Chunked load: BD×CTILE tiles | Cooperative across 128 threads |
| Points → registers | Strided: `pid·D + dd + bd` | Threads access different PIDs → no coalescing within warp |

### 4.2 Centroid: coalescing patterns

**Tiled centroid kernel:**

| Access | Pattern | Coalescing |
|---|---|---|
| Points read | `points[(p_start+i)·D + dim]` for `dim=lid` across simdgroup | **Perfect** — adjacent `dim` values = contiguous cache line |
| Assignments read | All threads read same `assignments[p]` | Broadcast (all threads read same address) |
| Shared memory write | `shared[label·D + dim]` per thread | No conflict (unique `dim` per thread) |
| Global atomic write | `centroid_sums[c·D + dim]` | Random access per label, but scattered across 7,812 threadgroups |

### 4.3 Shared memory bank conflicts

Apple GPU shared memory has 32 banks, 4 bytes wide (128 bytes/cycle).

**Simdgroup kernel**: The `sh_dots` array is indexed as `[dd·64 + p·8 + c]` where `p = lid & 7`. Since adjacent threads have adjacent `lid` values (within a simdgroup), consecutive `p` values access consecutive `sh_dots` entries. With 4-byte floats and 32 banks, this maps to distinct banks — **no bank conflicts**.

**Tiled centroid kernel**: `shared[label·D + dim]` is accessed with varying `label` across threads in a warp. Since `label` is random (uniform over K), the stride `D` across threads results in random bank access. With K up to 256 and PTILE=128, each warp of 32 threads may hit 1-32 distinct labels, resulting in **partial bank conflicts** — at worst 32-way conflict if all 32 threads hit the same label (unlikely for K ≥ 32).

## 5. Performance Results

### 5.1 Assign kernel comparison

| Kernel | N | D | K | Time | Notes |
|---|---|---|---|---|---|
| Simdgroup (CTILE=8) | 1M | 32 | 16 | ~1.0 ms | Standard path for K > 16 |
| Simdgroup (CTILE=16) | 1M | 32 | 16 | ~0.7 ms | Single tile, fewer barriers |
| Split-D | 500 | 128 | 32 | ~0.3 ms | D too large for simdgroup |
| Naive | 1M | 2 | 8 | ~0.9 ms | D < 8 fallback |

The CTILE=16 variant is ~1.4× faster than CTILE=8 for K=16 by eliminating the outer centroid loop.

### 5.2 Centroid update comparison

| Method | N=1M, D=32, K=16 | Contention |
|---|---|---|
| CPU (M3) | ~12 ms | — |
| Per-point atomics (GPU) | ~600 ms | 62,500 contenders/location |
| Per-cluster sorted (GPU) | ~38 ms | 7,812 contenders, 512 threads active |
| **Tiled (GPU, final)** | **<1 ms warm** | 7,812 contenders, 7,812 threadgroups × 32 threads |
| Speedup vs CPU | **~12×** | |

### 5.3 End-to-end fit

Full Lloyd iteration (warm, N=1M, D=32, K=16):

| Component | Old (CPU centroid) | New (GPU centroid) | Improvement |
|---|---|---|---|
| Assign kernel | ~1 ms | ~1 ms | — |
| Read labels | ~0.8 ms | **0 ms** (no longer needed in loop) | ∞ |
| Centroid update | ~12 ms | **<1 ms** | 12× |
| Total | ~14 ms | **~2 ms** | 7× |

For 15 iterations: ~210 ms → ~30 ms (plus k-means++ overhead).

## 6. Apple GPU Bugs & Workarounds

### Bug 1: `simdgroup_load` with non-zero col/row

`simdgroup_load(A, base, stride, col, row)` should load a matrix where element `[i][j]` = `base[(row+i)·stride + (col+j)]`. On Apple GPUs, non-zero `col` or `row` produces garbage.

**Workaround**: Always use `col=0, row=0` and adjust the base pointer: `simdgroup_load(A, base + col + row·stride, stride, 0, 0)`.

### Bug 2: `simdgroup_multiply` replaces instead of accumulating

The Metal specification states `simdgroup_multiply(acc, A, B)` computes `acc = A·B`, replacing the previous value. Some GPU implementations (NVIDIA CUDA wmma) accumulate (`acc += A·B`). This implementation uses `simdgroup_multiply` as specified (replacing), storing each dim-tile result to a separate shared-memory slot and summing via threads.

### Bug 3: Row-first vs column-first addressing

The Metal specification describes `simdgroup_matrix` as column-first (`pointer[row + i][col + j]`), but the implementation uses row-first addressing (`pointer[(row+i)·stride + (col+j)]`). This implementation uses row-first (verified empirically on Apple M3).

### Bug 4: `device atomic<float>` removed in macOS 26

Starting with macOS 26, the Metal compiler no longer supports `device atomic<float>` — the `atomic_fetch_add_explicit` overload for `float` was removed. The compiler suggests `_atomic` (an internal type), but no public replacement exists.

**Workaround**: Use `device atomic_uint` instead and implement float addition via a CAS (compare-and-swap) loop:

```metal
device atomic_uint* target = &centroid_sums[idx];
uint expected = atomic_load_explicit(target, memory_order_relaxed);
bool done = false;
while (!done) {
    float cur = as_type<float>(expected);
    done = atomic_compare_exchange_weak_explicit(
        target, &expected, as_type<uint>(cur + val),
        memory_order_relaxed, memory_order_relaxed);
}
```

This is used in `kmeans_centroid_tiled`. The `atomic_int` overload for integer counts is unaffected.

### Non-bug: `threadgroup atomic<float>` unavailable

Metal has never supported `threadgroup atomic<float>` — the `atomic_fetch_add_explicit` overload for `float` only accepts `device` address space. The tiled centroid kernel works around this by assigning unique dimensions to threads, eliminating the need for threadgroup atomics.
