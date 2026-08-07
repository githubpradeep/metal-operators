// Metal Gaussian Naive Bayes kernels.
//
// Reduction + predict kernels (mirrors the two-kernel design of the rest of
// the operator family):
//
//   - `gnb_reduce_partials` — one reduction pass per block of training samples.
//     Each threadgroup (256 threads) owns GNB_PTILE = 128 samples; each thread
//     owns a disjoint set of feature dimensions [lid, lid+TG, ...) so shared
//     memory has no write conflicts (no threadgroup atomics). Shared memory
//     accumulates per-class sum and sum-of-squares over the block, then drains
//     into a per-threadgroup partial buffer (same scheme as kmeans_centroid).
//   - `gnb_reduce_sum` — deterministic fixed-order reduction of the per-group
//     partials into global per-class sum / sumsq / counts. The host then
//     computes mean[c][j] = sum/cnt, var[c][j] = sumsq/cnt - mean^2, class
//     priors = cnt/n, and applies a variance floor.
//   - `gnb_predict_logp` — per-sample log posterior per class; the host takes
//     the argmax for `predict` and the softmax for `predict_proba`.
//
// The reduction runs exactly once per fit (not iteratively), so one CPU-GPU
// sync covers fit; predict is a single launch.

#include <metal_stdlib>
using namespace metal;

constant uint GNB_PTILE = 128;                 // training samples per threadgroup
constant float GNB_LOG_2PI = 1.8378770664093453f; // ln(2*pi)

// ── Kernel 1: per-threadgroup partials ─────────────────────────────────────
// Grid: ceil(n / 128) threadgroups. Shared layout (kd = k*d):
//   shared[       c*d + j ]  ->  sum   for class c, feature j
//   shared[ kd +  c*d + j ]  ->  sumsq for class c, feature j
//   shared[ 2*kd + c      ]  ->  count for class c
// Host bails if (2*d + 1)*k floats exceed the threadgroup shared budget.
kernel void gnb_reduce_partials(
    device const float* data     [[buffer(0)]],  // (n, d) row-major
    device const uint*  cls      [[buffer(1)]],  // (n,) class ids in [0,k)
    device float*       sum_p    [[buffer(2)]],  // (G, k, d)
    device float*       sumsq_p  [[buffer(3)]],  // (G, k, d)
    device float*       count_p  [[buffer(4)]],  // (G, k)
    constant uint&      n        [[buffer(5)]],
    constant uint&      k        [[buffer(6)]],
    constant uint&      d        [[buffer(7)]],
    threadgroup float*  shared   [[threadgroup(0)]],
    uint gid                     [[threadgroup_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    uint base = gid * GNB_PTILE;
    if (base >= n) return;
    uint count = min(GNB_PTILE, n - base);
    uint kd = k * d;

    // Zero this thread's dimensions in shared (both planes).
    for (uint j = lid; j < d; j += tg_size) {
        for (uint c = 0; c < k; c++) {
            shared[c * d + j] = 0.0f;
            shared[kd + c * d + j] = 0.0f;
        }
    }
    if (lid == 0) {
        for (uint c = 0; c < k; c++) {
            shared[2 * kd + c] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate the block: each thread iterates its own dimensions.
    for (uint i = 0; i < count; i++) {
        uint p = base + i;
        uint c = cls[p];
        for (uint j = lid; j < d; j += tg_size) {
            float v = data[p * d + j];
            shared[c * d + j] += v;
            shared[kd + c * d + j] += v * v;
        }
        if (lid == 0) {
            shared[2 * kd + c] += 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Drain into the per-threadgroup partial buffers (row-major [c][j]).
    for (uint c = 0; c < k; c++) {
        for (uint j = lid; j < d; j += tg_size) {
            uint off = gid * kd + c * d + j;
            sum_p[off] = shared[c * d + j];
            sumsq_p[off] = shared[kd + c * d + j];
        }
    }
    if (lid == 0) {
        for (uint c = 0; c < k; c++) {
            count_p[gid * k + c] = shared[2 * kd + c];
        }
    }
}

// ── Kernel 2: deterministic combine of partials ──────────────────────────
// One element per thread over k*d sum entries, then k count entries. The
// pass has one threadgroup of 256 threads; stride loops cover the range.
kernel void gnb_reduce_sum(
    device const float* partial_sum   [[buffer(0)]],  // (G, k, d)
    device const float* partial_sumsq [[buffer(1)]],  // (G, k, d)
    device const float* partial_cnt   [[buffer(2)]],  // (G, k)
    device float*       sum           [[buffer(3)]],  // (k, d)
    device float*       sumsq         [[buffer(4)]],  // (k, d)
    device float*       cnt           [[buffer(5)]],  // (k,)
    constant uint&      g             [[buffer(6)]],
    constant uint&      k             [[buffer(7)]],
    constant uint&      d             [[buffer(8)]],
    uint lid                           [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint kd = k * d;

    // sum / sumsq entries.
    for (uint e = lid; e < kd; e += TG) {
        float acc = 0.0f;
        float acc2 = 0.0f;
        for (uint gg = 0; gg < g; gg++) {
            acc += partial_sum[gg * kd + e];
            acc2 += partial_sumsq[gg * kd + e];
        }
        sum[e] = acc;
        sumsq[e] = acc2;
    }
    // count entries.
    for (uint e = lid; e < k; e += TG) {
        float acc = 0.0f;
        for (uint gg = 0; gg < g; gg++) {
            acc += partial_cnt[gg * k + e];
        }
        cnt[e] = acc;
    }
}

// ── Kernel 3: predict log posteriors ─────────────────────────────────────
// One sample per thread; grid = ceil(n / 256). Output n*k log posteriors,
// row-major [sample][class].
kernel void gnb_predict_logp(
    device const float* data   [[buffer(0)]],  // (n, d) row-major
    device const float* mean   [[buffer(1)]],  // (k, d) per-class means
    device const float* var    [[buffer(2)]],  // (k, d) per-class variances (>0)
    device const float* logp   [[buffer(3)]],  // (k,) log priors
    device float*       out    [[buffer(4)]],  // (n, k) log posteriors
    constant uint&      n      [[buffer(5)]],
    constant uint&      k      [[buffer(6)]],
    constant uint&      d      [[buffer(7)]],
    uint gid                    [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint s = gid * TG + lid;
    if (s >= n) return;

    for (uint c = 0; c < k; c++) {
        float lp = logp[c];
        uint cbase = c * d;
        for (uint j = 0; j < d; j++) {
            float dx = data[s * d + j] - mean[cbase + j];
            float v = var[cbase + j];
            lp += -0.5f * (GNB_LOG_2PI + log(v)) - 0.5f * dx * dx / v;
        }
        out[s * k + c] = lp;
    }
}