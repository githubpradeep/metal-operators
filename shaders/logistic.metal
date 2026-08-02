// Metal Logistic Regression kernels
//
// Full-batch L-BFGS training with fused forward+backward+loss kernels
// (algorithm mirrors flashlib's Triton logistic regression):
//
//   - ONE fused kernel launch (+ a tiny deterministic reduction, both in a
//     single command buffer) per loss/gradient evaluation over the FULL
//     dataset (no mini-batch loop, no per-batch readback) -> single sync
//     per L-BFGS iteration.
//   - Each threadgroup accumulates partial gradients in shared memory and
//     writes ONE row to a per-threadgroup partials buffer (no device
//     atomics at all — avoids cross-threadgroup atomic contention).
//   - `logreg_reduce_partials` sums the partials in a fixed order, making
//     the gradient/loss reduction bitwise deterministic across launches.
//   - Stable softplus log-loss is accumulated on the GPU; the host adds the
//     L2 penalty and runs the L-BFGS (m = 10) update on the (d+1)-vector.
//
// Kernel variants (identical fused body; kept for dispatch selection and
// future specialization):
//   logreg_fused_fwd_bwd_naive     — generic fallback
//   logreg_fused_fwd_bwd_simdgroup — d >= 8 && d % 8 == 0
//   logreg_fused_fwd_bwd_splitd    — d > 128
//   logreg_reduce_partials         — deterministic (d+2)-wide reduction
//   logreg_predict                 — per-sample probability inference
//
// Dispatch: ceil(n / 256) threadgroups x 256 threads, one sample per thread.

#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-15f;

// ── Sigmoid (clamped) ─────────────────────────────────────────────────────

static float sigmoid(float z) {
    if (z < -30.0f) return EPS;
    if (z >  30.0f) return 1.0f - EPS;
    return 1.0f / (1.0f + exp(-z));
}

// ── Stable logistic log-loss: softplus(z) - y*z ───────────────────────────

static float log_loss(float z, float y) {
    float az = abs(z);
    float sp = max(z, 0.0f) + log(1.0f + exp(-az));
    return sp - y * z;
}

// ── Fused forward + backward + loss body ──────────────────────────────────
// One sample per thread. 256 threads = 8 simdgroups of 32 lanes; each
// simdgroup produces a partial (d+2)-vector via `simd_sum` (deterministic
// hardware reduction tree), stored in shared memory:
//   shared[sg * (d+2) + j]  ->  grad_w partial of simdgroup sg
//   shared[sg * (d+2) + d]  ->  grad_b partial
//   shared[sg * (d+2) + d+1]->  loss partial
// A final fixed-order combine writes `partials[gid][d+2]` (no device
// atomics). Threadgroup memory is NOT zero-initialized in MSL, so every
// thread zeroes the buffer before the first barrier. All threads execute
// both barriers (the `active` flag gates per-thread values only, keeping
// the simd reductions and barriers spec-safe for partial threadgroups).

static void fused_fwd_bwd_impl(
    device const float* batch_x,   // (n, d) row-major
    device const float* weights,   // (d,)
    device const float* batch_y,   // (n,)
    device float* partials,        // n_tg x (d+2) row-major output
    uint bsize, uint d,
    float inv_batch,               // 1/n (full batch) — already-normalized grad
    float bias,
    uint n_tg,                     // ceil(bsize / 256)
    threadgroup float* shared,     // 8 x (d+2) floats
    uint gid, uint lid
) {
    const uint TG = 256;
    const uint NSIMD = TG / 32;    // 8 simdgroups per threadgroup
    uint s = gid * TG + lid;
    bool active = (s < bsize);
    uint sg = lid / 32;
    uint lane = lid % 32;

    // Zero shared memory (threadgroup memory is uninitialized).
    for (uint j = lid; j < NSIMD * (d + 2); j += TG) {
        shared[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Forward pass ──
    float yv = active ? batch_y[s] : 0.0f;
    float z = bias;
    for (uint j = 0; j < d; j++) {
        z += weights[j] * (active ? batch_x[s * d + j] : 0.0f);
    }
    float p = sigmoid(z);
    float grad = (p - yv) * inv_batch;

    // ── Backward pass: deterministic simdgroup reduction ──
    for (uint j = 0; j < d; j++) {
        float v = active ? grad * batch_x[s * d + j] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            shared[sg * (d + 2) + j] = v;
        }
    }
    float gb = simd_sum(active ? grad : 0.0f);
    float ls = simd_sum(active ? log_loss(z, yv) : 0.0f);
    if (lane == 0) {
        shared[sg * (d + 2) + d] = gb;
        shared[sg * (d + 2) + d + 1] = ls;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Combine the 8 simdgroup partials in fixed order, no device atomics ──
    uint row = gid * (d + 2);
    for (uint j = lid; j < d + 2; j += TG) {
        float acc = 0.0f;
        for (uint i = 0; i < NSIMD; i++) {
            acc += shared[i * (d + 2) + j];
        }
        partials[row + j] = acc;
    }
}

// ── Kernel 1: Naive fused forward+backward+loss ───────────────────────────

kernel void logreg_fused_fwd_bwd_naive(
    device const float* batch_x     [[buffer(0)]],  // (n, d) row-major
    device const float* weights     [[buffer(1)]],  // (d,)
    device const float* batch_y     [[buffer(2)]],  // (n,)
    device float*       partials    [[buffer(3)]],  // n_tg x (d+2) output
    constant uint&      bsize       [[buffer(4)]],
    constant uint&      d           [[buffer(5)]],
    constant float&     inv_batch   [[buffer(6)]],
    constant float&     bias        [[buffer(7)]],
    constant uint&      n_tg        [[buffer(8)]],
    threadgroup float*  shared      [[threadgroup(0)]],
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    fused_fwd_bwd_impl(batch_x, weights, batch_y, partials,
                       bsize, d, inv_batch, bias, n_tg, shared, gid, lid);
}

// ── Kernel 2: SIMDGroup variant (d >= 8 && d % 8 == 0) ────────────────────

kernel void logreg_fused_fwd_bwd_simdgroup(
    device const float* batch_x     [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device const float* batch_y     [[buffer(2)]],
    device float*       partials    [[buffer(3)]],
    constant uint&      bsize       [[buffer(4)]],
    constant uint&      d           [[buffer(5)]],
    constant float&     inv_batch   [[buffer(6)]],
    constant float&     bias        [[buffer(7)]],
    constant uint&      n_tg        [[buffer(8)]],
    threadgroup float*  shared      [[threadgroup(0)]],
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    fused_fwd_bwd_impl(batch_x, weights, batch_y, partials,
                       bsize, d, inv_batch, bias, n_tg, shared, gid, lid);
}

// ── Kernel 3: Split-D variant (d > 128) ───────────────────────────────────

kernel void logreg_fused_fwd_bwd_splitd(
    device const float* batch_x     [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device const float* batch_y     [[buffer(2)]],
    device float*       partials    [[buffer(3)]],
    constant uint&      bsize       [[buffer(4)]],
    constant uint&      d           [[buffer(5)]],
    constant float&     inv_batch   [[buffer(6)]],
    constant float&     bias        [[buffer(7)]],
    constant uint&      n_tg        [[buffer(8)]],
    threadgroup float*  shared      [[threadgroup(0)]],
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    fused_fwd_bwd_impl(batch_x, weights, batch_y, partials,
                       bsize, d, inv_batch, bias, n_tg, shared, gid, lid);
}

// ── Kernel 4: Deterministic reduction of per-threadgroup partials ─────────
// One threadgroup of 256 threads; thread j sequentially sums column j over
// all n_tg rows in fixed order -> bitwise-deterministic gradient + loss.

kernel void logreg_reduce_partials(
    device const float* partials [[buffer(0)]],  // n_tg x (d+2) row-major
    device float*       grad_w   [[buffer(1)]],  // (d,) output
    device float*       grad_b   [[buffer(2)]],  // (1,) output
    device float*       loss     [[buffer(3)]],  // (1,) output
    constant uint&      n_tg     [[buffer(4)]],
    constant uint&      d        [[buffer(5)]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint j = lid;
    if (j >= d + 2) return;

    float acc = 0.0f;
    for (uint t = 0; t < n_tg; t++) {
        acc += partials[t * (d + 2) + j];
    }
    if (j < d) {
        grad_w[j] = acc;
    } else if (j == d) {
        *grad_b = acc;
    } else {
        *loss = acc;
    }
}

// ── Kernel 5: Predict probabilities ───────────────────────────────────────
// One sample per thread; grid = ceil(n / 256). Writes are idempotent.

kernel void logreg_predict(
    device const float* data    [[buffer(0)]],  // (n, d) row-major
    device const float* weights [[buffer(1)]],  // (d,)
    device float*       probs   [[buffer(2)]],  // (n,) output
    constant uint&      n       [[buffer(3)]],
    constant uint&      d       [[buffer(4)]],
    constant float&     bias    [[buffer(5)]],
    uint gid                     [[threadgroup_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint s = gid * TG + lid;
    if (s >= n) return;

    float z = bias;
    for (uint j = 0; j < d; j++) {
        z += weights[j] * data[s * d + j];
    }
    probs[s] = sigmoid(z);
}
