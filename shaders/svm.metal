// Metal Support Vector Classification (SVC) kernels.
//
// Mirrors the operator-family design: the O(n²·d) / O(n_sv·d) kernel
// evaluations run on the GPU, while the host performs the (per-iteration, O(n²))
// Sequential Minimal Optimization (SMO) dual solve:
//
//   - `svm_kernel` — computes the full n×n kernel (Gram) matrix in one launch:
//
//         K[i][j] = κ(x_i, x_j)
//
//     for a configurable kernel (linear / poly / RBF / sigmoid). One thread
//     owns one (i, j) element. Since the Gram depends only on the *data*
//     (not the labels), the same matrix is reused verbatim by every binary
//     one-vs-rest sub-classifier during fit — the GPU does this once.
//
//   - `svm_predict` — computes the raw decision matrix over an arbitrary test
//     set against the pooled support vectors:
//
//         dec[m][kk] = b_kk + Σ_{s ∈ SV_kk} α_s·y_s · κ(x_m, x_s)
//
//     one thread per (test sample, classifier) pair. The host maps this to
//     hard labels (argmax over the one-vs-rest classifiers, or sign for the
//     single binary classifier).
//
// The host drives training: SMO updates the dual coefficients (with the
// precomputed Gram serving as the fast 2nd-order κ lookup), then the resulting
// support vectors / dual coefficients / intercepts are written back and fed to
// `svm_predict` for scoring.

#include <metal_stdlib>
using namespace metal;

// Kernel type ids (must match the Rust `SVCKernel` enum).
constant uint SVC_LINEAR   = 0;
constant uint SVC_POLY     = 1;
constant uint SVC_RBF      = 2;
constant uint SVC_SIGMOID  = 3;

constant float SVC_LOG_2PI = 1.8378770664093453f; // ln(2*pi)

// ── Shared kernel evaluator ──────────────────────────────────────────────
// Returns κ(x, y) for the configured kernel:
//   linear   :  γ = 1, κ = <x,y>
//   poly     :  (γ<x,y> + coef0)^degree
//   rbf      :  exp(-γ ||x - y||²)
//   sigmoid  :  tanh(γ<x,y> + coef0)
inline float svm_kern(
    device const float* x,            // d floats
    device const float* y,            // d floats
    uint d,
    uint type,
    float gamma,
    float degree,
    float coef0
) {
    float dot = 0.0f;
    for (uint k = 0; k < d; k++) {
        dot += x[k] * y[k];
    }
    if (type == SVC_LINEAR) {
        return dot;
    }
    if (type == SVC_POLY) {
        float t = gamma * dot + coef0;
        float p = 1.0f;
        uint nd = (uint)max(1.0f, degree);
        for (uint e = 0; e < nd; e++) {
            p *= t;
        }
        return p;
    }
    if (type == SVC_RBF) {
        float sq = 0.0f;
        for (uint k = 0; k < d; k++) {
            float dx = x[k] - y[k];
            sq += dx * dx;
        }
        return exp(-gamma * sq);
    }
    // sigmoid
    return tanh(gamma * dot + coef0);
}

// ── Gram (kernel) matrix ─────────────────────────────────────────────────
// One thread per (i, j); grid = (n, n) tiled by 16×16 threads.
//   data : (n, d) row-major training data
//   out  : (n, n) row-major Gram matrix K[i*n + j] = κ(x_i, x_j)
kernel void svm_kernel(
    device const float* data    [[buffer(0)]],  // (n, d)
    device float*       out     [[buffer(1)]],  // (n, n)
    constant uint&      n       [[buffer(2)]],
    constant uint&      d       [[buffer(3)]],
    constant uint&      ktype   [[buffer(4)]],
    constant float&     gamma   [[buffer(5)]],
    constant float&     degree  [[buffer(6)]],
    constant float&     coef0   [[buffer(7)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= n || j >= n) return;
    out[i * n + j] = svm_kern(data + i * d, data + j * d, d, ktype, gamma, degree, coef0);
}

// ── Decision-function kernel ─────────────────────────────────────────────
// One thread per (test sample, classifier) pair; grid = ceil(M·K / 256).
//   X     : (M, d) test inputs
//   S     : (Ns, d) pooled support vectors (concatenated over classifiers)
//   dual  : (Ns) α_s·y_s aligned to the pooled support order
//   bias  : (K) per-classifier intercept
//   off   : (K+1) prefix offsets into S/dual for each classifier
//   dec   : (M, K) raw decision scores
//     dec[m][kk] = bias[kk] + Σ_{s ∈ [off[kk], off[kk+1])} dual[s]·κ(x_m, S_s)
kernel void svm_predict(
    device const float* X     [[buffer(0)]],  // (M, d)
    device const float* S     [[buffer(1)]],  // (Ns, d)
    device const float* dual  [[buffer(2)]],  // (Ns)
    device const float* bias  [[buffer(3)]],  // (K)
    device const uint*  off   [[buffer(4)]],  // (K+1)
    device float*       dec   [[buffer(5)]],  // (M, K)
    constant uint&      M     [[buffer(6)]],
    constant uint&      K     [[buffer(7)]],
    constant uint&      Ns    [[buffer(8)]],
    constant uint&      d     [[buffer(9)]],
    constant uint&      ktype [[buffer(10)]],
    constant float&     gamma [[buffer(11)]],
    constant float&     degree[[buffer(12)]],
    constant float&     coef0 [[buffer(13)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= M * K) return;
    uint m = gid / K;
    uint kk = gid % K;
    device const float* xm = X + m * d;
    float acc = bias[kk];
    uint start = off[kk];
    uint end = off[kk + 1];
    for (uint s = start; s < end; s++) {
        acc += dual[s] * svm_kern(xm, S + s * d, d, ktype, gamma, degree, coef0);
    }
    dec[m * K + kk] = acc;
}