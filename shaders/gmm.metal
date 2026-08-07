// Metal Gaussian Mixture Model (GMM) kernels.
//
// Mirrors the operator-family design: the expensive per-sample loop runs on
// the GPU while the host performs the (small, per-component) parameter update.
//
//   - `gmm_e_step` — the EM expectation step. One thread owns one sample and
//     writes that sample's log-likelihood under every mixture component:
//
//         L[i][c] = log w_c - 0.5 ( d log(2π) + log|Σ_c| + (x_i-μ_c)ᵀ P_c (x_i-μ_c) )
//
//     where P_c = Σ_c⁻¹ (precision). The (n × k) log-likelihood matrix is the
//     only quantity that depends on every pair (sample, component), so pushing
//     it to the GPU removes the O(n·k·d²) bottleneck from the host.
//
// The host drives the EM loop: E-step (this kernel) → log-sum-exp/responsibi-
// lities → M-step (weighted mean/covariance update, per-component Cholesky to
// rebuild precision + logdet) → repeat until the lower bound converges.

#include <metal_stdlib>
using namespace metal;

constant float GMM_LOG_2PI = 1.8378770664093453f; // ln(2*pi)

// ── Kernel: EM expectation step ──────────────────────────────────────────
// One sample per thread; grid = ceil(n / 256).
//   data  : (n, d) row-major input
//   means : (k, d) row-major component means
//   prec  : (k, d, d) row-major per-component precision (= Σ_c⁻¹, symmetric)
//   logdet: (k,) log |Σ_c| for each component
//   logw  : (k,) log mixture weight
//   out   : (n, k) row-major log-likelihoods
kernel void gmm_e_step(
    device const float* data    [[buffer(0)]],  // (n, d)
    device const float* means   [[buffer(1)]],  // (k, d)
    device const float* prec    [[buffer(2)]],  // (k, d, d)
    device const float* logdet  [[buffer(3)]],  // (k,)
    device const float* logw    [[buffer(4)]],  // (k,)
    device float*       out     [[buffer(5)]],  // (n, k)
    constant uint&      n       [[buffer(6)]],
    constant uint&      k       [[buffer(7)]],
    constant uint&      d       [[buffer(8)]],
    uint gid                     [[threadgroup_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint s = gid * TG + lid;
    if (s >= n) return;
    const uint dd = d * d;

    for (uint c = 0; c < k; c++) {
        const uint cbase = c * d;
        const uint pbase = c * dd;
        float ll = logw[c] - 0.5f * (d * GMM_LOG_2PI + logdet[c]);

        // Quadratic form (x - μ)ᵀ Σ⁻¹ (x - μ) = Σ_a dx_a * (Σ⁻¹·dx)_a.
        float q = 0.0f;
        for (uint a = 0; a < d; a++) {
            float dxa = data[s * d + a] - means[cbase + a];
            float sdot = 0.0f;
            const uint pa = pbase + a * d;
            for (uint b = 0; b < d; b++) {
                sdot += prec[pa + b] * (data[s * d + b] - means[cbase + b]);
            }
            q += dxa * sdot;
        }

        out[s * k + c] = ll - 0.5f * q;
    }
}