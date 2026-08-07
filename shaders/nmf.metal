#include <metal_stdlib>
using namespace metal;

// NMF (Non-negative Matrix Factorization) kernels.
//
// We factor a non-negative data matrix V (N×D) into V ≈ W·H with W (N×K) and
// H (K×D), both non-negative, via the classic Lee–Seung multiplicative-update
// rules:
//
//   H ← H ⊙ (Wᵀ·V)      / (Wᵀ·W·H + ε)
//   W ← W ⊙ (V·Hᵀ)      / (W·H·Hᵀ + ε)
//
// Every O(N·D·K) step reduces to a matrix multiply, so the whole solver is
// built on a single tiled-by-rows `nmf_mm` kernel (with a transpose flag on
// either operand) plus one elementwise `nmf_update` kernel that applies the
// multiplicative rule in place.

// ── Kernel 1: general matrix multiply ───────────────────────────
// C (M×N) = A (M×K) · B (K×N).
// A_trans = 1: A is stored row-major as (N×K) and should be read transposed
//   (i.e. the logical operand is Aᵀ, contraction over M).
// B_trans = 1: B is stored row-major as (N×K) and should be read transposed
//   (i.e. the logical operand is Bᵀ).
//
// Row-major access (A in place):        A[r * K + s]
// A as stored transposed (N×K):         A[s * M + r]
// B as stored (K×N):                    B[s * N + c]
// B as stored transposed (N×K):         B[c * K + s]
kernel void nmf_mm(
    device const float* A              [[buffer(0)]],
    device const float* B              [[buffer(1)]],
    device float* C                    [[buffer(2)]],
    constant uint& M                   [[buffer(3)]],
    constant uint& N                   [[buffer(4)]],
    constant uint& K                   [[buffer(5)]],
    constant uint& A_trans             [[buffer(6)]],
    constant uint& B_trans             [[buffer(7)]],
    uint2 gid                          [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    if (r >= M || c >= N) return;

    float acc = 0.0;
    for (uint s = 0; s < K; s++) {
        float a = A_trans ? A[s * M + r] : A[r * K + s];
        float b = B_trans ? B[c * K + s] : B[s * N + c];
        acc += a * b;
    }
    C[r * N + c] = acc;
}

// ── Kernel 2: multiplicative update (in place) ──────────────────
// X (M×N) ← X ⊙ num / (den + eps), elementwise.
kernel void nmf_update(
    device float* X                    [[buffer(0)]],
    device const float* num            [[buffer(1)]],
    device const float* den            [[buffer(2)]],
    constant float& eps                [[buffer(3)]],
    constant uint& M                   [[buffer(4)]],
    constant uint& N                   [[buffer(5)]],
    uint2 gid                          [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    if (r >= M || c >= N) return;
    uint id = r * N + c;
    X[id] = X[id] * num[id] / (den[id] + eps);
}

// ── Kernel 3: CPU-facing reconstruction residuals ───────────────
// Computes E[id] = V[r*D+c] - W[r*K(k)] · H[k*D+c], i.e. the per-element
// reconstruction error V - W·H (used to cheaply compute the Frobenius
// reconstruction loss and to power the convergence test on the host).
kernel void nmf_diff(
    device const float* V              [[buffer(0)]],
    device const float* W              [[buffer(1)]],
    device const float* H              [[buffer(2)]],
    device float* E                    [[buffer(3)]],
    constant uint& N                   [[buffer(4)]],
    constant uint& D                   [[buffer(5)]],
    constant uint& K                   [[buffer(6)]],
    uint2 gid                          [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    if (r >= N || c >= D) return;
    float acc = 0.0;
    for (uint s = 0; s < K; s++) {
        acc += W[r * K + s] * H[s * D + c];
    }
    E[r * D + c] = V[r * D + c] - acc;
}