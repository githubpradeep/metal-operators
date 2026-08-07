// LDA (Linear Discriminant Analysis) kernels.
//
// The fit reuses the scatter (Gram) pattern from PCA/LinearRegression: the
// batch scatter kernel computes Xᵀ·X on the GPU (the only O(N·D²) step), while
// class statistics and the small (D×D) generalized eigenproblem are solved on
// the host. The transform kernel mirrors `pca_transform`.
//
// ── Kernel 1: Scatter — G = Xᵀ·X (D,D) ────────────────────────────
// X: (N, D) row-major. One thread per (i, j) output element.
kernel void lda_scatter(
    device const float* X        [[buffer(0)]],
    device float* G              [[buffer(1)]],
    constant uint& N             [[buffer(2)]],
    constant uint& D             [[buffer(3)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= D || j >= D) return;

    float acc = 0.0;
    for (uint k = 0; k < N; k++) {
        acc += X[k * D + i] * X[k * D + j];
    }
    G[i * D + j] = acc;
}

// ── Kernel 2: Transform — C = (X - mean) @ scalingsᵀ ─────────────
// X: (N, D), means: (D,), scalings: (K, D) row-major
// Output: C = (N, K)
kernel void lda_transform(
    device const float* X         [[buffer(0)]],
    device const float* means     [[buffer(1)]],
    device const float* scalings  [[buffer(2)]],
    device float* out             [[buffer(3)]],
    constant uint& N              [[buffer(4)]],
    constant uint& D              [[buffer(5)]],
    constant uint& K              [[buffer(6)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= K) return;

    float sum = 0.0;
    for (uint d = 0; d < D; d++) {
        sum += (X[row * D + d] - means[d]) * scalings[col * D + d];
    }
    out[row * K + col] = sum;
}