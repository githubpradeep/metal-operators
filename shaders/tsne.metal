#include <metal_stdlib>
using namespace metal;

// t-Distributed Stochastic Neighbor Embedding (t-SNE) kernels.
//
// Exact (O(N²)) t-SNE has two dominant costs, both of which are shipped here:
//   1. Computing the pairwise Gaussian affinities P (N×N) from the
//      high-dimensional input — a full squared-L2 distance matrix followed by
//      a per-row (independent) perplexity bisection. This is the O(N²·D) one-
//      time cost.
//   2. The t-SNE gradient each iteration — an O(N²) cost dominated by pairwise
//      terms (P_ij - Q_ij)·(y_i - y_j)·q_ij over the low-dimensional embedding.
// Every thread i owns an output row, so there are no intra-kernel races; the
// (÷ Z) Q-normalization is resolved by a per-thread Z accumulator read back to
// the host.

// Nex=exaggeration-aware squared distances for the input data: D_ij = ‖x_i−x_j‖².
kernel void tsne_distances(
    device const float* X      [[buffer(0)]],
    device float* D            [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    constant uint& d           [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n * n) return;
    const uint i = gid / n;
    const uint j = gid % n;
    float acc = 0.0f;
    for (uint k = 0; k < d; k++) {
        float diff = X[i * d + k] - X[j * d + k];
        acc += diff * diff;
    }
    D[i * n + j] = acc;
}

constant uint TSNE_SEARCH_ITERS = 50;
constant float TSNE_LOG_SIGMA_LO = -15.0f;
constant float TSNE_LOG_SIGMA_HI = 15.0f;

// Per-point perplexity bisection over log(σ). Thread i finds the σ_i whose
// conditional Gaussian entropy matches log(perplexity), then writes the
// normalized conditional row P_{j|i} (diagonal pinned to 0).
kernel void tsne_perplexity(
    device const float* D          [[buffer(0)]],
    device float* P               [[buffer(1)]],
    constant uint& n              [[buffer(2)]],
    constant float& perplexity    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const uint i = gid;
    const float target = log(perplexity);
    float lo = TSNE_LOG_SIGMA_LO;
    float hi = TSNE_LOG_SIGMA_HI;
    for (uint it = 0; it < TSNE_SEARCH_ITERS; it++) {
        float mid = 0.5f * (lo + hi);
        float sigma = exp(mid);
        float inv2s2 = 1.0f / (2.0f * sigma * sigma);
        float s = 0.0f;
        for (uint j = 0; j < n; j++) {
            if (j == i) continue;
            s += exp(-D[i * n + j] * inv2s2);
        }
        if (s <= 0.0f) { lo = mid; continue; }   // all underflowed -> need bigger σ
        float h = 0.0f;
        for (uint j = 0; j < n; j++) {
            if (j == i) continue;
            float p = exp(-D[i * n + j] * inv2s2) / s;
            h -= p * log(p);
        }
        if (h > target) { hi = mid; } else { lo = mid; }
    }
    float sigma = exp(0.5f * (lo + hi));
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    float s = 0.0f;
    for (uint j = 0; j < n; j++) {
        if (j == i) continue;
        s += exp(-D[i * n + j] * inv2s2);
    }
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (uint j = 0; j < n; j++) {
        if (j == i) { P[i * n + j] = 0.0f; continue; }
        P[i * n + j] = exp(-D[i * n + j] * inv2s2) * inv_s;
    }
}

// Per-iteration t-SNE gradient. Thread i computes, for its point:
//     Z_i = Σ_{j≠i} q_ij                    (first-order affinity sum)
//     A1 = Σ_{j≠i} p_ij·q_ij·(y_i − y_j)    (attraction, P-scaled)
//     A2 = Σ_{j≠i} q_ij²·(y_i − y_j)        (repulsion; /Z by the host)
// The host forms grad_i = 4·(A1_i − A2_i/Z) and applies the momentum update.
// A1/A2 must be zeroed by the host before each dispatch (they accumulate).
kernel void tsne_grad(
    device const float* Y      [[buffer(0)]],   // n × nc
    device const float* P      [[buffer(1)]],   // n × n
    device float* A1           [[buffer(2)]],   // n × nc
    device float* A2           [[buffer(3)]],   // n × nc
    device float* Zbuf         [[buffer(4)]],   // n
    constant uint& n           [[buffer(5)]],
    constant uint& nc          [[buffer(6)]],
    constant float& exag       [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const uint i = gid;
    float z = 0.0f;
    for (uint j = 0; j < n; j++) {
        if (j == i) continue;
        float d2 = 0.0f;
        for (uint c = 0; c < nc; c++) {
            float dc = Y[i * nc + c] - Y[j * nc + c];
            d2 += dc * dc;
        }
        float q = 1.0f / (1.0f + d2);
        float pq = P[i * n + j] * exag * q;
        float q2 = q * q;
        for (uint c = 0; c < nc; c++) {
            float dc = Y[i * nc + c] - Y[j * nc + c];
            A1[i * nc + c] += pq * dc;
            A2[i * nc + c] += q2 * dc;
        }
        z += q;
    }
    Zbuf[i] = z;
}