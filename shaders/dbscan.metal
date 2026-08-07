#include <metal_stdlib>
using namespace metal;

// DBSCAN ε-neighborhood kernels. These reuse the same squared-L2 distance
// approach as the KNN distance kernels (`knn.metal`) — direct Σ(x−y)² — but
// without the per-query `k` cap, so every neighbor of a point within `eps` is
// enumerated (exact DBSCAN connectivity).

// Pass A: count how many OTHER points lie within `eps` of each point.
kernel void dbscan_count(
    device const float* data      [[buffer(0)]],
    device uint* counts           [[buffer(1)]],
    constant uint& n              [[buffer(2)]],
    constant uint& d              [[buffer(3)]],
    constant float& eps_sq        [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const uint p = gid;
    uint c = 0;
    for (uint q = 0; q < n; q++) {
        if (q == p) continue;
        float acc = 0.0f;
        for (uint k = 0; k < d; k++) {
            float diff = data[p * d + k] - data[q * d + k];
            acc += diff * diff;
        }
        if (acc <= eps_sq) c++;
    }
    counts[p] = c;
}

// Pass B: write the compact neighbor lists (indexed by a CPU-side prefix sum
// over `counts`). Requires `offsets[p+1] - offsets[p] == counts[p]`.
kernel void dbscan_gather(
    device const float* data      [[buffer(0)]],
    device const uint* offsets    [[buffer(1)]],
    device uint* neighbors        [[buffer(2)]],
    constant uint& n              [[buffer(3)]],
    constant uint& d              [[buffer(4)]],
    constant float& eps_sq        [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const uint p = gid;
    uint slot = offsets[p];
    for (uint q = 0; q < n; q++) {
        if (q == p) continue;
        float acc = 0.0f;
        for (uint k = 0; k < d; k++) {
            float diff = data[p * d + k] - data[q * d + k];
            acc += diff * diff;
        }
        if (acc <= eps_sq) neighbors[slot++] = q;
    }
}