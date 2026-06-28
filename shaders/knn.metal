#include <metal_stdlib>
using namespace metal;

// ── helpers ────────────────────────────────────────────────────

void heap_insert(thread float* dists, thread uint* idxs, uint k, float d, uint idx) {
    if (d >= dists[k - 1]) return;
    uint pos = k - 1;
    while (pos > 0 && d < dists[pos - 1]) {
        dists[pos] = dists[pos - 1];
        idxs[pos]  = idxs[pos - 1];
        pos--;
    }
    dists[pos] = d;
    idxs[pos]  = idx;
}

// ── Dense KNN assign (direct device reads, small D) ────────────
// Each threadgroup: 128 threads, each processes one query.
// Grid: (ceil(nq / 128), 1, 1).  No M-split, no shared memory.
// Query fits in registers.  Used for D < 32.
kernel void knn_assign_dense(
    device const float* queries        [[buffer(0)]],
    device const float* corpus          [[buffer(1)]],
    device float* out_scores            [[buffer(2)]],
    device uint* out_indices            [[buffer(3)]],
    device const float* norms_Q         [[buffer(4)]],
    device const float* norms_C         [[buffer(5)]],
    constant uint& nq                   [[buffer(6)]],
    constant uint& nc                   [[buffer(7)]],
    constant uint& d                    [[buffer(8)]],
    constant uint& k                    [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint qid = gid.x * 128 + lid;
    if (qid >= nq) return;

    float my_q[32];
    for (uint dd = 0; dd < d; dd++) {
        my_q[dd] = queries[qid * d + dd];
    }

    float hd[64];
    uint  hi[64];
    for (uint j = 0; j < k; j++) { hd[j] = INFINITY; hi[j] = 0; }

    float dot, score;
    uint pos;
    for (uint c = 0; c < nc; c++) {
        dot = 0.0;
        for (uint dd = 0; dd < d; dd++) {
            dot += my_q[dd] * corpus[c * d + dd];
        }
        score = norms_C[c] - 2.0f * dot;
        if (score < hd[k - 1]) {
            pos = k - 1;
            while (pos > 0 && score < hd[pos - 1]) {
                hd[pos] = hd[pos - 1];
                hi[pos] = hi[pos - 1];
                pos--;
            }
            hd[pos] = score;
            hi[pos] = c;
        }
    }

    uint ob = qid * k;
    for (uint j = 0; j < k; j++) {
        out_scores[ob + j]  = hd[j];
        out_indices[ob + j] = hi[j];
    }
}

// ── Naive KNN assign (M-split aware) ───────────────────────────
// Grid: (ceil(nq / TG) * num_splits) flattened into 1D.
// gid = split * nq_blocks + q_block.
kernel void knn_assign_naive(
    device const float* queries        [[buffer(0)]],
    device const float* corpus          [[buffer(1)]],
    device float* split_scores          [[buffer(2)]],
    device uint* split_indices          [[buffer(3)]],
    device const float* norms_Q         [[buffer(4)]],
    device const float* norms_C         [[buffer(5)]],
    constant uint& nq                   [[buffer(6)]],
    constant uint& nc                   [[buffer(7)]],
    constant uint& d                    [[buffer(8)]],
    constant uint& k                    [[buffer(9)]],
    constant uint& num_splits           [[buffer(10)]],
    constant uint& m_per_split          [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint nq_blocks = (nq + 255) / 256;
    uint split = gid / nq_blocks;
    uint qid   = (gid % nq_blocks) * 256 + lid;
    if (split >= num_splits || qid >= nq) return;

    uint m_start = split * m_per_split;
    uint m_count = min(m_per_split, nc - m_start);

    float heap_d[64];
    uint  heap_i[64];
    uint  hk = min(k, (uint)64);
    for (uint j = 0; j < hk; j++) { heap_d[j] = INFINITY; heap_i[j] = 0; }

    uint end = m_start + m_count;
    for (uint c = m_start; c < end; c++) {
        float dot = 0.0;
        for (uint dim = 0; dim < d; dim++) {
            dot += queries[qid * d + dim] * corpus[c * d + dim];
        }
        // shift-invariant score (preserves ordering vs true L2)
        float score = norms_C[c] - 2.0f * dot;
        heap_insert(heap_d, heap_i, hk, score, c);
    }

    uint ob = split * nq * k + qid * k;
    for (uint j = 0; j < hk; j++) {
        split_scores[ob + j]  = heap_d[j];
        split_indices[ob + j] = heap_i[j];
    }
}

// ── M-Split KNN assign (SIMDGroup, BN=16) ─────────────────────
// Each threadgroup: 16 query points × one corpus split.
// Grid: (num_splits, ceil(nq / 16)).  128 threads.
// Uses shift-invariant scoring (ordering identical to true L2).
// Output: K (score, global_idx) pairs per query into split arrays.
// Requires D % 8 == 0.
kernel void knn_assign_splitm(
    device const float* queries        [[buffer(0)]],
    device const float* corpus          [[buffer(1)]],
    device float* split_scores          [[buffer(2)]],
    device uint* split_indices          [[buffer(3)]],
    device const float* norms_Q         [[buffer(4)]],
    device const float* norms_C         [[buffer(5)]],
    constant uint& nq                   [[buffer(6)]],
    constant uint& nc                   [[buffer(7)]],
    constant uint& d                    [[buffer(8)]],
    constant uint& k                    [[buffer(9)]],
    constant uint& num_splits           [[buffer(10)]],
    constant uint& m_per_split          [[buffer(11)]],
    threadgroup float* shared           [[threadgroup(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    uint split = gid.x;
    uint qb    = gid.y;
    uint q_off = qb * 16;

    // Compute this split's corpus range
    uint m_start = split * m_per_split;
    uint m_count = min(m_per_split, nc - m_start);

    constexpr uint BN = 16;
    constexpr uint BM = 8;
    uint num_tiles = (d + 7) / 8;

    threadgroup float* sh_q    = shared;
    threadgroup float* sh_c    = shared + BN * d;
    threadgroup float* sh_dots = shared + BN * d + d * BM;
    threadgroup float* sh_heap_d = sh_dots + num_tiles * BN * BM;
    threadgroup uint*  sh_heap_i = (threadgroup uint*)(sh_heap_d + BN * k);

    // init heap
    for (uint i = lid; i < BN * k; i += 128) {
        sh_heap_d[i] = INFINITY;
        sh_heap_i[i] = 0;
    }

    // load query points
    {
        uint total = BN * d;
        for (uint i = lid; i < total; i += 128) {
            uint po = i / d, pd = i % d;
            uint gi = q_off + po;
            sh_q[po * d + pd] = (gi < nq) ? queries[gi * d + pd] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // corpus loop over this split's range
    uint m_end = m_start + m_count;
    for (uint c_base = m_start; c_base < m_end; c_base += BM) {
        uint c_tile = min(BM, m_end - c_base);

        // load corpus tile (transposed)
        {
            uint load = c_tile * d;
            for (uint i = lid; i < load; i += 128) {
                uint co = i % c_tile, dim = i / c_tile;
                sh_c[dim * BM + co] = corpus[(c_base + co) * d + dim];
            }
            uint zero = d * (BM - c_tile);
            for (uint i = lid; i < zero; i += 128) {
                uint dd = i / (BM - c_tile), co = c_tile + i % (BM - c_tile);
                sh_c[dd * BM + co] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // simdgroup matmul: BN=16 > PTILE=8, so run two batches
        for (uint dd = simd_gid; dd < num_tiles; dd += 4) {
            simdgroup_float8x8 A, B, t = {};
            uint tile_off = dd * BN * BM;
            // batch 1: queries 0-7
            simdgroup_load(A, sh_q + dd * BM,        d,  0, 0);
            simdgroup_load(B, sh_c + dd * BM * BM,   BM, 0, 0);
            simdgroup_multiply(t, A, B);
            simdgroup_store(t, sh_dots + tile_off, BM, 0, 0);
            // batch 2: queries 8-15
            simdgroup_load(A, sh_q + dd * BM + 8 * d, d, 0, 0);
            simdgroup_load(B, sh_c + dd * BM * BM,     BM, 0, 0);
            simdgroup_multiply(t, A, B);
            simdgroup_store(t, sh_dots + tile_off + 8 * BM, BM, 0, 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // heap update (threads 0..BN-1)
        if (lid < BN && q_off + lid < nq) {
            uint pid = lid;
            float nx = norms_Q[q_off + pid];
            threadgroup float* hd = sh_heap_d + pid * k;
            threadgroup uint*  hi = sh_heap_i  + pid * k;

            for (uint c = 0; c < c_tile; c++) {
                float dot = 0.0f;
                for (uint dd = 0; dd < num_tiles; dd++) {
                    dot += sh_dots[dd * BN * BM + pid * BM + c];
                }
                // shift-invariant score (preserves ordering vs true L2)
                float score = norms_C[c_base + c] - 2.0f * dot;
                if (score < hd[k - 1]) {
                    uint pos = k - 1;
                    while (pos > 0 && score < hd[pos - 1]) {
                        hd[pos] = hd[pos - 1];
                        hi[pos] = hi[pos - 1];
                        pos--;
                    }
                    hd[pos] = score;
                    hi[pos] = c_base + c;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // write results
    if (lid < BN && q_off + lid < nq) {
        uint qid = q_off + lid;
        uint ob  = split * nq * k + qid * k;
        threadgroup float* hd = sh_heap_d + lid * k;
        threadgroup uint*  hi = sh_heap_i  + lid * k;
        for (uint j = 0; j < k; j++) {
            split_scores[ob + j]  = hd[j];
            split_indices[ob + j] = hi[j];
        }
    }
}

// ── Gather true squared-L2 distances ──────────────────────────
// Recomputed via direct subtraction to avoid expanded-formula cancellation.
kernel void knn_gather_l2(
    device const float* queries        [[buffer(0)]],
    device const float* corpus          [[buffer(1)]],
    device const uint* neighbor_idx     [[buffer(2)]],
    device float* out_dists             [[buffer(3)]],
    constant uint& nq                   [[buffer(4)]],
    constant uint& nc                   [[buffer(5)]],
    constant uint& d                    [[buffer(6)]],
    constant uint& k                    [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint qid = gid.y;
    uint kj  = gid.x;
    if (qid >= nq || kj >= k) return;

    uint cid = neighbor_idx[qid * k + kj];
    float dist = 0.0;
    // each thread handles one dim
    for (uint dim = lid; dim < d; dim += 128) {
        float diff = queries[qid * d + dim] - corpus[cid * d + dim];
        dist += diff * diff;
    }
    // threadgroup reduction
    threadgroup float tmp[128];
    tmp[lid] = dist;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 64) tmp[lid] += tmp[lid + 64];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) tmp[lid] += tmp[lid + 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16) tmp[lid] += tmp[lid + 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8) tmp[lid] += tmp[lid + 8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 4) tmp[lid] += tmp[lid + 4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 2) tmp[lid] += tmp[lid + 2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 1) out_dists[qid * k + kj] = tmp[0] + tmp[1];
}
