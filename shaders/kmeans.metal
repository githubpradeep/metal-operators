#include <metal_stdlib>
using namespace metal;

// Naive per-point loop over centroids from device memory.
// Used when centroids are too large to fit in threadgroup memory.
kernel void kmeans_assign(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* min_distances [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;

    float min_dist = INFINITY;
    uint best = 0;

    for (uint c = 0; c < k; c++) {
        float dist = 0.0;
        for (uint dim = 0; dim < d; dim++) {
            float diff = points[idx * d + dim] - centroids[c * d + dim];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }

    assignments[idx] = best;
    min_distances[idx] = min_dist;
}

// SIMD-group matrix-multiply kernel using ALL 4 simdgroups.
// Each threadgroup (128 threads) processes an 8×8 output tile
// (8 points × 8 centroids). Dim-tiles are distributed round-robin
// across simdgroups (sg 0 takes tiles 0,4,8,…; sg1 takes 1,5,9,…).
// For K > 8 the kernel loops over centroid tiles.
// Requires D % 8 == 0.
//
// NOTE: simdgroup_load/store with non-zero col or row produces
// garbage on this GPU. All loads/stores use col=0,row=0 and
// adjust the base pointer instead.
kernel void kmeans_assign_simdgroup(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* min_distances [[buffer(3)]],
    device const float* norms_X [[buffer(4)]],
    device const float* norms_C [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& d [[buffer(8)]],
    constant uint& tg_size [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint PTILE = 8;
    constexpr uint CTILE = 8;

    uint p_start = gid * PTILE;
    if (p_start >= n) return;

    threadgroup float* sh_pts  = shared;
    threadgroup float* sh_cent = shared + PTILE * d;
    uint num_tiles = (d + CTILE - 1) / CTILE;
    threadgroup float* sh_dots = shared + PTILE * d + d * CTILE;
    threadgroup float* sh_best_dist = sh_dots + num_tiles * CTILE * CTILE;
    threadgroup float* sh_best_lbl  = sh_best_dist + PTILE;

    // ── load points ──
    uint total_pt = PTILE * d;
    for (uint i = lid; i < total_pt; i += tg_size) {
        uint po = i / d, pd = i % d;
        uint gi = p_start + po;
        sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
    }

    if (lid < PTILE) { sh_best_dist[lid] = INFINITY; sh_best_lbl[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── loop over centroid tiles ──
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        // load centroids transposed
        {
            uint total_load = c_tile * d;
            for (uint i = lid; i < total_load; i += tg_size) {
                uint co = i % c_tile;
                uint dim = i / c_tile;
                sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
            }
            // zero unused centroid columns
            {
                uint zero_cnt = d * (CTILE - c_tile);
                for (uint i = lid; i < zero_cnt; i += tg_size) {
                    uint dd = i / (CTILE - c_tile);
                    uint co = c_tile + i % (CTILE - c_tile);
                    sh_cent[dd * CTILE + co] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── ALL 4 simdgroups compute dim tiles, distributed round-robin ──
        for (uint dd = simd_gid; dd < num_tiles; dd += 4) {
            simdgroup_float8x8 A, B, t = {};
            // A[i][j] = sh_pts[dd*8 + i*d + j] = point i, dim (dd*8 + j)
            simdgroup_load(A, sh_pts + dd * CTILE, d, 0, 0);
            // B[i][j] = sh_cent[dd*64 + i*8 + j] = centroid j, dim (dd*8 + i)
            simdgroup_load(B, sh_cent + dd * CTILE * CTILE, CTILE, 0, 0);
            simdgroup_multiply(t, A, B);
            simdgroup_store(t, sh_dots + dd * CTILE * CTILE, CTILE, 0, 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── each point thread updates best distance ──
        if (lid < PTILE && p_start + lid < n) {
            float best_d = sh_best_dist[lid];
            uint  best_l = (uint)sh_best_lbl[lid];
            float nx     = norms_X[p_start + lid];

            for (uint c = 0; c < c_tile; c++) {
                float dot = 0.0f;
                for (uint dd = 0; dd < num_tiles; dd++) {
                    dot += sh_dots[dd * CTILE * CTILE + lid * CTILE + c];
                }
                float dist = nx + norms_C[c_base + c] - 2.0f * dot;
                if (dist < best_d) {
                    best_d = dist;
                    best_l = c_base + c;
                }
            }
            sh_best_dist[lid] = best_d;
            sh_best_lbl[lid]  = (float)best_l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── write final results ──
    if (lid < PTILE && p_start + lid < n) {
        uint pid = p_start + lid;
        assignments[pid]   = (uint)sh_best_lbl[lid];
        min_distances[pid] = sh_best_dist[lid];
    }
}

// CTILE=16 simdgroup kernel: for K ≤ 16 (single centroid tile, fewer barriers).
// Same as kmeans_assign_simdgroup but with CTILE=16, centroid tiles of 16 instead of 8.
// Processes 2 centroid batches per dim tile in the matmul.
kernel void kmeans_assign_simdgroup_c16(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* min_distances [[buffer(3)]],
    device const float* norms_X [[buffer(4)]],
    device const float* norms_C [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& d [[buffer(8)]],
    constant uint& tg_size [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint PTILE = 8;
    constexpr uint CTILE = 16;

    uint p_start = gid * PTILE;
    if (p_start >= n) return;

    threadgroup float* sh_pts  = shared;
    threadgroup float* sh_cent = shared + PTILE * d;
    uint dim_tiles = (d + 7) / 8;
    threadgroup float* sh_dots    = shared + PTILE * d + d * CTILE;
    threadgroup float* sh_best_dist = sh_dots + dim_tiles * PTILE * CTILE;
    threadgroup float* sh_best_lbl  = sh_best_dist + PTILE;

    // ── load points ──
    {
        uint total_pt = PTILE * d;
        for (uint i = lid; i < total_pt; i += tg_size) {
            uint po = i / d, pd = i % d;
            uint gi = p_start + po;
            sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
        }
        if (lid < PTILE) { sh_best_dist[lid] = INFINITY; sh_best_lbl[lid] = 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── loop over centroid tiles ──
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        // ── centroid load (transposed) ──
        {
            uint total_load = c_tile * d;
            for (uint i = lid; i < total_load; i += tg_size) {
                uint co = i % c_tile;
                uint dim = i / c_tile;
                sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
            }
            // zero unused centroid columns
            {
                uint zero_cnt = d * (CTILE - c_tile);
                for (uint i = lid; i < zero_cnt; i += tg_size) {
                    uint dd = i / (CTILE - c_tile);
                    uint co = c_tile + i % (CTILE - c_tile);
                    sh_cent[dd * CTILE + co] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── matmul: 2 centroid batches per dim tile ──
        for (uint dd = simd_gid; dd < dim_tiles; dd += 4) {
            simdgroup_float8x8 A;
            simdgroup_load(A, sh_pts + dd * 8, d, 0, 0);

            // batch 0: cents 0..7
            simdgroup_float8x8 B0, t0 = {};
            simdgroup_load(B0, sh_cent + dd * 8 * CTILE, CTILE, 0, 0);
            simdgroup_multiply(t0, A, B0);
            simdgroup_store(t0, sh_dots + dd * PTILE * CTILE, CTILE, 0, 0);

            // batch 1: cents 8..15
            simdgroup_float8x8 B1, t1 = {};
            simdgroup_load(B1, sh_cent + dd * 8 * CTILE + 8, CTILE, 0, 0);
            simdgroup_multiply(t1, A, B1);
            simdgroup_store(t1, sh_dots + dd * PTILE * CTILE + 8, CTILE, 0, 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── best distance update ──
        if (lid < PTILE && p_start + lid < n) {
            float best_d = sh_best_dist[lid];
            uint  best_l = (uint)sh_best_lbl[lid];
            float nx     = norms_X[p_start + lid];

            for (uint c = 0; c < c_tile; c++) {
                float dot = 0.0f;
                for (uint dd = 0; dd < dim_tiles; dd++) {
                    dot += sh_dots[dd * PTILE * CTILE + lid * CTILE + c];
                }
                float dist = nx + norms_C[c_base + c] - 2.0f * dot;
                if (dist < best_d) {
                    best_d = dist;
                    best_l = c_base + c;
                }
            }
            sh_best_dist[lid] = best_d;
            sh_best_lbl[lid]  = (float)best_l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── write final results ──
    if (lid < PTILE && p_start + lid < n) {
        uint pid = p_start + lid;
        assignments[pid]   = (uint)sh_best_lbl[lid];
        min_distances[pid] = sh_best_dist[lid];
    }
}

// Split-D assign kernel: for large D where simdgroup shared memory doesn't fit.
// Each threadgroup (128 threads) handles 128 points, each thread one point.
// Outer loop over centroids in CTILE=8 chunks, inner loop over D in BD=32 chunks.
// Cross term accumulated in per-thread registers across D chunks.
// Shared memory: only BD × CTILE = 256 floats for centroids chunk.
// Works for any D (no D % 8 requirement, no D limit).
kernel void kmeans_assign_splitd(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    device float* min_distances [[buffer(3)]],
    device const float* norms_X [[buffer(4)]],
    device const float* norms_C [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& d [[buffer(8)]],
    constant uint& tg_size [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    constexpr uint PTILE = 128;
    constexpr uint CTILE = 8;
    constexpr uint BD = 32;

    uint pid = gid * PTILE + lid;
    if (pid >= n) return;

    // shared[0 .. BD*CTILE) = centroids chunk (transposed: dim × centroid)
    threadgroup float* sh_cent = shared;

    float best_dist = INFINITY;
    uint  best_lbl = 0;
    float px_norm = norms_X[pid];

    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        // cross accumulator per centroid in this K tile (registers across D loop)
        float dot_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (uint dd = 0; dd < d; dd += BD) {
            uint bd_tile = min(BD, d - dd);

            // ── cooperative load: centroids chunk (BD × CTILE) into shared ──
            uint total = bd_tile * c_tile;
            for (uint i = lid; i < total; i += PTILE) {
                uint bd = i / c_tile;
                uint cc = i % c_tile;
                sh_cent[bd * CTILE + cc] = centroids[(c_base + cc) * d + dd + bd];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── each thread accumulates its point × centroid dot products ──
            float px_base = points[pid * d + dd]; // just hint to compiler
            for (uint co = 0; co < c_tile; co++) {
                float partial = 0.0f;
                for (uint bd = 0; bd < bd_tile; bd++) {
                    partial += points[pid * d + dd + bd] * sh_cent[bd * CTILE + co];
                }
                dot_acc[co] += partial;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── compute distances & update best ──
        for (uint co = 0; co < c_tile; co++) {
            float cent_norm = norms_C[c_base + co];
            float dist = px_norm + cent_norm - 2.0f * dot_acc[co];
            dist = max(dist, 0.0f);
            if (dist < best_dist) {
                best_dist = dist;
                best_lbl = c_base + co;
            }
        }
    }

    assignments[pid] = best_lbl;
    min_distances[pid] = best_dist;
}

kernel void kmeans_compute_min_distances(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device float* min_dists [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& num_centroids [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;

    float min_dist = INFINITY;
    for (uint c = 0; c < num_centroids; c++) {
        float dist = 0.0;
        for (uint dim = 0; dim < d; dim++) {
            float diff = points[idx * d + dim] - centroids[c * d + dim];
            dist += diff * diff;
        }
        if (dist < min_dist) min_dist = dist;
    }
    min_dists[idx] = min_dist;
}

// ── GPU centroid update: per-point device atomics ──
// Each thread handles BATCH=4 points to reduce atomic contention 4×.
kernel void kmeans_centroid_accum(
    device const float    *points         [[buffer(0)]],
    device const uint     *assignments    [[buffer(1)]],
    device atomic<float>  *centroid_sums  [[buffer(2)]],
    device atomic_int     *centroid_counts[[buffer(3)]],
    constant uint& n  [[buffer(4)]],
    constant uint& k  [[buffer(5)]],
    constant uint& d  [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * 4;
    uint count = min(4u, n - base);
    for (uint p = 0; p < count; p++) {
        uint pid = base + p;
        uint label = assignments[pid];
        if (label >= k) label = 0;
        uint sum_off = label * d;
        uint pt_off  = pid * d;
        for (uint dd = 0; dd < d; dd++) {
            atomic_fetch_add_explicit(&centroid_sums[sum_off + dd],
                points[pt_off + dd], memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&centroid_counts[label], 1, memory_order_relaxed);
    }
}

// ── GPU centroid update: tiled, no threadgroup atomics needed ──
// Each thread handles exactly one unique dimension (dim = lid).
// The outer loop iterates over points; each thread reads its dimension
// from each point and accumulates to shared[label*d + dim].
// Since each thread has a unique dim, there are no write conflicts
// on shared memory — no threadgroup atomics needed.
// Thread 0 (dim=0) also handles the per-label counts.
// Then each thread flushes its per-label sums to global with device
// atomics (one atomic per non-zero label per dim).
// Requires: d <= tg_size (currently 128) and K*D + K floats in shared.
kernel void kmeans_centroid_tiled(
    device const float* points [[buffer(0)]],
    device const uint* assignments [[buffer(1)]],
    device atomic<float>* centroid_sums [[buffer(2)]],
    device atomic_int* centroid_counts [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    constexpr uint PTILE = 128;
    uint p_start = gid * PTILE;
    if (p_start >= n) return;

    uint dim = lid;
    if (dim >= d) return; // extra threads are idle

    uint count = min(PTILE, n - p_start);

    // Zero this thread's dimension in shared (k entries, one per label)
    for (uint c = 0; c < k; c++) {
        shared[c * d + dim] = 0.0f;
    }
    if (dim == 0) {
        for (uint c = 0; c < k; c++) {
            shared[k * d + c] = 0.0f; // counts
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process all points: each thread reads only its dimension
    for (uint i = 0; i < count; i++) {
        uint p = p_start + i;
        uint label = assignments[p];
        shared[label * d + dim] += points[p * d + dim];
        if (dim == 0) {
            shared[k * d + label] += 1.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush per-label sums to global (device atomics)
    for (uint c = 0; c < k; c++) {
        float val = shared[c * d + dim];
        if (val != 0.0f) {
            atomic_fetch_add_explicit(&centroid_sums[c * d + dim], val, memory_order_relaxed);
        }
    }
    if (dim == 0) {
        for (uint c = 0; c < k; c++) {
            float val = shared[k * d + c];
            if (val != 0.0f) {
                atomic_fetch_add_explicit(&centroid_counts[c], (int)val, memory_order_relaxed);
            }
        }
    }
}
