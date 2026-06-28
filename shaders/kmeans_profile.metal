#include <metal_stdlib>
using namespace metal;

// ── Profiling kernel: conditional stages controlled by ENABLE_* defines ──
// Set ENABLE_LOADCENT, ENABLE_MATMUL, ENABLE_BESTDIST to 0 or 1.

#ifndef ENABLE_LOADCENT
#define ENABLE_LOADCENT 1
#endif
#ifndef ENABLE_MATMUL
#define ENABLE_MATMUL 1
#endif
#ifndef ENABLE_BESTDIST
#define ENABLE_BESTDIST 1
#endif

kernel void kmeans_assign_profile(
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

    // shared memory layout
    threadgroup float* sh_pts     = shared;
    threadgroup float* sh_cent    = shared + PTILE * d;
    uint num_tiles = (d + CTILE - 1) / CTILE;
    threadgroup float* sh_dots    = shared + PTILE * d + d * CTILE;
    threadgroup float* sh_best_dist = sh_dots + num_tiles * CTILE * CTILE;
    threadgroup float* sh_best_lbl  = sh_best_dist + PTILE;

    // ── load points ──
    {
        uint total_pt = PTILE * d;
        for (uint i = lid; i < total_pt; i += tg_size) {
            uint po = i / d, pd = i % d;
            uint gi = p_start + po;
            sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
        }
        if (lid < PTILE) {
            sh_best_dist[lid] = INFINITY;
            sh_best_lbl[lid]  = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── loop over centroid tiles ──
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        // ── centroid load ──
        #if ENABLE_LOADCENT
        {
            uint total_load = c_tile * d;
            for (uint i = lid; i < total_load; i += tg_size) {
                uint co = i % c_tile;
                uint dim = i / c_tile;
                sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
            }
            // zero unused columns
            uint zero_cnt = d * (CTILE - c_tile);
            for (uint i = lid; i < zero_cnt; i += tg_size) {
                uint dd = i / (CTILE - c_tile);
                uint co = c_tile + i % (CTILE - c_tile);
                sh_cent[dd * CTILE + co] = 0.0f;
            }
        }
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── matmul: dim tiles round-robin across 4 simdgroups ──
        #if ENABLE_MATMUL
        for (uint dd = simd_gid; dd < num_tiles; dd += 4) {
            simdgroup_float8x8 A, B, t = {};
            simdgroup_load(A, sh_pts + dd * CTILE, d, 0, 0);
            simdgroup_load(B, sh_cent + dd * CTILE * CTILE, CTILE, 0, 0);
            simdgroup_multiply(t, A, B);
            simdgroup_store(t, sh_dots + dd * CTILE * CTILE, CTILE, 0, 0);
        }
        #else
        // zero out dots so best-dist still gets plausible values
        if (simd_gid == 0 && lid < PTILE * CTILE) {
            for (uint dd = 0; dd < num_tiles; dd++) {
                sh_dots[dd * CTILE * CTILE + lid] = 0.0f;
            }
        }
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── best distance update ──
        #if ENABLE_BESTDIST
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
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── write results ──
    if (lid < PTILE && p_start + lid < n) {
        uint pid = p_start + lid;
        assignments[pid]   = (uint)sh_best_lbl[lid];
        min_distances[pid] = sh_best_dist[lid];
    }
}

// ── PTILE=16 variant ──
// Same kernel but with 16 points per threadgroup.
// Matmul processes 2 batches of 8 points per dim tile.
// Shared memory layout: points (16×d) + cents (d×8) + dots (num_tiles×128) + best (32 floats)
#define PTILE16 16
kernel void kmeans_assign_p16(
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
    constexpr uint CTILE = 8;

    uint p_start = gid * PTILE16;
    if (p_start >= n) return;

    threadgroup float* sh_pts  = shared;
    threadgroup float* sh_cent = shared + PTILE16 * d;
    uint num_tiles = (d + CTILE - 1) / CTILE;
    threadgroup float* sh_dots    = shared + PTILE16 * d + d * CTILE;
    threadgroup float* sh_best_dist = sh_dots + num_tiles * PTILE16 * CTILE;
    threadgroup float* sh_best_lbl  = sh_best_dist + PTILE16;

    // ── load points ──
    {
        uint total_pt = PTILE16 * d;
        for (uint i = lid; i < total_pt; i += tg_size) {
            uint po = i / d, pd = i % d;
            uint gi = p_start + po;
            sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
        }
        if (lid < PTILE16) {
            sh_best_dist[lid] = INFINITY;
            sh_best_lbl[lid]  = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── loop over centroid tiles ──
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        #if ENABLE_LOADCENT
        {
            uint total_load = c_tile * d;
            for (uint i = lid; i < total_load; i += tg_size) {
                uint co = i % c_tile;
                uint dim = i / c_tile;
                sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
            }
            uint zero_cnt = d * (CTILE - c_tile);
            for (uint i = lid; i < zero_cnt; i += tg_size) {
                uint dd = i / (CTILE - c_tile);
                uint co = c_tile + i % (CTILE - c_tile);
                sh_cent[dd * CTILE + co] = 0.0f;
            }
        }
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── matmul: 2 batches of 8 points per dim tile ──
        #if ENABLE_MATMUL
        for (uint dd = simd_gid; dd < num_tiles; dd += 4) {
            simdgroup_float8x8 B;
            simdgroup_load(B, sh_cent + dd * CTILE * CTILE, CTILE, 0, 0);

            // batch 0: points 0..7
            simdgroup_float8x8 A0, t0 = {};
            simdgroup_load(A0, sh_pts + dd * CTILE, d, 0, 0);
            simdgroup_multiply(t0, A0, B);
            simdgroup_store(t0, sh_dots + dd * PTILE16 * CTILE, CTILE, 0, 0);

            // batch 1: points 8..15
            simdgroup_float8x8 A1, t1 = {};
            simdgroup_load(A1, sh_pts + 8 * d + dd * CTILE, d, 0, 0);
            simdgroup_multiply(t1, A1, B);
            simdgroup_store(t1, sh_dots + dd * PTILE16 * CTILE + 8 * CTILE, CTILE, 0, 0);
        }
        #else
        if (simd_gid == 0 && lid < PTILE16 * CTILE) {
            for (uint dd = 0; dd < num_tiles; dd++) {
                sh_dots[dd * PTILE16 * CTILE + lid] = 0.0f;
            }
        }
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── best distance update ──
        #if ENABLE_BESTDIST
        if (lid < PTILE16 && p_start + lid < n) {
            float best_d = sh_best_dist[lid];
            uint  best_l = (uint)sh_best_lbl[lid];
            float nx     = norms_X[p_start + lid];

            for (uint c = 0; c < c_tile; c++) {
                float dot = 0.0f;
                for (uint dd = 0; dd < num_tiles; dd++) {
                    dot += sh_dots[dd * PTILE16 * CTILE + lid * CTILE + c];
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
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < PTILE16 && p_start + lid < n) {
        uint pid = p_start + lid;
        assignments[pid]   = (uint)sh_best_lbl[lid];
        min_distances[pid] = sh_best_dist[lid];
    }
}

// ── CTILE=16 variant ──
// 16 centroids per tile (instead of 8). Matmul processes 2 centroid batches per dim tile.
// Half the number of centroid tiles = fewer barriers & centroid loads.
kernel void kmeans_assign_c16(
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
    uint num_tiles = (d + CTILE - 1) / CTILE; // d / 8, rounded up
    // Note: num_tiles is still based on 8-dim tiles, not CTILE.
    // But we use CTILE=16 for the dot product matrix dimension.
    // Let's use dim_tile_count = (d + 7) / 8 (CTILE in the original sense = 8 for dim tile size)
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
        if (lid < PTILE) {
            sh_best_dist[lid] = INFINITY;
            sh_best_lbl[lid]  = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── loop over centroid tiles ──
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        #if ENABLE_LOADCENT
        {
            // load centroids transposed: sh_cent[dim * CTILE + co]
            uint total_load = c_tile * d;
            for (uint i = lid; i < total_load; i += tg_size) {
                uint co = i % c_tile;
                uint dim = i / c_tile;
                sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
            }
            // zero unused centroid columns (co = c_tile .. CTILE-1)
            uint zero_cnt = d * (CTILE - c_tile);
            for (uint i = lid; i < zero_cnt; i += tg_size) {
                uint dd = i / (CTILE - c_tile);
                uint co = c_tile + i % (CTILE - c_tile);
                sh_cent[dd * CTILE + co] = 0.0f;
            }
        }
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── matmul: 2 centroid batches per dim tile ──
        #if ENABLE_MATMUL
        for (uint dd = simd_gid; dd < dim_tiles; dd += 4) {
            simdgroup_float8x8 A;
            simdgroup_load(A, sh_pts + dd * 8, d, 0, 0);

            // batch 0: cents 0..7
            simdgroup_float8x8 B0, t0 = {};
            simdgroup_load(B0, sh_cent + (dd * 8) * CTILE, CTILE, 0, 0);
            simdgroup_multiply(t0, A, B0);
            simdgroup_store(t0, sh_dots + dd * PTILE * CTILE, CTILE, 0, 0);

            // batch 1: cents 8..15
            simdgroup_float8x8 B1, t1 = {};
            simdgroup_load(B1, sh_cent + (dd * 8) * CTILE + 8, CTILE, 0, 0);
            simdgroup_multiply(t1, A, B1);
            simdgroup_store(t1, sh_dots + dd * PTILE * CTILE + 8, CTILE, 0, 0);
        }
        #else
        if (simd_gid == 0 && lid < PTILE * CTILE) {
            for (uint dd = 0; dd < dim_tiles; dd++) {
                sh_dots[dd * PTILE * CTILE + lid] = 0.0f;
            }
        }
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── best distance update ──
        #if ENABLE_BESTDIST
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
        #endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < PTILE && p_start + lid < n) {
        uint pid = p_start + lid;
        assignments[pid]   = (uint)sh_best_lbl[lid];
        min_distances[pid] = sh_best_dist[lid];
    }
}
