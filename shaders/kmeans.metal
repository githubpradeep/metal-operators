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

// SIMD-group matrix-multiply kernel.
// Each threadgroup (128 threads) processes an 8×8 output tile
// (8 points × 8 centroids) using ONLY simdgroup 0.
// For K > 8 the kernel loops over centroid tiles.
// Requires D % 8 == 0.
//
// NOTE: Only simd_gid=0 does matrix work. Multiple simdgroups doing
// simdgroup_load/store at different column offsets on the same
// threadgroup buffer was found to produce incorrect results on
// Apple GPUs (simd_gid >= 1 gives garbage). By restricting to
// a single 8-centroid tile per iteration we avoid this issue.
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
    constexpr uint PTILE = 8;   // points per threadgroup
    constexpr uint CTILE = 8;   // centroids per inner loop (1 simdgroup × 8)

    uint p_start = gid * PTILE;
    if (p_start >= n) return;

    // Threadgroup memory layout:
    //   [0    .. PTILE*d)                – points tile (row-major: point × dim)
    //   [PTILE*d .. +d*CTILE)            – centroids tile (transposed: dim × centroid)
    //   [+d*CTILE .. +num_tiles*64)      – per-tile dot-product results (num_tiles × CTILE × CTILE)
    //   [+num_tiles*64 .. +num_tiles*64+PTILE)       – best distance per point
    //   [+num_tiles*64+PTILE .. +num_tiles*64+2*PTILE) – best label per point (stored as float)
    threadgroup float* sh_pts  = shared;
    threadgroup float* sh_cent = shared + PTILE * d;
    uint num_tiles = (d + CTILE - 1) / CTILE;
    threadgroup float* sh_dots = shared + PTILE * d + d * CTILE;
    threadgroup float* sh_best_dist = sh_dots + num_tiles * CTILE * CTILE;
    threadgroup float* sh_best_lbl  = sh_best_dist + PTILE;

    // ── load points into shared memory ──
    uint total_pt = PTILE * d;
    for (uint i = lid; i < total_pt; i += tg_size) {
        uint po = i / d, pd = i % d;
        uint gi = p_start + po;
        sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
    }

    // initialise best dist / label per point
    if (lid < PTILE) { sh_best_dist[lid] = INFINITY; sh_best_lbl[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── loop over centroid tiles ──
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);

        // load centroids TRANSPOSED: sh_cent[dim * CTILE + co] = C[c_base + co, dim]
        {
            uint total_load = c_tile * d;
            for (uint i = lid; i < total_load; i += tg_size) {
                uint co = i % c_tile;
                uint dim = i / c_tile;
                sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
            }
            // zero out unused centroid columns (c_tile..CTILE-1)
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

        // only simdgroup 0 does matrix work (see note at top)
        if (simd_gid == 0) {
            uint num_tiles = (d + CTILE - 1) / CTILE;

            // Compute each dim tile separately and store to its own sh_dots slot
            for (uint dd = 0; dd < num_tiles; dd++) {
                simdgroup_float8x8 A, B, t = {};
                // Row-first addressing: sh_pts[(row+i)*stride + (col+j)]
                // A[i][j] = (sh_pts + dd*8)[i*d + j] = point i, dim (dd*8 + j)
                simdgroup_load(A, sh_pts + dd * CTILE, d, 0, 0);
                // B[i][j] = (sh_cent + dd*64)[i*CTILE + j] = centroid j, dim (dd*8 + i)
                simdgroup_load(B, sh_cent + dd * CTILE * CTILE, CTILE, 0, 0);
                // t = A * B on this GPU replaces t (does not accumulate)
                simdgroup_multiply(t, A, B);
                // t[row=i][col=j] → sh_dots[dd*64 + i*8 + j] = dot over dims dd*8..dd*8+7
                simdgroup_store(t, sh_dots + dd * CTILE * CTILE, CTILE, 0, 0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each of the first PTILE threads picks one point & updates best result
        if (lid < PTILE && p_start + lid < n) {
            float best_d = sh_best_dist[lid];
            uint  best_l = (uint)sh_best_lbl[lid];
            float nx     = norms_X[p_start + lid];

            uint num_tiles = (d + CTILE - 1) / CTILE;
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
