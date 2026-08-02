// Metal Linear Regression kernels
//
// Closed-form ordinary least squares via the normal equations (matches
// flashlib's linear regression and sklearn's 'cholesky' solver):
//
//   minimize  (1/2) ||X w + b - y||^2 + (alpha/2) ||w||^2
//
// with optional L2 ridge (alpha) and optional intercept (b). The (d+1)
// augmented Gram system is
//
//   [ XᵀX + alpha·I   colsum ] [w]   [ Xᵀy ]
//   [ colsumᵀ           n    ] [b] = [ Σy  ]
//
// where colsum = Xᵀ·1 (column sums) and n = number of rows.
//
// Kernel design (deterministic; no device atomics):
//   - `linreg_gram_xtx_tiled` — d <= 128. One threadgroup per row-block;
//     the block is staged once into shared memory (B x d tile, <= 30 KB),
//     then all d² elements of XᵀX are accumulated from the tile into
//     per-thread registers (d²/256 slots per thread). Each group writes one
//     partial row to a per-group partials buffer.
//   - `linreg_gram_xtx_naive` — d > 128. One thread per (j,k) element of
//     XᵀX; thread p (j = p % d, k = p / d) streams the dataset once. Each
//     element has exactly one owner, so the output is final (no reduction).
//   - `linreg_gram_xty` — chunked Xᵀy / column sums / count / Σy. Each
//     group owns a set of elements e in [0, 2d+2):
//       e < d       ->  xty[e]   = Σ_s X[s,e]·y[s]
//       d <= e < 2d ->  colsum[e-d] = Σ_s X[s,e-d]
//       e == 2d     ->  n   (row count)
//       e == 2d+1   ->  Σy
//     accumulating across its row-chunks in registers; one partial row per
//     group.
//   - `linreg_reduce_gram` — deterministic fixed-order reduction of the
//     per-group partial rows into the final XᵀX / Xᵀy / colsum / stats.
//   - `linreg_predict` — per-sample prediction z = X·w + b.
//
// The (d+1)x(d+1) augmented system is solved on the host with Gaussian
// elimination + partial pivoting (O(d^3) with d <= a few hundred).

#include <metal_stdlib>
using namespace metal;

// ── Kernel 1a: XᵀX via shared-memory row tiles (d <= 128) ──────────────────
// Grid: G groups (G = min(row-chunks, cap)). Group gid processes chunks
// gid, gid+G, ... loading each B x d row-block into shared memory once and
// accumulating ALL d² pair-products from the tile into 64 register slots
// (d²/256 <= 64 for d <= 128). Writes one partial row (d² floats) per group;
// `linreg_reduce_gram` sums the rows in fixed order.

kernel void linreg_gram_xtx_tiled(
    device const float* X          [[buffer(0)]],  // (n, d) row-major
    device float*       partials   [[buffer(1)]],  // G x d² row-major
    constant uint&      n          [[buffer(2)]],
    constant uint&      d          [[buffer(3)]],
    constant uint&      B          [[buffer(4)]],  // rows per chunk
    constant uint&      G          [[buffer(5)]],  // number of groups
    threadgroup float*  tile       [[threadgroup(0)]],  // B x d floats
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint n_slots = (d * d + TG - 1) / TG;
    float acc[64];
    for (uint q = 0; q < n_slots; q++) {
        acc[q] = 0.0f;
    }

    uint total = B * d;
    for (uint chunk = gid; chunk * B < n; chunk += G) {
        // Cooperative coalesced load of rows [chunk*B, chunk*B+B) (zero-padded).
        for (uint i = lid; i < total; i += TG) {
            uint r = i / d;
            uint c = i % d;
            uint s = chunk * B + r;
            tile[i] = (s < n) ? X[s * d + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate this thread's d²/256 pair slots from the tile.
        for (uint q = 0; q < n_slots; q++) {
            uint p = lid + q * TG;
            if (p < d * d) {
                uint j = p % d;
                uint k = p / d;
                float a = 0.0f;
                for (uint r = 0; r < B; r++) {
                    a += tile[r * d + j] * tile[r * d + k];
                }
                acc[q] += a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write this group's partial row.
    for (uint q = 0; q < n_slots; q++) {
        uint p = lid + q * TG;
        if (p < d * d) {
            partials[gid * (d * d) + p] = acc[q];
        }
    }
}

// ── Kernel 1b: XᵀX via one thread per (j,k) element (d > 128) ──────────────
// Thread p owns element xtx[j][k] (j = p % d, k = p / d) and sums
// X[s,j]·X[s,k] over all s. No cross-thread communication -> deterministic
// and race-free; each element has exactly one owner so the output is final.

kernel void linreg_gram_xtx_naive(
    device const float* X          [[buffer(0)]],  // (n, d) row-major
    device float*       xtx        [[buffer(1)]],  // (d*d,) output
    constant uint&      n          [[buffer(2)]],
    constant uint&      d          [[buffer(3)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint p = gid * TG + lid;
    uint n_pairs = d * d;
    if (p >= n_pairs) return;

    uint j = p % d;
    uint k = p / d;

    float acc = 0.0f;
    for (uint s = 0; s < n; s++) {
        acc += X[s * d + j] * X[s * d + k];
    }
    xtx[p] = acc;
}

// ── Kernel 2: chunked Xᵀy, column sums, row count and Σy ───────────────────
// Grid: G groups; group gid processes chunks gid, gid+G, ... of 256 rows.
// Thread t owns elements e = t, t+256, ... of [0, 2d+2) and accumulates its
// columns across the group's chunks in registers (2d+2 <= 32*256 guarded on
// the host). One partial row (2d+2 floats) per group.

kernel void linreg_gram_xty(
    device const float* X          [[buffer(0)]],  // (n, d) row-major
    device const float* y          [[buffer(1)]],  // (n,)
    device float*       partials   [[buffer(2)]],  // G x (2d+2) row-major
    constant uint&      n          [[buffer(3)]],
    constant uint&      d          [[buffer(4)]],
    constant uint&      G          [[buffer(5)]],  // number of groups
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    const uint B = 256;
    uint n_elems = 2 * d + 2;
    uint n_slots = (n_elems + TG - 1) / TG;
    float acc[32];
    for (uint q = 0; q < n_slots; q++) {
        acc[q] = 0.0f;
    }

    for (uint chunk = gid; chunk * B < n; chunk += G) {
        uint row0 = chunk * B;
        for (uint q = 0; q < n_slots; q++) {
            uint e = lid + q * TG;
            if (e >= n_elems) continue;
            float a = 0.0f;
            if (e < d) {
                // Xᵀy column e
                for (uint r = 0; r < B; r++) {
                    uint s = row0 + r;
                    if (s < n) a += X[s * d + e] * y[s];
                }
            } else if (e < 2 * d) {
                // column sum of column (e - d)
                uint j = e - d;
                for (uint r = 0; r < B; r++) {
                    uint s = row0 + r;
                    if (s < n) a += X[s * d + j];
                }
            } else if (e == 2 * d) {
                // count of rows in this chunk
                uint cnt = (row0 + B <= n) ? B : (n - row0);
                a = (float)cnt;
            } else {
                // sum of y
                for (uint r = 0; r < B; r++) {
                    uint s = row0 + r;
                    if (s < n) a += y[s];
                }
            }
            acc[q] += a;
        }
    }

    // Write this group's partial row.
    for (uint q = 0; q < n_slots; q++) {
        uint e = lid + q * TG;
        if (e < n_elems) {
            partials[gid * n_elems + e] = acc[q];
        }
    }
}

// ── Kernel 3: deterministic reduction of per-group partials ────────────────
// 256 threads; thread t sums column t (and t+256, ...) over all group rows
// in fixed order, writing [XᵀX (d²) | Xᵀy (d) | colsum (d) | n | Σy].
// With G_xtx == 0 (naive XᵀX path) the XᵀX part is left untouched.

kernel void linreg_reduce_gram(
    device const float* xtx_partials [[buffer(0)]],  // G_xtx x d²
    device const float* xty_partials [[buffer(1)]],  // G_xty x (2d+2)
    device float*       xtx          [[buffer(2)]],  // (d²,)
    device float*       xty          [[buffer(3)]],  // (d,)
    device float*       colsum       [[buffer(4)]],  // (d,)
    device float*       stats        [[buffer(5)]],  // (2,) = [n, Σy]
    constant uint&      g_xtx        [[buffer(6)]],
    constant uint&      g_xty        [[buffer(7)]],
    constant uint&      d            [[buffer(8)]],
    uint lid                          [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint n_pairs = d * d;
    uint n_elems = 2 * d + 2;

    // With g_xtx == 0 (naive XᵀX path, which writes xtx directly) the XᵀX
    // part must be left untouched, not overwritten with zero.
    if (g_xtx > 0) {
        for (uint p = lid; p < n_pairs; p += TG) {
            float acc = 0.0f;
            for (uint g = 0; g < g_xtx; g++) {
                acc += xtx_partials[g * n_pairs + p];
            }
            xtx[p] = acc;
        }
    }

    for (uint e = lid; e < n_elems; e += TG) {
        float acc = 0.0f;
        for (uint g = 0; g < g_xty; g++) {
            acc += xty_partials[g * n_elems + e];
        }
        if (e < d) {
            xty[e] = acc;
        } else if (e < 2 * d) {
            colsum[e - d] = acc;
        } else if (e == 2 * d) {
            stats[0] = acc;
        } else {
            stats[1] = acc;
        }
    }
}

// ── Kernel 4: Predict ──────────────────────────────────────────────────────
// One sample per thread; z = X·w + b.

kernel void linreg_predict(
    device const float* data    [[buffer(0)]],  // (n, d) row-major
    device const float* weights [[buffer(1)]],  // (d,)
    device float*       out     [[buffer(2)]],  // (n,) output
    constant uint&      n       [[buffer(3)]],
    constant uint&      d       [[buffer(4)]],
    constant float&     bias    [[buffer(5)]],
    uint gid                     [[threadgroup_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]]
) {
    const uint TG = 256;
    uint s = gid * TG + lid;
    if (s >= n) return;

    float z = bias;
    for (uint j = 0; j < d; j++) {
        z += weights[j] * data[s * d + j];
    }
    out[s] = z;
}
