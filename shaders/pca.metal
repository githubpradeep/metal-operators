// Metal PCA kernels — super-fast GPU pipeline
// Mirrors flashlib's design: mean→center→cov/gram(GPU)→eigh(GPU)→sort/extract(GPU)→transform(GPU)

#include <metal_stdlib>
using namespace metal;

constant uint TILE = 8;

// ── Kernel 1: Column means (block-level partial sums) ─────────────
kernel void pca_mean(
    device const float* data        [[buffer(0)]],
    device float* means             [[buffer(1)]],
    device float* block_sums        [[buffer(2)]],
    constant uint& N                [[buffer(3)]],
    constant uint& D                [[buffer(4)]],
    constant uint& num_blocks       [[buffer(5)]],
    threadgroup float* shared       [[threadgroup(0)]],
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    uint block_id = gid;
    uint dim = lid;
    if (dim >= D) return;

    float sum = 0.0;
    uint row_start = block_id * 256;
    uint row_end = min(row_start + 256, N);
    for (uint r = row_start; r < row_end; r++) {
        sum += data[r * D + dim];
    }
    block_sums[block_id * D + dim] = sum;
}

// ── Kernel 2: Final reduction of block sums to means ──────────────
kernel void pca_mean_final(
    device const float* block_sums   [[buffer(0)]],
    device float* means             [[buffer(1)]],
    constant uint& num_blocks       [[buffer(2)]],
    constant uint& D                [[buffer(3)]],
    constant uint& N                [[buffer(4)]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    if (lid >= D) return;
    float sum = 0.0;
    for (uint b = 0; b < num_blocks; b++) {
        sum += block_sums[b * D + lid];
    }
    means[lid] = sum / float(N);
}

// ── Kernel 3: Center data (subtract column means) ─────────────────
kernel void pca_center(
    device const float* data    [[buffer(0)]],
    device const float* means   [[buffer(1)]],
    device float* centered      [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& D            [[buffer(4)]],
    uint id                     [[thread_position_in_grid]]
) {
    uint total = N * D;
    if (id >= total) return;
    uint row = id / D;
    uint col = id % D;
    centered[id] = data[id] - means[col];
}

// ── Kernel 4: Transpose (N,D) → (D,N) ────────────────────────────
kernel void pca_transpose(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    constant uint& D         [[buffer(3)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= D) return;
    dst[col * N + row] = src[row * D + col];
}

// ── Kernel 5: Matmul C = A @ B (simple, any dimensions) ───────────
// A: (M, K) row-major, stride_a
// B: (K, N) row-major, stride_b
// C: (M, N) row-major
kernel void pca_matmul(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    constant uint& M            [[buffer(3)]],
    constant uint& K            [[buffer(4)]],
    constant uint& N            [[buffer(5)]],
    constant uint& stride_a     [[buffer(6)]],
    constant uint& stride_b     [[buffer(7)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * stride_a + k] * B[k * stride_b + col];
    }
    C[row * N + col] = sum;
}

// ── Kernel 6: Jacobi eigendecomposition (single-thread, N ≤ 128) ──
// A: (N,N) symmetric — overwritten with eigenvalues on diagonal
// V: (N,N) eigenvectors as columns — overwritten
kernel void pca_eigh_jacobi(
    device float* A          [[buffer(0)]],
    device float* V          [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    constant float& tol      [[buffer(3)]],
    constant uint& max_sweeps [[buffer(4)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    for (uint i = 0; i < N; i++) {
        for (uint j = 0; j < N; j++) {
            V[i * N + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (uint sweep = 0; sweep < max_sweeps; sweep++) {
        bool converged = true;

        for (uint i = 0; i < N; i++) {
            for (uint j = i + 1; j < N; j++) {
                float a_ii = A[i * N + i];
                float a_jj = A[j * N + j];
                float a_ij = A[i * N + j];

                float threshold = tol * (fabs(a_ii) + fabs(a_jj)) * 0.5;
                if (fabs(a_ij) <= threshold) continue;
                converged = false;

                float tau = (a_jj - a_ii) / (2.0 * a_ij);
                float t;
                if (tau >= 0.0) {
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                } else {
                    t = 1.0 / (tau - sqrt(1.0 + tau * tau));
                }
                float c = 1.0 / sqrt(1.0 + t * t);
                float s = t * c;

                float A_ii_new = c*c*A[i*N+i] + s*s*A[j*N+j] - 2.0*c*s*A[i*N+j];
                float A_jj_new = s*s*A[i*N+i] + c*c*A[j*N+j] + 2.0*c*s*A[i*N+j];
                A[i*N+i] = A_ii_new;
                A[j*N+j] = A_jj_new;
                A[i*N+j] = 0.0;
                A[j*N+i] = 0.0;

                for (uint k = 0; k < N; k++) {
                    if (k != i && k != j) {
                        float A_ik = A[i*N + k];
                        float A_jk = A[j*N + k];
                        A[i*N + k] = c * A_ik - s * A_jk;
                        A[k*N + i] = A[i*N + k];
                        A[j*N + k] = s * A_ik + c * A_jk;
                        A[k*N + j] = A[j*N + k];
                    }
                    // V = V * J  (right-multiply: update columns i and j)
                    float V_ki = V[k*N + i];
                    float V_kj = V[k*N + j];
                    V[k*N + i] = c * V_ki - s * V_kj;
                    V[k*N + j] = s * V_ki + c * V_kj;
                }
            }
        }

        if (converged) break;
    }
}

// ── Kernel 7: Sort eigenvalues and eigenvectors (ascending) ───────
// Single-thread insertion sort (N ≤ 128).
kernel void pca_sort_eigen(
    device float* eigvals    [[buffer(0)]],
    device float* eigvecs    [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    for (uint i = 1; i < N; i++) {
        float key_val = eigvals[i];
        float key_vec[128];
        for (uint k = 0; k < N; k++) {
            key_vec[k] = eigvecs[k * N + i];
        }
        int j = int(i) - 1;
        while (j >= 0 && eigvals[j] > key_val) {
            eigvals[j + 1] = eigvals[j];
            for (uint k = 0; k < N; k++) {
                eigvecs[k * N + (j + 1)] = eigvecs[k * N + j];
            }
            j--;
        }
        eigvals[j + 1] = key_val;
        for (uint k = 0; k < N; k++) {
            eigvecs[k * N + (j + 1)] = key_vec[k];
        }
    }
}

// ── Kernel 8: Extract top-K eigenpairs (ascending → top-K) ───────
kernel void pca_extract_topk(
    device const float* eigvals  [[buffer(0)]],
    device const float* eigvecs  [[buffer(1)]],
    device float* out_vals       [[buffer(2)]],
    device float* out_vecs       [[buffer(3)]],
    constant uint& N             [[buffer(4)]],
    constant uint& K             [[buffer(5)]],
    constant uint& dim           [[buffer(6)]],
    uint tid                     [[thread_position_in_grid]]
) {
    for (uint i = tid; i < K; i += 256) {
        uint src_idx = N - K + i;
        out_vals[i] = eigvals[src_idx];
        for (uint j = 0; j < dim; j++) {
            out_vecs[j * K + i] = eigvecs[j * N + src_idx];
        }
    }
}

// ── Kernel 9: Explained variance ratio ────────────────────────────
kernel void pca_explained_variance(
    device const float* eigvals       [[buffer(0)]],
    device float* explained_var       [[buffer(1)]],
    constant uint& K                  [[buffer(2)]],
    constant float& total_var         [[buffer(3)]],
    uint id                           [[thread_position_in_grid]]
) {
    if (id >= K) return;
    explained_var[id] = eigvals[id] / total_var;
}

// ── Kernel 10: Transform — C = (X - mean) @ components^T ──────────
// X: (N, D), means: (D,), components: (K, D) row-major
// Output: (N, K)
kernel void pca_transform(
    device const float* X         [[buffer(0)]],
    device const float* means     [[buffer(1)]],
    device const float* components [[buffer(2)]],
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
        sum += (X[row * D + d] - means[d]) * components[col * D + d];
    }
    out[row * K + col] = sum;
}
