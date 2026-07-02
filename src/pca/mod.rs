use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn ssyevd_(
        jobz: *const u8, uplo: *const u8, n: *const i32,
        a: *mut f32, lda: *const i32, w: *mut f32,
        work: *mut f32, lwork: *const i32,
        iwork: *mut i32, liwork: *const i32,
        info: *mut i32,
    );
}

const SHADER_SRC: &str = include_str!("../../shaders/pca.metal");

struct PipelineCache {
    mean: OnceLock<ComputePipelineState>,
    mean_final: OnceLock<ComputePipelineState>,
    center: OnceLock<ComputePipelineState>,
    transpose: OnceLock<ComputePipelineState>,
    matmul: OnceLock<ComputePipelineState>,
    transform: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            mean: OnceLock::new(),
            mean_final: OnceLock::new(),
            center: OnceLock::new(),
            transpose: OnceLock::new(),
            matmul: OnceLock::new(),
            transform: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "pca_mean" => &self.mean,
            "pca_mean_final" => &self.mean_final,
            "pca_center" => &self.center,
            "pca_transpose" => &self.transpose,
            "pca_matmul" => &self.matmul,
            "pca_transform" => &self.transform,
            _ => anyhow::bail!("Unknown pipeline: {}", name),
        };
        if let Some(p) = slot.get() {
            return Ok(p);
        }
        let p = ctx.compile_kernel(SHADER_SRC, name)?;
        slot.set(p).map_err(|_| anyhow::anyhow!("pipeline race"))?;
        Ok(slot.get().unwrap())
    }
}

pub struct PCAConfig {
    pub n_components: usize,
}

impl Default for PCAConfig {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

pub struct PCA {
    config: PCAConfig,
    components: Vec<f32>,
    explained_variance: Vec<f32>,
    explained_variance_ratio: Vec<f32>,
    mean: Vec<f32>,
    singular_values: Vec<f32>,
    noise_variance: f32,
    n: usize,
    d: usize,
    pipelines: PipelineCache,
}

impl PCA {
    pub fn new(config: PCAConfig) -> Self {
        Self {
            config,
            components: Vec::new(),
            explained_variance: Vec::new(),
            explained_variance_ratio: Vec::new(),
            mean: Vec::new(),
            singular_values: Vec::new(),
            noise_variance: 0.0,
            n: 0,
            d: 0,
            pipelines: PipelineCache::new(),
        }
    }

    pub fn components(&self) -> &[f32] { &self.components }
    pub fn explained_variance(&self) -> &[f32] { &self.explained_variance }
    pub fn explained_variance_ratio(&self) -> &[f32] { &self.explained_variance_ratio }
    pub fn mean(&self) -> &[f32] { &self.mean }
    pub fn singular_values(&self) -> &[f32] { &self.singular_values }
    pub fn noise_variance(&self) -> f32 { self.noise_variance }
    pub fn n_features(&self) -> usize { self.d }
    pub fn n_samples(&self) -> usize { self.n }

    /// Fit PCA — GPU-accelerated Gram matrix, CPU eigendecomposition.
    ///
    /// Strategy (mirrors flashlib):
    ///   - GPU: mean → center → transpose → matmul (chained, single cmd buffer)
    ///   - CPU: eigendecomposition on the small (min(N,D)) Gram matrix
    ///   - CPU: sort, extract, recover (if gram path), convert to sklearn
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(n > 0 && d > 0, "Data must be non-empty");
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");
        anyhow::ensure!(
            self.config.n_components > 0 && self.config.n_components <= d,
            "n_components must be between 1 and D ({})", d
        );

        let k = self.config.n_components;
        self.n = n;
        self.d = d;

        let use_gram_path = n < d;
        let gram_dim = if use_gram_path { n } else { d };

        // ── GPU pipeline: mean → center → transpose → matmul ──────
        //
        // All buffers allocated upfront; all kernels dispatched
        // sequentially in one command buffer; only the Gram matrix
        // is read back to CPU for eigendecomposition.

        let data_buf = ctx.new_buffer(data);

        let block_size: u64 = 256;
        let num_blocks = ((n as u64) + block_size - 1) / block_size;
        let means_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let block_sums_buf =
            ctx.new_buffer_uninitialized((num_blocks * d as u64 * 4) as u64);
        let centered_buf = ctx.new_buffer_uninitialized((n * d * 4) as u64);
        let gram_buf = ctx.new_buffer_uninitialized((gram_dim * gram_dim * 4) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();

        // ── Encoder 1: pca_mean ──
        let enc1 = cmd_buf.new_compute_command_encoder();
        enc1.set_compute_pipeline_state(self.pipelines.get(ctx, "pca_mean")?);
        enc1.set_buffer(0, Some(&data_buf), 0);
        enc1.set_buffer(1, Some(&means_buf), 0);
        enc1.set_buffer(2, Some(&block_sums_buf), 0);
        set_u32(&enc1, 3, n as u32);
        set_u32(&enc1, 4, d as u32);
        set_u32(&enc1, 5, num_blocks as u32);
        enc1.set_threadgroup_memory_length(0, (d * 4) as u64);
        let tg1 = MTLSize { width: d as u64, height: 1, depth: 1 };
        let grp1 = MTLSize { width: num_blocks, height: 1, depth: 1 };
        enc1.dispatch_thread_groups(grp1, tg1);
        enc1.end_encoding();

        // ── Encoder 2: pca_mean_final ──
        let enc2 = cmd_buf.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(self.pipelines.get(ctx, "pca_mean_final")?);
        enc2.set_buffer(0, Some(&block_sums_buf), 0);
        enc2.set_buffer(1, Some(&means_buf), 0);
        set_u32(&enc2, 2, num_blocks as u32);
        set_u32(&enc2, 3, d as u32);
        set_u32(&enc2, 4, n as u32);
        let tg2 = MTLSize { width: d as u64, height: 1, depth: 1 };
        let grp2 = MTLSize { width: 1, height: 1, depth: 1 };
        enc2.dispatch_thread_groups(grp2, tg2);
        enc2.end_encoding();

        // ── Encoder 3: pca_center ──
        let enc3 = cmd_buf.new_compute_command_encoder();
        enc3.set_compute_pipeline_state(self.pipelines.get(ctx, "pca_center")?);
        enc3.set_buffer(0, Some(&data_buf), 0);
        enc3.set_buffer(1, Some(&means_buf), 0);
        enc3.set_buffer(2, Some(&centered_buf), 0);
        set_u32(&enc3, 3, n as u32);
        set_u32(&enc3, 4, d as u32);
        let total = (n * d) as u64;
        let tg3 = MTLSize { width: 256, height: 1, depth: 1 };
        let grp3 = MTLSize { width: (total + 255) / 256, height: 1, depth: 1 };
        enc3.dispatch_thread_groups(grp3, tg3);
        enc3.end_encoding();

        let centered_t_buf = ctx.new_buffer_uninitialized((n * d * 4) as u64);

        // Transpose: centered(N,D) → centered_t(D,N)
        let enc4 = cmd_buf.new_compute_command_encoder();
        enc4.set_compute_pipeline_state(self.pipelines.get(ctx, "pca_transpose")?);
        enc4.set_buffer(0, Some(&centered_buf), 0);
        enc4.set_buffer(1, Some(&centered_t_buf), 0);
        set_u32(&enc4, 2, n as u32);
        set_u32(&enc4, 3, d as u32);
        let tg4 = MTLSize { width: 16, height: 16, depth: 1 };
        let grp4 = MTLSize {
            width: (d as u64 + 15) / 16,
            height: (n as u64 + 15) / 16,
            depth: 1,
        };
        enc4.dispatch_thread_groups(grp4, tg4);
        enc4.end_encoding();

        // Matmul: gram = centered_t @ centered  or  centered @ centered_t
        // gram_dim = min(N, D), inner_dim = max(N, D)
        let enc5 = cmd_buf.new_compute_command_encoder();
        enc5.set_compute_pipeline_state(self.pipelines.get(ctx, "pca_matmul")?);
        if use_gram_path {
            // gram(N,N) = centered(N,D) @ centered_t(D,N)
            enc5.set_buffer(0, Some(&centered_buf), 0);
            enc5.set_buffer(1, Some(&centered_t_buf), 0);
            set_u32(&enc5, 3, n as u32);
            set_u32(&enc5, 4, d as u32);
            set_u32(&enc5, 5, n as u32);
            set_u32(&enc5, 6, d as u32);
            set_u32(&enc5, 7, n as u32);
        } else {
            // gram(D,D) = centered_t(D,N) @ centered(N,D)
            enc5.set_buffer(0, Some(&centered_t_buf), 0);
            enc5.set_buffer(1, Some(&centered_buf), 0);
            set_u32(&enc5, 3, d as u32);
            set_u32(&enc5, 4, n as u32);
            set_u32(&enc5, 5, d as u32);
            set_u32(&enc5, 6, n as u32);
            set_u32(&enc5, 7, d as u32);
        }
        enc5.set_buffer(2, Some(&gram_buf), 0);
        let gd = gram_dim as u64;
        let tg5 = MTLSize { width: 16, height: 16, depth: 1 };
        let grp5 = MTLSize { width: (gd + 15) / 16, height: (gd + 15) / 16, depth: 1 };
        enc5.dispatch_thread_groups(grp5, tg5);
        enc5.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // ── Read means and Gram matrix back to CPU ──────────────────
        let means: Vec<f32> = ctx.read_buffer(&means_buf, d);
        self.mean = means.clone();

        let mut gram: Vec<f32> = ctx.read_buffer(&gram_buf, gram_dim * gram_dim);

        // Divide Gram by N (covariance scaling)
        let inv_n = 1.0 / n as f32;
        for v in gram.iter_mut() { *v *= inv_n; }

        // ── CPU eigendecomposition ──────────────────────────────────
        // Use Jacobi for small matrices, Accelerate LAPACK for larger ones.
        let (eigvals_sorted, eigvecs_sorted) = if gram_dim <= 128 {
            self.eigh_cpu(&gram, gram_dim)?
        } else {
            self.eigh_accelerate(&gram, gram_dim)?
        };

        // ── Extract top-K ──────────────────────────────────────────
        let m = gram_dim;
        let k_actual = k.min(m);
        let start_idx = m - k_actual;

        let mut top_vals = vec![0.0f32; k_actual];
        let mut top_vecs = vec![0.0f32; d * k_actual];

        if use_gram_path {
            // Recover D-dim eigenvectors: V = X^T @ U · diag(1/√(N·λ))
            // But X^T is on GPU. For now, do recovery on CPU with centered data.
            // We could read centered data back from GPU, but it's more efficient
            // to recompute: actually we have data and means, so center on CPU.
            // BUT: centered is on GPU and large. Let's read it back.
            let centered: Vec<f32> = ctx.read_buffer(&centered_buf, n * d);

            for j in 0..k_actual {
                let src_col = start_idx + j;
                top_vals[j] = eigvals_sorted[src_col];
                let inv_sqrt = 1.0 / (eigvals_sorted[src_col].max(1e-30) * n as f32).sqrt();
                for i in 0..d {
                    let mut sum = 0.0;
                    for r in 0..n {
                        sum += centered[r * d + i] * eigvecs_sorted[r * gram_dim + src_col];
                    }
                    top_vecs[i * k_actual + j] = sum * inv_sqrt;
                }
            }
        } else {
            for j in 0..k_actual {
                let src_col = start_idx + j;
                top_vals[j] = eigvals_sorted[src_col];
                for i in 0..d {
                    top_vecs[i * k_actual + j] = eigvecs_sorted[i * gram_dim + src_col];
                }
            }
        }

        // ── Convert to sklearn convention ───────────────────────────
        let total_var: f32 = eigvals_sorted.iter().sum();

        let mut comps = vec![0.0f32; k_actual * d];
        let mut expl_var = vec![0.0f32; k_actual];
        let mut expl_var_ratio = vec![0.0f32; k_actual];

        for j in 0..k_actual {
            let flip_j = k_actual - 1 - j;
            expl_var[j] = top_vals[flip_j];
            expl_var_ratio[j] = top_vals[flip_j] / total_var.max(1e-30);
            for i in 0..d {
                comps[j * d + i] = top_vecs[i * k_actual + flip_j];
            }
        }

        self.components = comps;
        self.explained_variance = expl_var;
        self.explained_variance_ratio = expl_var_ratio;
        self.singular_values = self.explained_variance.iter()
            .map(|v| (self.n as f32 * v).sqrt())
            .collect();
        self.noise_variance = if k_actual < m {
            eigvals_sorted[..m - k_actual].iter().sum::<f32>() / (m - k_actual) as f32
        } else {
            0.0
        };

        Ok(())
    }

    /// Project data onto principal components (GPU).
    pub fn transform(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(d == self.d,
            "Data dimension mismatch: got {}, expected {}", d, self.d);
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");
        anyhow::ensure!(!self.components.is_empty(), "PCA not fitted");

        let k = self.config.n_components;
        self.transform_gpu(ctx, data, &self.mean, &self.components, n, d, k)
    }

    pub fn fit_transform(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.fit(ctx, data, n, d)?;
        self.transform(ctx, data, n, d)
    }

    // ── GPU transform ────────────────────────────────────────────

    fn transform_gpu(
        &self, ctx: &MetalContext,
        x: &[f32], means: &[f32], components: &[f32],
        n: usize, d: usize, k: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let x_buf = ctx.new_buffer(x);
        let means_buf = ctx.new_buffer(means);
        let comps_buf = ctx.new_buffer(components);
        let out_buf = ctx.new_buffer_uninitialized((n * k * 4) as u64);

        let tg = MTLSize { width: 16, height: 16, depth: 1 };
        let grp = MTLSize {
            width: ((k as u64) + 15) / 16,
            height: ((n as u64) + 15) / 16,
            depth: 1,
        };

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipelines.get(ctx, "pca_transform")?);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&means_buf), 0);
        enc.set_buffer(2, Some(&comps_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        set_u32(&enc, 4, n as u32);
        set_u32(&enc, 5, d as u32);
        set_u32(&enc, 6, k as u32);
        enc.dispatch_thread_groups(grp, tg);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let result: Vec<f32> = ctx.read_buffer(&out_buf, n * k);
        Ok(result)
    }

    // ── CPU eigendecomposition (Accelerate LAPACK ssyevd) ─────────

    fn eigh_accelerate(&self, a: &[f32], m: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(a.len() == m * m, "Matrix size mismatch");
        let n = m as i32;
        let lda = n;
        let mut a_mat = a.to_vec();
        let mut w = vec![0.0f32; m];

        let mut info: i32 = 0;
        let jobz: u8 = b'V';  // compute eigenvalues + eigenvectors
        let uplo: u8 = b'U';  // upper triangle stored

        // Query optimal workspace size
        let mut lwork: i32 = -1;
        let mut liwork: i32 = -1;
        let mut work_size: f32 = 0.0;
        let mut iwork_size: i32 = 0;
        unsafe {
            ssyevd_(
                &jobz, &uplo, &n, a_mat.as_mut_ptr(), &lda, w.as_mut_ptr(),
                &mut work_size, &lwork, &mut iwork_size, &liwork, &mut info,
            );
        }
        lwork = work_size as i32;
        liwork = iwork_size;

        let mut work = vec![0.0f32; lwork as usize];
        let mut iwork = vec![0i32; liwork as usize];
        unsafe {
            ssyevd_(
                &jobz, &uplo, &n, a_mat.as_mut_ptr(), &lda, w.as_mut_ptr(),
                work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &liwork, &mut info,
            );
        }
        anyhow::ensure!(info == 0, "ssyevd failed with info={}", info);

        // ssyevd returns eigenvalues in ascending order, eigenvectors in columns.
        // w = eigenvalues (ascending), a_mat = eigenvectors (columns)
        let mut sorted_vecs = vec![0.0f32; m * m];
        for col in 0..m {
            for row in 0..m {
                sorted_vecs[row * m + col] = a_mat[row * m + col];
            }
        }
        Ok((w, sorted_vecs))
    }

    // ── CPU eigendecomposition (Jacobi fallback) ──────────────────

    pub fn eigh_cpu(&self, a: &[f32], m: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(a.len() == m * m, "Matrix size mismatch");
        let tol = 1e-8f32;
        let max_sweeps = 15;

        let mut a_mat = a.to_vec();
        let mut v_mat = vec![0.0f32; m * m];
        for i in 0..m { v_mat[i * m + i] = 1.0; }

        for _sweep in 0..max_sweeps {
            let mut converged = true;
            for p in 0..m {
                for q in (p + 1)..m {
                    let a_pq = a_mat[p * m + q];
                    let a_pp = a_mat[p * m + p];
                    let a_qq = a_mat[q * m + q];
                    let threshold = tol * (a_pp.abs() + a_qq.abs()) * 0.5;
                    if a_pq.abs() <= threshold { continue; }
                    converged = false;

                    let tau = (a_qq - a_pp) / (2.0 * a_pq);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        1.0 / (tau - (1.0 + tau * tau).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    let a_pp_new = c * c * a_pp + s * s * a_qq - 2.0 * c * s * a_pq;
                    let a_qq_new = s * s * a_pp + c * c * a_qq + 2.0 * c * s * a_pq;
                    a_mat[p * m + p] = a_pp_new;
                    a_mat[q * m + q] = a_qq_new;
                    a_mat[p * m + q] = 0.0;
                    a_mat[q * m + p] = 0.0;

                    for r in 0..m {
                        if r != p && r != q {
                            let a_pr = a_mat[p * m + r];
                            let a_qr = a_mat[q * m + r];
                            a_mat[p * m + r] = c * a_pr - s * a_qr;
                            a_mat[r * m + p] = a_mat[p * m + r];
                            a_mat[q * m + r] = s * a_pr + c * a_qr;
                            a_mat[r * m + q] = a_mat[q * m + r];
                        }
                        let v_rp = v_mat[r * m + p];
                        let v_rq = v_mat[r * m + q];
                        v_mat[r * m + p] = c * v_rp - s * v_rq;
                        v_mat[r * m + q] = s * v_rp + c * v_rq;
                    }
                }
            }
            if converged { break; }
        }

        let mut eigvals = vec![0.0f32; m];
        for i in 0..m { eigvals[i] = a_mat[i * m + i]; }

        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());

        let sorted_vals: Vec<f32> = indices.iter().map(|&i| eigvals[i]).collect();
        let mut sorted_vecs = vec![0.0f32; m * m];
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..m {
                sorted_vecs[row * m + new_col] = v_mat[row * m + old_col];
            }
        }

        Ok((sorted_vals, sorted_vecs))
    }
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    let ptr = std::ptr::from_ref(&value);
    encoder.set_bytes(index, len, ptr.cast());
}
