use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

// ── Accelerate LAPACK ssyevd (symmetric eigen/symmetric) ─────────

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn ssyevd_(
        jobz: *const u8,
        uplo: *const u8,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        w: *mut f32,
        work: *mut f32,
        lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        info: *mut i32,
    );
}

const SHADER_SRC: &str = include_str!("../../shaders/lda.metal");

// ── Pipeline cache ────────────────────────────────────────────────

struct PipelineCache {
    scatter: OnceLock<ComputePipelineState>,
    transform: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            scatter: OnceLock::new(),
            transform: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "lda_scatter" => &self.scatter,
            "lda_transform" => &self.transform,
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

// ── Config ────────────────────────────────────────────────────────

/// Configuration for linear discriminant analysis (LDA).
///
/// LDA is a supervised linear dimensionality-reduction / subspace method
/// (sklearn `LinearDiscriminantAnalysis`). The between-class scatter is
/// maximized relative to the within-class scatter, so the resulting
/// components are the eigendirections of `S_w⁻¹ · S_b`.
#[derive(Clone, Debug)]
pub struct LDAConfig {
    /// Number of components/axes to keep. Clamped at runtime to
    /// `min(n_classes - 1, n_features)` (the maximum informative rank).
    pub n_components: usize,
}

impl Default for LDAConfig {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

/// Linear discriminant analysis: supervised linear dimensionality reduction.
///
/// Strategy (matches the flashlib/PCA style guidance — reuse Gram/scatter):
///   - GPU: a single batched scatter kernel computes the Gram matrix `Xᵀ·X`
///     (the O(N·D²) bottleneck).
///   - GPU: `lda_transform` projects data onto the learned components.
///   - CPU: class means/counts, then the *within-class* (`S_w`) and
///     *between-class* (`S_b`) scatter matrices are assembled from `Xᵀ·X` and
///     the per-class sum-of-squares (no need to re-center every sample).
///   - CPU: solve the symmetric generalized eigenproblem via whitening —
///     `eigh(S_w)` → whiten → `eigh(Wᵀ S_b W)` — and keep the top-k axes.
pub struct LDA {
    config: LDAConfig,
    /// Raw Gram `Xᵀ·X` accumulated on the GPU (kept for diagnostics/testing).
    gram: Vec<f32>,
    /// Discriminant axes, row-major `(K, D)`. `scalings_[r][d]` is component
    /// `r` along feature `d`.
    scalings: Vec<f32>,
    /// Overall feature mean `(D,)` used to center data before projecting.
    mean: Vec<f32>,
    /// Rank-faithful number of components after clamping.
    n_components: usize,
    /// Number of samples / features.
    n: usize,
    d: usize,
    /// Discriminant eigenvalues (descending), length `n_components`.
    eigenvalues: Vec<f32>,
    /// Unique class labels in ascending order (`C,`).
    classes: Vec<f32>,
    /// Per-class counts (priors), length `C`.
    class_counts: Vec<f32>,
    /// Per-class feature means `(C, d)`.
    means: Vec<f32>,
    /// Projected class centers `(C, k)` used for `predict`.
    proj_means: Vec<f32>,
    pipelines: PipelineCache,
}

impl LDA {
    pub fn new(config: LDAConfig) -> Self {
        Self {
            config,
            gram: Vec::new(),
            scalings: Vec::new(),
            mean: Vec::new(),
            n_components: 0,
            n: 0,
            d: 0,
            eigenvalues: Vec::new(),
            classes: Vec::new(),
            class_counts: Vec::new(),
            means: Vec::new(),
            proj_means: Vec::new(),
            pipelines: PipelineCache::new(),
        }
    }

    pub fn scalings(&self) -> &[f32] {
        &self.scalings
    }
    pub fn coeffs(&self) -> &[f32] {
        &self.scalings
    }
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }
    pub fn eigenvalues(&self) -> &[f32] {
        &self.eigenvalues
    }
    pub fn classes(&self) -> &[f32] {
        &self.classes
    }
    pub fn class_counts(&self) -> &[f32] {
        &self.class_counts
    }
    pub fn class_means(&self) -> &[f32] {
        &self.means
    }
    pub fn n_components(&self) -> usize {
        self.n_components
    }
    pub fn n_features(&self) -> usize {
        self.d
    }
    pub fn n_samples(&self) -> usize {
        self.n
    }

    /// Fit LDA on the GPU.
    ///
    /// * `data` — flat row-major `n × d` feature matrix.
    /// * `labels` — class label per sample, shape `(n,)`.
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        labels: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(n > 0 && d > 0, "Data must be non-empty");
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");
        anyhow::ensure!(labels.len() == n, "Labels length mismatch");

        self.n = n;
        self.d = d;

        // ── Class statistics (host side; O(N·D + C·D)) ───────────
        // Distinct classes in ascending order.
        let mut classes: Vec<f32> = labels.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);
        anyhow::ensure!(classes.len() >= 2, "LDA needs at least 2 classes");
        let c = classes.len();

        let mut sums = vec![0.0f64; c * d]; // class feature sums
        let mut counts = vec![0.0f64; c];
        for i in 0..n {
            let idx = classes
                .iter()
                .position(|&cl| (cl - labels[i]).abs() <= 1e-9)
                .ok_or_else(|| anyhow::anyhow!("internal: label not found"))?;
            counts[idx] += 1.0;
            for jj in 0..d {
                sums[idx * d + jj] += data[i * d + jj] as f64;
            }
        }
        let mut class_means = vec![0.0f32; c * d];
        for idx in 0..c {
            for jj in 0..d {
                class_means[idx * d + jj] = (sums[idx * d + jj] / counts[idx]) as f32;
            }
        }
        // Overall mean.
        let mut overall = vec![0.0f64; d];
        for jj in 0..d {
            let s: f64 = sums.iter().skip(jj).step_by(d).sum::<f64>();
            overall[jj] = s / n as f64;
        }
        let gmean: Vec<f32> = overall.iter().map(|&v| v as f32).collect();

        // ── GPU: batch scatter → XᵀX (D,D) ───────────────────────
        let x_buf = ctx.new_buffer(data);
        let gram_buf = ctx.new_buffer_uninitialized((d * d * 4) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipelines.get(ctx, "lda_scatter")?);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&gram_buf), 0);
        set_u32(&enc, 2, n as u32);
        set_u32(&enc, 3, d as u32);
        let tg = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let grp = MTLSize {
            width: ((d as u64) + 15) / 16,
            height: ((d as u64) + 15) / 16,
            depth: 1,
        };
        enc.dispatch_thread_groups(grp, tg);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let gram: Vec<f32> = ctx.read_buffer(&gram_buf, d * d);
        self.gram = gram.clone();

        // ── Assemble S_w and S_b (D×D) on the host ───────────────
        // S_w = XᵀX − Σ_c (1/n_c)·(sum_c ⊗ sum_c)
        // S_b = Σ_c (1/n_c)·(sum_c ⊗ sum_c) − n·(mean ⊗ mean)
        let mut sw = gram.clone();
        let mut sb = vec![0.0f32; d * d];
        for idx in 0..c {
            let inv = 1.0f64 / counts[idx];
            for p in 0..d {
                for q in 0..d {
                    let val = (sums[idx * d + p] * sums[idx * d + q] * inv) as f32;
                    sw[p * d + q] -= val;
                    sb[p * d + q] += val;
                }
            }
        }
        for p in 0..d {
            for q in 0..d {
                sb[p * d + q] -= (n as f64 * overall[p] * overall[q]) as f32;
            }
        }
        symmetrize(&mut sw, d);
        symmetrize(&mut sb, d);

        let max_k = c.saturating_sub(1).min(d);
        let k = self.config.n_components.min(max_k).max(1);
        self.n_components = k;

        // ── Solve S_w⁻¹ S_b via whitening ────────────────────────
        let (scalings, eigenvalues) = solve_generalized_eigen(&sw, &sb, k, d)?;

        self.scalings = scalings;
        self.eigenvalues = eigenvalues;
        self.mean = gmean.clone();
        self.classes = classes.iter().map(|&v| v as f32).collect();
        self.class_counts = counts.iter().map(|&v| v as f32).collect();
        self.means = class_means;

        // Projected class centers in the new space (C,k).
        let mut proj = vec![0.0f32; c * k];
        for idx in 0..c {
            for r in 0..k {
                let mut s = 0.0f32;
                for jj in 0..d {
                    s += (self.means[idx * d + jj] - gmean[jj]) * self.scalings[r * d + jj];
                }
                proj[idx * k + r] = s;
            }
        }
        self.proj_means = proj;

        Ok(())
    }

    /// Project `data` (n × d) into the discriminant subspace → (n, k).
    pub fn transform(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            d == self.d,
            "Data dimension mismatch: got {}, expected {}",
            d,
            self.d
        );
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");
        anyhow::ensure!(!self.scalings.is_empty(), "LDA not fitted");

        let k = self.n_components;
        let x_buf = ctx.new_buffer(data);
        let mean_buf = ctx.new_buffer(&self.mean);
        let scale_buf = ctx.new_buffer(&self.scalings);
        let out_buf = ctx.new_buffer_uninitialized((n * k * 4) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipelines.get(ctx, "lda_transform")?);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&mean_buf), 0);
        enc.set_buffer(2, Some(&scale_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        set_u32(&enc, 4, n as u32);
        set_u32(&enc, 5, d as u32);
        set_u32(&enc, 6, k as u32);
        let tg = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let grp = MTLSize {
            width: ((k as u64) + 15) / 16,
            height: ((n as u64) + 15) / 16,
            depth: 1,
        };
        enc.dispatch_thread_groups(grp, tg);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(ctx.read_buffer(&out_buf, n * k))
    }

    pub fn fit_transform(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        labels: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.fit(ctx, data, labels, n, d)?;
        self.transform(ctx, data, n, d)
    }

    /// Predict class indices (0..C-1) by nearest projected class center.
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<usize>> {
        anyhow::ensure!(!self.scalings.is_empty(), "LDA not fitted");
        let c = self.classes.len();
        let k = self.n_components;
        let proj = self.transform(ctx, data, n, d)?;

        let mut preds = vec![0usize; n];
        for i in 0..n {
            let mut best = usize::MAX;
            let mut best_d = f32::INFINITY;
            for idx in 0..c {
                let mut s = 0.0f32;
                for r in 0..k {
                    let diff = proj[i * k + r] - self.proj_means[idx * k + r];
                    s += diff * diff;
                }
                if s < best_d {
                    best_d = s;
                    best = idx;
                }
            }
            preds[i] = best;
        }
        Ok(preds)
    }

    /// Accuracy on the given labeled data against the fitted classes.
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        labels: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(labels.len() == n, "Labels length mismatch");
        let preds = self.predict(ctx, data, n, d)?;
        let mut correct = 0usize;
        for i in 0..n {
            let idx = self
                .classes
                .iter()
                .position(|&cl| (cl - labels[i]).abs() <= 1e-9)
                .ok_or_else(|| anyhow::anyhow!("label {} not seen in fit", labels[i]))?;
            if preds[i] == idx {
                correct += 1;
            }
        }
        Ok(correct as f32 / n as f32)
    }
}

// ── Generalized eigenproblem (whitening) ─────────────────────────

/// Solve the LDA generalized eigenproblem `S_w·v = λ·S_b·v` (equivalently
/// the top-`k` axes of `S_w⁻¹·S_b`) via symmetric whitening:
///  1. `eigh(S_w)` = `V Λ Vᵀ`; whiten `W = V·diag(Λ⁻½)` so `Wᵀ S_w W = I`.
///  2. `eigh(Wᵀ S_b W)` → keep the largest-`k` eigenvectors.
///  3. Un-whiten: component `= W·q`.
/// The returned matrix is row-major `(k, d)`.
fn solve_generalized_eigen(
    sw: &[f32],
    sb: &[f32],
    k: usize,
    d: usize,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    // Add a small ridge so a rank-deficient within-scatter stays invertible.
    let mut sw_r = sw.to_vec();
    let trace: f32 = (0..d).map(|i| sw[i * d + i]).sum();
    let eps = 1e-7f32 * (1.0 + trace.abs() / d as f32);
    for i in 0..d {
        sw_r[i * d + i] += eps;
    }

    // 1) eigh(S_w): vals_w ascending, vecs_w rows.
    let (vals_w, vec_w) = eigh_sym(&sw_r, d)?;
    let floor = vals_w[d - 1].max(1e-12) * 1e-6; // smallest allowed sqrt-scale

    // Whiten matrix W (d,d): W[j,i] = vec_w[i,j]/sqrt(max(vals_w[i], floor)).
    let mut whit = vec![0.0f32; d * d];
    for j in 0..d {
        for i in 0..d {
            let val = vals_w[i].max(floor).sqrt();
            whit[j * d + i] = vec_w[i * d + j] / val;
        }
    }

    // 2) A = Wᵀ S_b W  (d,d), symmetric.
    let mut a = vec![0.0f32; d * d];
    for p in 0..d {
        for q in 0..d {
            let mut acc = 0.0f32;
            for r in 0..d {
                for s in 0..d {
                    acc += whit[r * d + p] * sb[r * d + s] * whit[s * d + q];
                }
            }
            a[p * d + q] = acc;
        }
    }
    symmetrize(&mut a, d);

    // 3) eigh(A): vals asc, vec rows = eigen-directions in whitened space.
    let (vals_a, vec_a) = eigh_sym(&a, d)?;

    // 4) Scale back: scalings[r] = W @ q where q = vec_a[largest r].
    let mut scalings = vec![0.0f32; k * d];
    // Iterate source rows from largest eigenvalue to smallest so the output
    // rows are ordered by descending discriminative power.
    for (out_r, src_r) in (d - k..d).rev().enumerate() {
        for j in 0..d {
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += whit[j * d + i] * vec_a[src_r * d + i];
            }
            scalings[out_r * d + j] = acc;
        }
    }

    // Eigenvalues (descending, aligned with the output component rows).
    let evals: Vec<f32> = (d - k..d).map(|r| vals_a[r]).rev().collect();

    Ok((scalings, evals))
}

/// Symmetric eigendecomposition via Accelerate LAPACK `ssyevd`.
/// Returns `(eigenvalues ascending, eigenvectors)` where the eigenvector for
/// eigenvalue `r` is stored as row `r` of `vecs` (row-major `m×m`).
fn eigh_sym(a: &[f32], m: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    anyhow::ensure!(a.len() == m * m, "Matrix size mismatch");
    let n = m as i32;
    let lda = n;
    let mut a_mat = a.to_vec();
    let mut w = vec![0.0f32; m];

    let mut info: i32 = 0;
    let jobz: u8 = b'V';
    let uplo: u8 = b'U';

    let mut lwork: i32 = -1;
    let mut liwork: i32 = -1;
    let mut work_size: f32 = 0.0;
    let mut iwork_size: i32 = 0;
    unsafe {
        ssyevd_(
            &jobz,
            &uplo,
            &n,
            a_mat.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            &mut work_size,
            &lwork,
            &mut iwork_size,
            &liwork,
            &mut info,
        );
    }
    lwork = work_size as i32;
    liwork = iwork_size;

    let mut work = vec![0.0f32; lwork as usize];
    let mut iwork = vec![0i32; liwork as usize];
    unsafe {
        ssyevd_(
            &jobz,
            &uplo,
            &n,
            a_mat.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }
    anyhow::ensure!(info == 0, "ssyevd failed with info={}", info);

    // w ascending; a_mat eigenvectors in columns → store as rows.
    let mut vecs = vec![0.0f32; m * m];
    for col in 0..m {
        for row in 0..m {
            vecs[row * m + col] = a_mat[row * m + col];
        }
    }
    Ok((w, vecs))
}

fn symmetrize(a: &mut [f32], d: usize) {
    for i in 0..d {
        for j in i..d {
            let v = 0.5 * (a[i * d + j] + a[j * d + i]);
            a[i * d + j] = v;
            a[j * d + i] = v;
        }
    }
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    let ptr = std::ptr::from_ref(&value);
    encoder.set_bytes(index, len, ptr.cast());
}
