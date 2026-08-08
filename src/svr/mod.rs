//! Support Vector Regression (SVR) with GPU-accelerated kernel evaluation.
//!
//! Mirrors scikit-learn's `sklearn.svm.SVR` (ε-insensitive, RBF by default)
//! while keeping the two heavy linear-algebra stages on the GPU and the
//! per-iteration scalar updates on the host:
//!
//! 1. **Kernel (Gram) matrix (GPU)** — the exact same `svm_kernel`
//!    (`shaders/svm.metal`) computes the full (n × n) Gram matrix
//!    `K[i][j] = κ(x_i, x_j)` in a single launch (depends only on the data).
//! 2. **Host SMO (ε-SVR)** — a Sequential Optimizer on the dual coefficients
//!    `β_i = α_i⁺ - α_i⁻ ∈ [-C, C]` with the sum constraint `Σ_i β_i = 0`.
//!    Each iteration refines a pair (i, j) to exactly maximize the
//!    two-variable quadratic (including the non-smooth ε-tube penalty) via the
//!    Gram `K` as the fast κ lookup, then recovers a single intercept `b`.
//! 3. **Decision (GPU)** — `svm_predict` returns the regression outputs
//!    `f(x_m) = b + Σ_s β_s·κ(x_m, x_s)` over the pooled support vectors in a
//!    single launch (K = 1). SVR is a single real-valued regressor, so
//!    `predict` and `decision_function` are the same.

use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

pub use crate::svm::SVCKernel;

// The shader kernels `svm_kernel` (Gram build) and `svm_predict` (regression
// outputs) are shared with SVC — SVR reuses both verbatim.
const SHADER_SRC: &str = include_str!("../../shaders/svm.metal");

// ── Pipeline cache ────────────────────────────────────────────────────────

struct PipelineCache {
    kernel: OnceLock<ComputePipelineState>,
    predict: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            kernel: OnceLock::new(),
            predict: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "svm_kernel" => &self.kernel,
            "svm_predict" => &self.predict,
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

// ── Config ────────────────────────────────────────────────────────────────

/// Configuration for the Support Vector Regressor.
#[derive(Clone, Debug)]
pub struct SVRConfig {
    /// Kernel type used for `κ(x, y)` (default `Rbf`).
    pub kernel: SVCKernel,
    /// Manual kernel width `gamma`. Values `<= 0` select the automatic default
    /// `1 / n_features` (the `1/n_features` fallback of `gamma="scale"`).
    pub gamma: f32,
    /// Polynomial kernel degree (only used when `kernel == Poly`).
    pub degree: f32,
    /// Independent term in poly / sigmoid kernels (default 0.0).
    pub coef0: f32,
    /// Regularization parameter `C` (default 1.0).
    pub c: f32,
    /// ε-insensitive tube width: residuals within `eps` cost nothing (default 0.1).
    pub eps: f32,
    /// SMO convergence tolerance (default 1e-3).
    pub tolerance: f32,
    /// Maximum number of SMO passes (default 200).
    pub max_iter: usize,
    /// Seed for the SMO index sampling (reproducible).
    pub seed: u64,
}

impl Default for SVRConfig {
    fn default() -> Self {
        Self {
            kernel: SVCKernel::Rbf,
            gamma: 0.0, // auto: 1 / n_features
            degree: 3.0,
            coef0: 0.0,
            c: 1.0,
            eps: 0.1,
            tolerance: 1e-3,
            max_iter: 200,
            seed: 42,
        }
    }
}

// ── Operator ──────────────────────────────────────────────────────────────

/// Support Vector Regressor fitted with a GPU Gram matrix + host SMO.
pub struct SVR {
    config: SVRConfig,
    /// Resolved kernel width (auto `1 / n_features` becomes concrete here).
    gamma: f32,
    d: usize,
    n: usize,
    /// Dual coefficients `β_i = α_i⁺ - α_i⁻ ∈ [-C, C]` for every training row
    /// (support vectors are the rows with `|β_i| > 0`).
    beta: Vec<f32>,
    /// Original training indices of the support vectors.
    sv_idx: Vec<usize>,
    /// Regression intercept `b`.
    intercept: f32,
    /// Pooled support vectors for the GPU decision kernel, flat (ns × d).
    pooled_sv: Vec<f32>,
    /// Pooled dual coefficients `β` aligned to `pooled_sv` (ns,).
    pooled_beta: Vec<f32>,
    /// Prefix offsets into `pooled_sv`/`pooled_beta` for the GPU kernel. SVR
    /// uses a single classifier, so this is always `[0, ns]`.
    offsets: Vec<u32>,
    /// SMO passes actually run.
    n_iter_: usize,
    pipelines: PipelineCache,
}

impl SVR {
    pub fn new(config: SVRConfig) -> Self {
        Self {
            config,
            gamma: 0.0,
            d: 0,
            n: 0,
            beta: Vec::new(),
            sv_idx: Vec::new(),
            intercept: 0.0,
            pooled_sv: Vec::new(),
            pooled_beta: Vec::new(),
            offsets: Vec::new(),
            n_iter_: 0,
            pipelines: PipelineCache::new(),
        }
    }

    pub fn n_features(&self) -> usize {
        self.d
    }
    pub fn n_samples(&self) -> usize {
        self.n
    }
    pub fn gamma(&self) -> f32 {
        self.gamma
    }
    pub fn intercept(&self) -> f32 {
        self.intercept
    }
    /// Number of support vectors.
    pub fn support_count(&self) -> usize {
        self.pooled_beta.len()
    }
    /// Flattened pooled support vectors, row-major `(ns × n_features)`.
    pub fn support_vectors(&self) -> &[f32] {
        &self.pooled_sv
    }
    /// Pooled dual coefficients `β = α⁺ − α⁻` aligned to `support_vectors`.
    pub fn dual_coef(&self) -> &[f32] {
        &self.pooled_beta
    }
    /// SMO passes actually run.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Fit the regressor with GPU Gram + host SMO.
    ///
    /// * `ctx` - Metal context
    /// * `data` - Flat row-major training data (n × d)
    /// * `y` - Regression targets, one per row (n,).
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(n > 0 && d > 0, "Data must be non-empty");
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        anyhow::ensure!(y.len() == n, "Target length mismatch");
        anyhow::ensure!(
            self.config.c > 0.0 && self.config.c.is_finite(),
            "C must be a finite, positive value"
        );
        anyhow::ensure!(
            self.config.eps >= 0.0 && self.config.eps.is_finite(),
            "eps must be a finite, non-negative value"
        );
        anyhow::ensure!(
            self.config.tolerance > 0.0 && self.config.tolerance.is_finite(),
            "tolerance must be a finite, positive value"
        );
        anyhow::ensure!(self.config.max_iter > 0, "max_iter must be > 0");

        // Resolve gamma: explicit if positive, else auto 1/n_features.
        let gamma = if self.config.gamma > 0.0 && self.config.gamma.is_finite() {
            self.config.gamma
        } else {
            1.0 / (d as f32)
        };

        // One GPU launch: the full n×n Gram matrix.
        let gram = self.gram(ctx, data, n, d, gamma)?;

        // Host ε-SVR SMO → (β, intercept, iters).
        let (beta, b, iters) = self.smo(&gram, y, n);

        // Support Vectors: |β_i| beyond numerical noise.
        let sv_idx: Vec<usize> = (0..n).filter(|&i| beta[i].abs() > 1e-5).collect();

        let mut pooled_sv = Vec::new();
        let mut pooled_beta = Vec::new();
        for &i in &sv_idx {
            pooled_sv.extend_from_slice(&data[i * d..(i + 1) * d]);
            pooled_beta.push(beta[i]);
        }
        let offsets = vec![0u32, pooled_beta.len() as u32];

        self.gamma = gamma;
        self.d = d;
        self.n = n;
        self.beta = beta;
        self.sv_idx = sv_idx;
        self.intercept = b;
        self.pooled_sv = pooled_sv;
        self.pooled_beta = pooled_beta;
        self.offsets = offsets;
        self.n_iter_ = iters;
        Ok(())
    }

    // ── GPU: Gram (kernel) matrix ──────────────────────────────────────────

    fn gram(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
        gamma: f32,
    ) -> anyhow::Result<Vec<f32>> {
        let p = self.pipelines.get(ctx, "svm_kernel")?.clone();
        let out_len = n * n;
        let data_buf = ctx.new_buffer(data);
        let out_buf = ctx.new_buffer_uninitialized((out_len * std::mem::size_of::<f32>()) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        set_u32(&enc, 2, n as u32);
        set_u32(&enc, 3, d as u32);
        set_u32(&enc, 4, self.config.kernel.as_u32());
        set_f32(&enc, 5, gamma);
        set_f32(&enc, 6, self.config.degree);
        set_f32(&enc, 7, self.config.coef0);
        const TG: u64 = 16;
        let groups = ((n as u64) + TG - 1) / TG;
        enc.dispatch_thread_groups(
            MTLSize {
                width: groups,
                height: groups,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: TG,
                depth: 1,
            },
        );
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(ctx.read_buffer::<f32>(&out_buf, out_len))
    }

    // ── Host: ε-SVR dual optimization (SMO) ────────────────────────────────

    /// Solve the ε-insensitive SVR dual with a pair-SMO over consecutive
    /// coordinate pairs, using the Gram `K` as the κ lookup.
    ///
    /// The objective is the concave `D(β) = -½ βᵀKβ + βᵀy - ε·Σ|β_i|` over
    /// `β_i ∈ [-C, C]`, `Σβ = 0`. Every inner iteration picks two indices
    /// `i, j` and, keeping `β_i + β_j = S` fixed, sets the new `β_i` to the
    /// exact maximizer of the two-variable restriction (evaluated among the
    /// interval boundaries, the ε-smooth breakpoints at 0 and `S`, and the
    /// smooth quadratic optimum), which is guaranteed to never decrease the
    /// objective. Returns `(β, intercept, iters)`.
    fn smo(&self, gram: &[f32], y: &[f32], n: usize) -> (Vec<f32>, f32, usize) {
        let c = self.config.c;
        let eps = self.config.eps;
        let max_pass = self.config.max_iter;
        let tiny = 1e-5f32;
        let mut rng = fastrand::Rng::with_seed(self.config.seed);

        // Dense incremental state: `g[i] = Σ_j K[j][i]·β_j` (i.e. the
        // K·β product) plus the coefficient vector itself.
        let mut beta = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];

        let mut passes = 0usize;
        let mut iter = 0usize;
        while passes < max_pass {
            let mut num_changed = 0usize;
            for i in 0..n {
                // Second index j != i.
                let j = if n == 1 {
                    i
                } else {
                    let mut jj = rng.usize(0..n);
                    while jj == i {
                        jj = rng.usize(0..n);
                    }
                    jj
                };

                let s = beta[i] + beta[j]; // conserved sum
                let low = 0.0f32.max(s - c);
                let high = c.min(s + c);
                if (high - low).abs() < 1e-12 {
                    continue;
                }

                let eta = gram[i * n + i] + gram[j * n + j] - 2.0 * gram[i * n + j];
                if eta < tiny {
                    continue;
                }

                let bi = beta[i];
                let gi = g[i];
                let gj = g[j];
                let yi = y[i];
                let yj = y[j];

                // Smooth (ε-free) optimum of the quadratic: β_i' = bi + δ*,
                // δ* = ((y_i - g_i) - (y_j - g_j)) / η. (Inside the tube
                // region 0 < a < s the ±ε signs cancel, so this is also the
                // exact interior maximizer of the two-variable restriction.)
                let t_star = ((yi - gi) - (yj - gj)) / eta;
                let a_unc = (bi + t_star).max(low).min(high);

                // Candidate maximizers: boundaries, ε-smooth breakpoints
                // (β_i' = 0 and β_i' = s), unconstrained optimum, and the
                // current value (so a move never decreases the objective).
                let mut best_a = bi;
                let mut best = pair_q(bi, bi, gi, gj, yi, yj, eta, eps, s);
                for a in [low, high, 0.0, s, a_unc] {
                    if a < low - tiny || a > high + tiny {
                        continue;
                    }
                    let a = a.max(low).min(high);
                    let q = pair_q(a, bi, gi, gj, yi, yj, eta, eps, s);
                    if q > best {
                        best = q;
                        best_a = a;
                    }
                }

                let delta = best_a - bi;
                if delta.abs() < tiny {
                    continue;
                }

                // Apply: β_i → best_a, β_j → s - best_a, and update the K·β product.
                beta[i] = best_a;
                beta[j] = s - best_a;
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    g[k] += delta * (gram[k * n + i] - gram[k * n + j]);
                }
                num_changed += 1;
            }
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
            iter += 1;
        }

        let b = self.intercept_from(&g, y, &beta, n, c);
        (beta, b, iter)
    }

    /// Mean within-margin prediction used for the intercept: `b = mean(y - Kβ)`
    /// over the "free" support vectors (not pinned at the C+ε bound), falling
    /// back to all support vectors when none are free.
    fn intercept_from(&self, g: &[f32], y: &[f32], beta: &[f32], n: usize, c: f32) -> f32 {
        let free: Vec<usize> = (0..n)
            .filter(|&i| beta[i].abs() > 1e-5 && beta[i].abs() < c - 1e-5)
            .collect();
        let sel: Vec<usize> = if free.is_empty() {
            (0..n).filter(|&i| beta[i].abs() > 1e-5).collect()
        } else {
            free
        };
        if sel.is_empty() {
            return 0.0;
        }
        sel.iter().map(|&i| y[i] - g[i]).sum::<f32>() / sel.len() as f32
    }

    // ── GPU: regression outputs ────────────────────────────────────────────

    /// Raw regression values `f(x) = b + Σ_s β_s·κ(x_m, x_s)` over the pooled
    /// support vectors; one GPU launch (K = 1, so the output length is `n`).
    pub fn decision_function(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(self.n > 0, "Model is not fitted; call fit() first");
        anyhow::ensure!(d == self.d, "Feature dimension mismatch");
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");
        let m = n;
        let ns = self.pooled_beta.len() as u32;
        let out_len = m; // K = 1
        let p = self.pipelines.get(ctx, "svm_predict")?.clone();
        let x_buf = ctx.new_buffer(data); // (m, d)
        let s_buf = ctx.new_buffer(&self.pooled_sv); // (ns, d)
        let dual_buf = ctx.new_buffer(&self.pooled_beta); // (ns)
        let bias_buf = ctx.new_buffer(&[self.intercept]); // (1)
        let off_buf = ctx.new_buffer(&self.offsets); // (2)
        let dec_buf = ctx.new_buffer_uninitialized((out_len * std::mem::size_of::<f32>()) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&s_buf), 0);
        enc.set_buffer(2, Some(&dual_buf), 0);
        enc.set_buffer(3, Some(&bias_buf), 0);
        enc.set_buffer(4, Some(&off_buf), 0);
        enc.set_buffer(5, Some(&dec_buf), 0);
        set_u32(&enc, 6, m as u32);
        set_u32(&enc, 7, 1); // K = 1
        set_u32(&enc, 8, ns);
        set_u32(&enc, 9, self.d as u32);
        set_u32(&enc, 10, self.config.kernel.as_u32());
        set_f32(&enc, 11, self.gamma);
        set_f32(&enc, 12, self.config.degree);
        set_f32(&enc, 13, self.config.coef0);
        const TG: u64 = 256;
        let groups = ((out_len as u64) + TG - 1) / TG;
        enc.dispatch_thread_groups(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(ctx.read_buffer::<f32>(&dec_buf, out_len))
    }

    /// Regression predictions `f(x)` for each row (same as the decision
    /// function; SVR is a single real-valued regressor).
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.decision_function(ctx, data, n, d)
    }

    /// R² coefficient of determination `1 - SS_res / SS_tot` (clipped to 0
    /// when the model scores no better than predicting the target mean).
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(y.len() == n, "Target length mismatch");
        let preds = self.predict(ctx, data, n, d)?;
        let mean = y.iter().sum::<f32>() / n as f32;
        let mut ss_res = 0.0f32;
        let mut ss_tot = 0.0f32;
        for i in 0..n {
            let r = y[i] - preds[i];
            ss_res += r * r;
            let t = y[i] - mean;
            ss_tot += t * t;
        }
        if ss_tot == 0.0 {
            return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
        }
        Ok((1.0 - ss_res / ss_tot).max(0.0))
    }
}

/// Two-variable benefit at candidate `a` (new β_i; then β_j = s - a), where
/// `delta = a - bi`. This is the change `Q(a)` to the dual objective for the
/// (i, j) pair, ignoring the terms constant in `a`:
///   Q(δ) = -½η·δ² + (y_i - y_j)·δ - (g_i - g_j)·δ - ε·(|a| + |s - a|)
/// with `δ = a - bi`, `η = K_ii + K_jj - 2K_ij`.
#[inline]
fn pair_q(a: f32, bi: f32, gi: f32, gj: f32, yi: f32, yj: f32, eta: f32, eps: f32, s: f32) -> f32 {
    let delta = a - bi;
    -0.5 * eta * delta * delta - (gi - gj) * delta + (yi - yj) * delta
        - eps * (a.abs() + (s - a).abs())
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(index, 4, &value as *const u32 as *const std::ffi::c_void);
}

fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    encoder.set_bytes(index, 4, &value as *const f32 as *const std::ffi::c_void);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::MetalContext;

    fn rng_data_y(seed: u64, n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut data = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut v = 0.5f32;
            for dim in 0..d {
                let xv = rng.f32() * 2.0 - 1.0;
                data[i * d + dim] = xv;
                v += (dim as f32 + 1.0) * xv;
            }
            y[i] = v;
        }
        (data, y)
    }

    #[test]
    fn svr_linear_fit() {
        let ctx = MetalContext::new().unwrap();
        let n = 120;
        let d = 3;
        let (data, y) = rng_data_y(7, n, d);
        let mut m = SVR::new(SVRConfig {
            kernel: SVCKernel::Linear,
            gamma: 1.0,
            c: 100.0,
            eps: 0.05,
            max_iter: 300,
            ..Default::default()
        });
        m.fit(&ctx, &data, &y, n, d).unwrap();
        let r2 = m.score(&ctx, &data, &y, n, d).unwrap();
        assert!(
            r2 > 0.95,
            "linear SVR should capture the linear target, got R²={}",
            r2
        );
        assert!(m.support_count() > 0 && m.support_count() <= n);
    }

    #[test]
    fn svr_rbf_nonlinear() {
        let ctx = MetalContext::new().unwrap();
        // 1-D noisy sine: not linearly separable — RBF must fit it.
        let n = 150;
        let d = 1;
        let mut rng = fastrand::Rng::with_seed(11);
        let mut data = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let x = rng.f32() * 6.28f32;
            data[i] = x;
            y[i] = x.sin();
        }
        let mut m = SVR::new(SVRConfig {
            kernel: SVCKernel::Rbf,
            gamma: 0.6,
            c: 5.0,
            eps: 0.05,
            max_iter: 300,
            ..Default::default()
        });
        m.fit(&ctx, &data, &y, n, d).unwrap();
        let r2 = m.score(&ctx, &data, &y, n, d).unwrap();
        assert!(
            r2 > 0.9,
            "RBF SVR should fit a smooth sinusoid, got R²={}",
            r2
        );
        // Predictions are bounded within the target range (no huge overshoot).
        let preds = m.predict(&ctx, &data, n, d).unwrap();
        for &p in &preds {
            assert!(
                p.is_finite() && (-2.0..=2.0).contains(&p),
                "prediction {} out of expected range",
                p
            );
        }
    }

    #[test]
    fn svr_invalid_inputs() {
        let ctx = MetalContext::new().unwrap();
        let (data, y) = rng_data_y(1, 10, 2);
        let mut m = SVR::new(SVRConfig::default());
        assert!(m.fit(&ctx, &data, &y[..5], 10, 2).is_err());
        assert!(m.fit(&ctx, &data, &y, 12, 2).is_err());
        assert!(m.fit(&ctx, &data, &y, 10, 2).is_ok());
        // Unused was the *regression* target: y length mismatch handled above.
    }

    #[test]
    fn svr_deterministic_seed() {
        let ctx = MetalContext::new().unwrap();
        let (data, y) = rng_data_y(5, 60, 2);
        let fit = |seed: u64, iters: usize| -> Vec<f32> {
            let mut m = SVR::new(SVRConfig {
                kernel: SVCKernel::Rbf,
                gamma: 1.0,
                c: 2.0,
                eps: 0.1,
                seed,
                max_iter: iters,
                ..Default::default()
            });
            m.fit(&ctx, &data, &y, 60, 2).unwrap();
            m.predict(&ctx, &data, 60, 2).unwrap()
        };
        let a = fit(1234, 100);
        let b = fit(1234, 100);
        assert_eq!(a, b, "same seed must reproduce identical predictions");
    }
}
