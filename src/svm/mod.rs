//! Support Vector Classification (SVC) with GPU-accelerated kernel evaluation.
//!
//! Mirrors scikit-learn's `sklearn.svm.SVC` (one-vs-rest, RBF by default)
//! while keeping the two heavy linear-algebra stages on the GPU and the
//! per-iteration scalar updates on the host:
//!
//! 1. **Kernel (Gram) matrix (GPU)** — `svm_kernel` (`shaders/svm.metal`)
//!    computes the full (n × n) Gram matrix `K[i][j] = κ(x_i, x_j)` in a
//!    single launch for the configured kernel (linear / poly / RBF / sigmoid).
//!    Because the Gram depends only on the *data* (not the labels), the same
//!    matrix is reused verbatim by every one-vs-rest binary sub-problem.
//! 2. **Sequential Minimal Optimization (host SMO)** — for every unique class
//!    `cc ∈ classes`, a binary classifier is fit with the simplified Platt SMO
//!    using `K` as the fast 2nd-order κ lookup: an O(n²)/pass scan updates the
//!    dual coefficients `α_i` and the intercept `b` until a tolerance/max-iter
//!    limit. The per-problem `α·y` duals and intercepts are stored.
//! 3. **Decision (GPU)** — `svm_predict` computes the raw decision matrix
//!    `dec[m][kk] = b_kk + Σ_d α_d·y_d·κ(x_m, SV_d)` over the pooled support
//!    vectors in one launch. The host maps it to hard labels (argmax over the
//!    one-vs-rest classifiers) or exposes it directly as `decision_function`.
//!
//! Outputs: original class labels `classes`, per-class support indices, pooled
//! support vectors / duals / intercepts, and per-class SMO iteration counts.

use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/svm.metal");

// ── Kernel type ───────────────────────────────────────────────────────────

/// Kernel type for SVC (must match the ids in `shaders/svm.metal`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SVCKernel {
    Linear,
    Poly,
    Rbf,
    Sigmoid,
}

impl SVCKernel {
    pub fn as_u32(self) -> u32 {
        match self {
            SVCKernel::Linear => 0,
            SVCKernel::Poly => 1,
            SVCKernel::Rbf => 2,
            SVCKernel::Sigmoid => 3,
        }
    }
}

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

/// Configuration for the Support Vector Classifier.
#[derive(Clone, Debug)]
pub struct SVCConfig {
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
    /// SMO convergence tolerance (default 1e-3).
    pub tolerance: f32,
    /// Maximum number of SMO passes (default 200).
    pub max_iter: usize,
    /// Seed for the SMO index sampling (reproducible).
    pub seed: u64,
}

impl Default for SVCConfig {
    fn default() -> Self {
        Self {
            kernel: SVCKernel::Rbf,
            gamma: 0.0, // auto: 1 / n_features
            degree: 3.0,
            coef0: 0.0,
            c: 1.0,
            tolerance: 1e-3,
            max_iter: 200,
            seed: 42,
        }
    }
}

// ── Operator ──────────────────────────────────────────────────────────────

/// Support Vector Classifier fitted with a GPU Gram matrix + host SMO.
pub struct SVC {
    config: SVCConfig,
    /// Resolved kernel width (auto `1 / n_features` becomes concrete here).
    gamma: f32,
    /// Unique class values in the order they appear in `fit` (k,).
    classes: Vec<f32>,
    k: usize,
    d: usize,
    n: usize,
    /// Per-class (one-vs-rest, index 0..k-1) SMO solutions:
    /// `sv_idx[c]` are original training indices of the support vectors for
    /// classifier `c`; `dual[c]` holds the aligned `α_s · y_s` per support.
    sv_idx: Vec<Vec<usize>>,
    dual: Vec<Vec<f32>>,
    /// Per-class intercept `b` (k,).
    intercept: Vec<f32>,
    /// Pooled support vectors for the GPU decision kernel, flat (ns × d).
    pooled_sv: Vec<f32>,
    /// Pooled duals `α_s·y_s`, aligned to `pooled_sv` (ns,).
    pooled_dual: Vec<f32>,
    /// Prefix offsets into `pooled_sv` / `pooled_dual` per class (k+1).
    offsets: Vec<u32>,
    /// SMO passes actually run per class (k,).
    n_iter_: Vec<usize>,
    pipelines: PipelineCache,
}

impl SVC {
    pub fn new(config: SVCConfig) -> Self {
        Self {
            config,
            gamma: 0.0,
            classes: Vec::new(),
            k: 0,
            d: 0,
            n: 0,
            sv_idx: Vec::new(),
            dual: Vec::new(),
            intercept: Vec::new(),
            pooled_sv: Vec::new(),
            pooled_dual: Vec::new(),
            offsets: Vec::new(),
            n_iter_: Vec::new(),
            pipelines: PipelineCache::new(),
        }
    }

    pub fn n_classes(&self) -> usize {
        self.k
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
    pub fn classes(&self) -> &[f32] {
        &self.classes
    }
    pub fn intercept(&self) -> &[f32] {
        &self.intercept
    }
    /// Number of support vectors per class (k,).
    pub fn n_support(&self) -> Vec<usize> {
        self.sv_idx.iter().map(|v| v.len()).collect()
    }
    /// Total number of pooled support vectors (across all one-vs-rest
    /// classifiers; a training point reused by several classifiers appears
    /// once per classifier).
    pub fn support_count(&self) -> usize {
        self.pooled_dual.len()
    }
    /// Flattened pooled support vectors, row-major `(ns × n_features)`.
    pub fn support_vectors(&self) -> &[f32] {
        &self.pooled_sv
    }
    /// Pooled dual coefficients `α_s·y_s` aligned to `support_vectors`.
    pub fn dual_coef(&self) -> &[f32] {
        &self.pooled_dual
    }
    /// SMO passes run per class.
    pub fn n_iter(&self) -> &[usize] {
        &self.n_iter_
    }

    /// Fit the classifier with GPU Gram + host SMO.
    ///
    /// * `ctx` - Metal context
    /// * `data` - Flat row-major training data (n × d)
    /// * `y` - Labels, one per row (n,); a label may be any `f32` (values that
    ///   differ by `< 0.5` are treated as the same class).
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
        anyhow::ensure!(y.len() == n, "Label length mismatch");
        anyhow::ensure!(
            self.config.c > 0.0 && self.config.c.is_finite(),
            "C must be a finite, positive value"
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

        // Unique classes in order of first appearance.
        let mut classes: Vec<f32> = Vec::new();
        for &yi in y {
            if !classes.iter().any(|&c| (c - yi).abs() < 0.5) {
                classes.push(yi);
            }
        }
        anyhow::ensure!(classes.len() >= 2, "SVC requires at least two classes");

        // One GPU launch: the full n×n Gram matrix, shared by all OVR problems.
        let gram = self.gram(ctx, data, n, d, gamma)?;

        let mut sv_idx: Vec<Vec<usize>> = Vec::with_capacity(classes.len());
        let mut dual: Vec<Vec<f32>> = Vec::with_capacity(classes.len());
        let mut intercept = Vec::with_capacity(classes.len());
        let mut n_iter_ = Vec::with_capacity(classes.len());

        for (ci, &cc) in classes.iter().enumerate() {
            let yt: Vec<f32> = (0..n)
                .map(|i| if (y[i] - cc).abs() < 0.5 { 1.0 } else { -1.0 })
                .collect();
            let (a, b, iters) = self.smo(&yt, &gram, n, ci);
            let sv: Vec<usize> = (0..n).filter(|&i| a[i] > 1e-5).collect();
            sv_idx.push(sv.clone());
            dual.push(sv.iter().map(|&i| a[i] * yt[i]).collect());
            intercept.push(b);
            n_iter_.push(iters);
        }

        // Pooled buffers for the decision kernel.
        let mut pooled_sv = Vec::new();
        let mut pooled_dual = Vec::new();
        let mut offsets = vec![0u32; classes.len() + 1];
        for c in 0..classes.len() {
            for &i in &sv_idx[c] {
                pooled_sv.extend_from_slice(&data[i * d..(i + 1) * d]);
            }
            pooled_dual.extend_from_slice(&dual[c]);
            offsets[c + 1] = pooled_dual.len() as u32;
        }

        self.classes = classes;
        self.gamma = gamma;
        self.k = self.classes.len();
        self.d = d;
        self.n = n;
        self.sv_idx = sv_idx;
        self.dual = dual;
        self.intercept = intercept;
        self.pooled_sv = pooled_sv;
        self.pooled_dual = pooled_dual;
        self.offsets = offsets;
        self.n_iter_ = n_iter_;
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

    // ── Host: Platt (simplified) SMO ───────────────────────────────────────

    /// Solve one one-vs-rest problem with the simplified Platt SMO, using the
    /// precomputed Gram `K` as the fast 2nd-order κ lookup.
    ///
    /// Returns `(α, b, passes_run)`.
    fn smo(&self, y: &[f32], gram: &[f32], n: usize, ci: usize) -> (Vec<f32>, f32, usize) {
        let c = self.config.c;
        let tol = self.config.tolerance;
        let max_pass = self.config.max_iter;
        let mut rng = fastrand::Rng::with_seed(self.config.seed + ci as u64 * 2654435761);

        let mut alpha = vec![0.0f32; n];
        let mut b = 0.0f32;
        let mut passes = 0usize;
        let mut iter = 0usize;
        let eps = 1e-5f32;

        while passes < max_pass {
            let mut num_changed = 0usize;
            for i in 0..n {
                let ei = self.error_i(gram, &alpha, y, b, i, n);
                let violates =
                    (y[i] * ei < -tol && alpha[i] < c) || (y[i] * ei > tol && alpha[i] > 0.0);
                if !violates {
                    continue;
                }
                let j = if n == 1 {
                    i
                } else {
                    let mut jj = rng.usize(0..n);
                    while jj == i {
                        jj = rng.usize(0..n);
                    }
                    jj
                };
                let ej = self.error_i(gram, &alpha, y, b, j, n);
                let ai_old = alpha[i];
                let aj_old = alpha[j];
                let (l, h) = if y[i] != y[j] {
                    (0.0f32.max(aj_old - ai_old), c.min(c + aj_old - ai_old))
                } else {
                    (0.0f32.max(ai_old + aj_old - c), c.min(ai_old + aj_old))
                };
                if (l - h).abs() < 1e-12 {
                    continue;
                }
                let eta = 2.0 * gram[j * n + i] - gram[i * n + i] - gram[j * n + j];
                if eta >= 0.0 {
                    continue;
                }
                let aj_new = (aj_old - y[j] * (ei - ej) / eta).max(l).min(h);
                if (aj_new - aj_old).abs() < eps {
                    continue;
                }
                let ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new);
                let b1 = b
                    - ei
                    - y[i] * (ai_new - ai_old) * gram[i * n + i]
                    - y[j] * (aj_new - aj_old) * gram[i * n + j];
                let b2 = b
                    - ej
                    - y[i] * (ai_new - ai_old) * gram[i * n + j]
                    - y[j] * (aj_new - aj_old) * gram[j * n + j];
                b = if 0.0 < ai_new && ai_new < c {
                    b1
                } else if 0.0 < aj_new && aj_new < c {
                    b2
                } else {
                    0.5 * (b1 + b2)
                };
                alpha[i] = ai_new;
                alpha[j] = aj_new;
                num_changed += 1;
            }
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
            iter += 1;
        }
        (alpha, b, iter)
    }

    /// Decision-function error at index `i`: `f(x_i) - y_i`, where
    /// `f(x_i) = Σ_j α_j y_j K[i][j] + b`.
    fn error_i(&self, gram: &[f32], alpha: &[f32], y: &[f32], b: f32, i: usize, n: usize) -> f32 {
        let mut sum = 0.0f32;
        for j in 0..n {
            if alpha[j] != 0.0 {
                sum += alpha[j] * y[j] * gram[i * n + j];
            }
        }
        sum + b - y[i]
    }

    // ── GPU: decision matrix + labels ──────────────────────────────────────

    /// Raw decision matrix (M × k) `dec[m][cc] = b_cc + Σ α_s y_s κ(x_m, SV_s)`
    /// over the pooled support vectors; one GPU launch.
    pub fn decision_function(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(self.k > 0, "Model is not fitted; call fit() first");
        anyhow::ensure!(d == self.d, "Feature dimension mismatch");
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");
        let m = n;
        let ns = self.pooled_dual.len() as u32;
        let out_len = m * self.k;
        let p = self.pipelines.get(ctx, "svm_predict")?.clone();
        let x_buf = ctx.new_buffer(data); // (m, d)
        let s_buf = ctx.new_buffer(&self.pooled_sv); // (ns, d)
        let dual_buf = ctx.new_buffer(&self.pooled_dual); // (ns)
        let bias_buf = ctx.new_buffer(&self.intercept); // (k)
        let off_buf = ctx.new_buffer(&self.offsets); // (k+1)
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
        set_u32(&enc, 7, self.k as u32);
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

    /// Hard labels: argmax over the one-vs-rest decision scores (sign of the
    /// binary score when there are exactly two classes).
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let dec = self.decision_function(ctx, data, n, d)?;
        let mut preds = vec![0.0f32; n];
        for i in 0..n {
            if self.k == 2 {
                preds[i] = if dec[i * self.k] > 0.0 {
                    self.classes[0]
                } else {
                    self.classes[1]
                };
                continue;
            }
            let mut best = 0usize;
            let mut best_v = dec[i * self.k];
            for c in 1..self.k {
                if dec[i * self.k + c] > best_v {
                    best_v = dec[i * self.k + c];
                    best = c;
                }
            }
            preds[i] = self.classes[best];
        }
        Ok(preds)
    }

    /// Prediction accuracy: `predict(X) == y`.
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(y.len() == n, "Label length mismatch");
        let preds = self.predict(ctx, data, n, d)?;
        let mut correct = 0usize;
        for i in 0..n {
            if (preds[i] - y[i]).abs() < 0.5 {
                correct += 1;
            }
        }
        Ok(correct as f32 / n as f32)
    }
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

    /// `k * n_per` Gaussian blobs, blob `c` centered at 10 on axis `c`.
    fn blobs(seed: u64, n_per: usize, k: usize, d: usize, spread: f32) -> (Vec<f32>, Vec<f32>) {
        let n = n_per * k;
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut data = vec![0.0f32; n * d];
        let mut labels = Vec::with_capacity(n);
        let mut idx = 0usize;
        for c in 0..k {
            for _ in 0..n_per {
                for dim in 0..d {
                    let center = if dim == c { 10.0 } else { 0.0 };
                    data[idx * d + dim] = center + (rng.f32() - 0.5) * spread;
                }
                labels.push(c as f32);
                idx += 1;
            }
        }
        (data, labels)
    }

    fn svc() -> SVC {
        SVC::new(SVCConfig::default())
    }

    #[test]
    fn svm_blobs_linear() {
        let ctx = MetalContext::new().unwrap();
        let (data, labels) = blobs(1, 40, 3, 4, 1.0);
        let mut m = SVC::new(SVCConfig {
            kernel: SVCKernel::Linear,
            c: 1.0,
            tolerance: 1e-4,
            max_iter: 200,
            seed: 7,
            ..Default::default()
        });
        m.fit(&ctx, &data, &labels, 120, 4).unwrap();
        let preds = m.predict(&ctx, &data, 120, 4).unwrap();
        let acc = m.score(&ctx, &data, &labels, 120, 4).unwrap();
        assert!(acc > 0.95, "accuracy {} too low", acc);
        assert_eq!(m.classes(), &[0.0, 1.0, 2.0]);
        assert_eq!(m.n_support().len(), 3);
        assert_eq!(m.support_vectors().len() % 4, 0);
        assert_eq!(m.intercept().len(), 3);
        // every pooled dual is alpha*y of a support (alpha > 1e-5 => |dual| > 0)
        assert!(m.dual_coef().iter().all(|&v| v.abs() > 0.0));
        // decision matrix shape (n, k)
        let dec = m.decision_function(&ctx, &data, 120, 4).unwrap();
        assert_eq!(dec.len(), 120 * 3);
        // prediction is argmax of the decision rows
        for i in 0..120 {
            let best = (0..3)
                .max_by(|&a, &b| dec[i * 3 + a].partial_cmp(&dec[i * 3 + b]).unwrap())
                .unwrap();
            assert!((preds[i] - [0.0, 1.0, 2.0][best]).abs() < 0.5);
        }
    }

    #[test]
    fn svm_rbf_nonlinear() {
        // XOR-like 2-D pattern: separable in RBF space, not linearly.
        let ctx = MetalContext::new().unwrap();
        let data: Vec<f32> = vec![
            0.0, 0.0, 1.0, 1.0, // class 0
            0.0, 1.0, 1.0, 0.0, // class 1
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let mut m = SVC::new(SVCConfig {
            kernel: SVCKernel::Rbf,
            gamma: 0.5,
            c: 100.0,
            tolerance: 1e-6,
            max_iter: 400,
            seed: 3,
            ..Default::default()
        });
        m.fit(&ctx, &data, &labels, 4, 2).unwrap();
        let preds = m.predict(&ctx, &data, 4, 2).unwrap();
        assert_eq!(preds, labels);
    }

    #[test]
    fn svm_invalid_inputs() {
        let ctx = MetalContext::new().unwrap();
        let mut m = svc();
        // not fitted => decision errors
        assert!(m.predict(&ctx, &[1.0, 2.0], 1, 2).is_err());
        // one class only
        let (data, _) = blobs(1, 10, 1, 3, 1.0);
        let labels = vec![0.0f32; 10];
        assert!(m.fit(&ctx, &data, &labels, 10, 3).is_err());
        // bad C
        let mut m2 = SVC::new(SVCConfig {
            c: 0.0,
            ..Default::default()
        });
        let (data, labels) = blobs(1, 10, 2, 3, 1.0);
        assert!(m2.fit(&ctx, &data, &labels, 20, 3).is_err());
    }

    #[test]
    fn svm_deterministic_seed() {
        let ctx = MetalContext::new().unwrap();
        let (data, labels) = blobs(9, 30, 3, 2, 0.8);
        let mut a = SVC::new(SVCConfig {
            kernel: SVCKernel::Rbf,
            seed: 11,
            ..Default::default()
        });
        let mut b = SVC::new(SVCConfig {
            kernel: SVCKernel::Rbf,
            seed: 11,
            ..Default::default()
        });
        a.fit(&ctx, &data, &labels, 90, 2).unwrap();
        b.fit(&ctx, &data, &labels, 90, 2).unwrap();
        assert_eq!(a.intercept(), b.intercept());
        assert_eq!(
            a.predict(&ctx, &data, 90, 2).unwrap(),
            b.predict(&ctx, &data, 90, 2).unwrap()
        );
    }
}
