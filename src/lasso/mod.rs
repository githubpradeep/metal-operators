//! Lasso (L1-regularized linear regression) with GPU-accelerated Gram build.
//!
//! Mirrors scikit-learn's `sklearn.linear_model.Lasso` (L1-regularized linear
//! regression) while keeping the heavy linear-algebra stage on the GPU and the
//! iterative coordinate-descent on the host:
//!
//! 1. **Augmented Gram system (GPU)** — the exact same `linreg_gram_xtx_*`,
//!    `linreg_gram_xty` and `linreg_reduce_gram` kernels (`shaders/linear.metal`)
//!    stream the dataset once to build the `(d+1)×(d+1)` normal-equations system
//!    `[XᵀX | Xᵀ·1; 1ᵀ·X | n]` and right-hand side `[Xᵀy; Σy]` (augmented with an
//!    intercept column when `fit_intercept`).
//! 2. **Coordinate descent (host)** — a Gauss-Seidel sweep over the small cached
//!    Gram matrix computes each coefficient update:
//!    `w_j = soft_threshold(rho_j, alpha) / (X_j·X_j)`, with the intercept column
//!    unpenalized, until a full pass changes nothing more than `tol` or
//!    `max_iterations` is reached. The `(d+1)²` Gram keeps each iteration O(d²).
//! 3. **Predict (GPU)** — the same `linreg_predict` kernel returns `Xw + b`.

use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/linear.metal");

// ── Kernel dispatch selection ─────────────────────────────────────

/// Cap on the per-group partials memory (bytes), same as linear_regression.
const PARTIALS_CAP_BYTES: usize = 32 * 1024 * 1024;

fn pick_gram_kernel_name(d: usize) -> &'static str {
    if d <= 128 {
        "linreg_gram_xtx_tiled"
    } else {
        "linreg_gram_xtx_naive"
    }
}

// ── Pipeline cache ────────────────────────────────────────────────

struct PipelineCache {
    gram_xtx_tiled: OnceLock<ComputePipelineState>,
    gram_xtx_naive: OnceLock<ComputePipelineState>,
    gram_xty: OnceLock<ComputePipelineState>,
    reduce_gram: OnceLock<ComputePipelineState>,
    predict: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            gram_xtx_tiled: OnceLock::new(),
            gram_xtx_naive: OnceLock::new(),
            gram_xty: OnceLock::new(),
            reduce_gram: OnceLock::new(),
            predict: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "linreg_gram_xtx_tiled" => &self.gram_xtx_tiled,
            "linreg_gram_xtx_naive" => &self.gram_xtx_naive,
            "linreg_gram_xty" => &self.gram_xty,
            "linreg_reduce_gram" => &self.reduce_gram,
            "linreg_predict" => &self.predict,
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

/// Configuration for Lasso training.
///
/// The model minimizes `0.5·‖Xw − y‖² + alpha·‖w‖₁` on the cached augmented
/// Gram with host coordinate descent, matching sklearn's `Lasso` objective
/// (intercept is not penalized).
#[derive(Clone, Debug)]
pub struct LassoConfig {
    /// L1 regularization strength (sklearn `alpha`). Larger values drive more
    /// coefficients to exactly zero. `0.0` reduces to ordinary least squares.
    pub alpha: f32,
    /// Whether to fit an intercept (bias) term (sklearn default: true).
    pub fit_intercept: bool,
    /// Coordinate-descent convergence tolerance: stop when a full sweep moves
    /// every coefficient by at most `tol` (sklearn default 1e-4).
    pub tol: f32,
    /// Maximum coordinate-descent sweeps (sklearn default 1000).
    pub max_iterations: usize,
    /// Random seed (kept for API symmetry; coordinate descent is deterministic).
    pub seed: u64,
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            tol: 1e-4,
            max_iterations: 1000,
            seed: 42,
        }
    }
}

// ── Lasso struct ──────────────────────────────────────────────────

/// Lasso regression solved by host coordinate descent over a GPU-built Gram.
pub struct Lasso {
    config: LassoConfig,
    pipelines: PipelineCache,
    /// Learned feature weights, shape (d,).
    pub weights: Vec<f32>,
    /// Learned intercept.
    pub bias: f32,
    /// Number of features (dimensionality).
    pub d: usize,
    /// Number of coordinate-descent sweeps actually performed at fit time.
    pub n_iter: usize,
    /// Whether the solver converged within `max_iterations`.
    pub converged: bool,
    /// Mean squared error on the training data after fit.
    pub final_loss: f32,
}

impl Lasso {
    /// Create a new `Lasso` with the given configuration.
    pub fn new(config: LassoConfig) -> Self {
        let pipelines = PipelineCache::new();
        Self {
            config,
            pipelines,
            weights: Vec::new(),
            bias: 0.0,
            d: 0,
            n_iter: 0,
            converged: false,
            final_loss: f32::INFINITY,
        }
    }

    /// The fitted coefficient vector of shape `(d,)`.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Alias for [`Lasso::weights`] (sklearn-style `coef_`).
    pub fn coef(&self) -> &[f32] {
        &self.weights
    }

    /// The fitted intercept.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Alias for [`Lasso::bias`] (sklearn-style `intercept_`).
    pub fn intercept(&self) -> f32 {
        self.bias
    }

    /// Number of features the model was fitted with.
    pub fn n_features(&self) -> usize {
        self.d
    }

    // ── Fit ────────────────────────────────────────────────────────

    /// Fit the Lasso to the data. The Gram system is built on the GPU in one
    /// command buffer (three kernels) and coordinate descent runs on the host
    /// over the cached `(d+1)²` system.
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
            "Data length mismatch: expected {} got {}",
            n * d,
            data.len()
        );
        anyhow::ensure!(
            y.len() == n,
            "Targets length mismatch: expected {} got {}",
            n,
            y.len()
        );
        anyhow::ensure!(
            self.config.alpha >= 0.0 && self.config.alpha.is_finite(),
            "alpha must be a finite, non-negative value"
        );
        anyhow::ensure!(
            2 * d + 2 <= 32 * 256,
            "d too large ({}): the Xᵀy element grid exceeds 32 register slots",
            d
        );

        self.d = d;

        // ── Pass 1: build the augmented Gram system on the GPU ──
        let gram_xtx = self.pipelines.get(ctx, pick_gram_kernel_name(d))?;
        let gram_xty = self.pipelines.get(ctx, "linreg_gram_xty")?;
        let reduce_gram = self.pipelines.get(ctx, "linreg_reduce_gram")?;
        let predict = self.pipelines.get(ctx, "linreg_predict")?;

        const TG: u64 = 256;
        let tiled = d <= 128;

        let b_xtx = if tiled {
            ((7680 / d).max(16)).min(256)
        } else {
            256
        };
        let n_chunks_xtx = (n + b_xtx - 1) / b_xtx;
        let g_xtx = if tiled {
            n_chunks_xtx.min((PARTIALS_CAP_BYTES / (d * d * 4)).max(1))
        } else {
            0
        };
        let n_chunks_xty = (n + 255) / 256;
        let g_xty = n_chunks_xty.min((PARTIALS_CAP_BYTES / ((2 * d + 2) * 4)).max(1));

        let x_buf = ctx.new_buffer(data);
        let y_buf = ctx.new_buffer(y);
        let xtx_buf = ctx.new_buffer_uninitialized((d * d * 4) as u64);
        let xtx_partials_buf = ctx.new_buffer_uninitialized(((g_xtx * d * d).max(1) * 4) as u64);
        let xty_partials_buf =
            ctx.new_buffer_uninitialized(((g_xty * (2 * d + 2)).max(1) * 4) as u64);
        let xty_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let colsum_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let stats_buf = ctx.new_buffer_uninitialized(8);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Kernel A: XᵀX (tiled with per-group partials, or naive direct).
        enc.set_compute_pipeline_state(&gram_xtx);
        enc.set_buffer(0, Some(&x_buf), 0);
        if tiled {
            enc.set_buffer(1, Some(&xtx_partials_buf), 0);
        } else {
            enc.set_buffer(1, Some(&xtx_buf), 0);
        }
        set_u32(&enc, 2, n as u32);
        set_u32(&enc, 3, d as u32);
        if tiled {
            set_u32(&enc, 4, b_xtx as u32);
            set_u32(&enc, 5, g_xtx as u32);
            enc.set_threadgroup_memory_length(0, (b_xtx * d * 4) as u64);
        }
        enc.dispatch_thread_groups(
            MTLSize {
                width: if tiled {
                    g_xtx as u64
                } else {
                    (d * d) as u64 / TG + 1
                },
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );

        // Kernel B: Xᵀy / column sums / count / Σy partials.
        enc.set_compute_pipeline_state(&gram_xty);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&y_buf), 0);
        enc.set_buffer(2, Some(&xty_partials_buf), 0);
        set_u32(&enc, 3, n as u32);
        set_u32(&enc, 4, d as u32);
        set_u32(&enc, 5, g_xty as u32);
        enc.dispatch_thread_groups(
            MTLSize {
                width: g_xty as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );

        // Kernel C: deterministic fixed-order reduction of the partials.
        enc.set_compute_pipeline_state(&reduce_gram);
        enc.set_buffer(0, Some(&xtx_partials_buf), 0);
        enc.set_buffer(1, Some(&xty_partials_buf), 0);
        enc.set_buffer(2, Some(&xtx_buf), 0);
        enc.set_buffer(3, Some(&xty_buf), 0);
        enc.set_buffer(4, Some(&colsum_buf), 0);
        enc.set_buffer(5, Some(&stats_buf), 0);
        set_u32(&enc, 6, g_xtx as u32);
        set_u32(&enc, 7, g_xty as u32);
        set_u32(&enc, 8, d as u32);
        enc.dispatch_thread_groups(
            MTLSize {
                width: 1,
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

        let xtx: Vec<f32> = ctx.read_buffer(&xtx_buf, d * d);
        let xty: Vec<f32> = ctx.read_buffer(&xty_buf, d);
        let colsum: Vec<f32> = ctx.read_buffer(&colsum_buf, d);
        let stats: Vec<f32> = ctx.read_buffer(&stats_buf, 2);

        // ── Host coordinate descent on the augmented (d+1)² system ──
        let dim = if self.config.fit_intercept { d + 1 } else { d };
        // `a` is the (row-major) augmented Gram `[XᵀX | Xᵀ·1; 1ᵀ·X | n]`,
        // `b` the right-hand side `[Xᵀy; Σy]`; the last column/row is the
        // intercept (a column of ones) and is never penalized.
        let mut a = vec![0.0f32; dim * dim];
        let mut b = vec![0.0f32; dim];
        for j in 0..d {
            for k in 0..d {
                a[j * dim + k] = xtx[j * d + k];
            }
            b[j] = xty[j];
        }
        if self.config.fit_intercept {
            for j in 0..d {
                a[j * dim + d] = colsum[j];
                a[d * dim + j] = colsum[j];
            }
            a[d * dim + d] = stats[0]; // n
            b[d] = stats[1]; // Σy
        }

        // Coordinate descent (Gauss-Seidel). For each coefficient:
        //   rho_j   = b_j − Σ_{k, k≠j} a[j][k]·w_k   =  (X_j)ᵀ·r + a[j][j]·w_j
        //   w_j (<- intercept col) = rho_j / a[j][j]
        //   w_j (penalized)        = soft_threshold(rho_j, alpha) / a[j][j]
        let mut w = vec![0.0f32; dim];
        let mut iters = 0usize;
        let mut converged = false;
        for _ in 0..self.config.max_iterations {
            let mut max_change = 0.0f32;
            for j in 0..dim {
                // rho_j = b_j − Σ_k a[j][k]·w_k + a[j][j]·w_j
                let mut rho = b[j];
                for k in 0..dim {
                    rho -= a[j * dim + k] * w[k];
                }
                let diag = a[j * dim + j];
                rho += diag * w[j];
                let denom = if diag.abs() < f32::EPSILON {
                    f32::EPSILON
                } else {
                    diag
                };
                let new_w = if self.config.fit_intercept && j == dim - 1 {
                    rho / denom
                } else {
                    soft_threshold(rho, self.config.alpha) / denom
                };
                max_change = max_change.max((new_w - w[j]).abs());
                w[j] = new_w;
            }
            iters += 1;
            if max_change <= self.config.tol {
                converged = true;
                break;
            }
        }

        self.weights = w[..d].to_vec();
        self.bias = if self.config.fit_intercept { w[d] } else { 0.0 };
        self.n_iter = iters;
        self.converged = converged;

        // ── Pass 2: predict on the training data to report the MSE ──
        let w_buf = ctx.new_buffer(&self.weights);
        let pred_buf = ctx.new_buffer_uninitialized((n * 4) as u64);
        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&predict);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&pred_buf), 0);
        set_u32(&enc, 3, n as u32);
        set_u32(&enc, 4, d as u32);
        set_f32(&enc, 5, self.bias);
        enc.dispatch_thread_groups(
            MTLSize {
                width: (n as u64 + TG - 1) / TG,
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

        let preds: Vec<f32> = ctx.read_buffer(&pred_buf, n);
        self.final_loss = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| {
                let r = p - t;
                r * r
            })
            .sum::<f32>()
            / n as f32;

        Ok(())
    }

    // ── Predict ────────────────────────────────────────────────────

    /// Predict continuous targets for input data using GPU.
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            d == self.d,
            "Data dimension mismatch: expected {} got {}",
            self.d,
            d
        );
        anyhow::ensure!(!self.weights.is_empty(), "Model not fitted");
        anyhow::ensure!(data.len() == n * d, "Data length mismatch");

        let pipeline = self.pipelines.get(ctx, "linreg_predict")?;
        let x_buf = ctx.new_buffer(data);
        let w_buf = ctx.new_buffer(&self.weights);
        let out_buf = ctx.new_buffer_uninitialized((n * 4) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        set_u32(&enc, 3, n as u32);
        set_u32(&enc, 4, d as u32);
        set_f32(&enc, 5, self.bias);

        const TG: u64 = 256;
        enc.dispatch_thread_groups(
            MTLSize {
                width: (n as u64 + TG - 1) / TG,
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

        Ok(ctx.read_buffer(&out_buf, n))
    }

    /// Compute the coefficient of determination (R²) score.
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(
            y.len() == n,
            "Targets length mismatch: expected {} got {}",
            n,
            y.len()
        );
        let preds = self.predict(ctx, data, n, d)?;

        let y_mean = y.iter().sum::<f32>() / n as f32;
        let ss_res: f32 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| {
                let r = p - t;
                r * r
            })
            .sum();
        let ss_tot: f32 = y
            .iter()
            .map(|t| {
                let r = t - y_mean;
                r * r
            })
            .sum();

        if ss_tot <= f32::EPSILON {
            return Ok(if ss_res <= f32::EPSILON { 1.0 } else { 0.0 });
        }
        Ok(1.0 - ss_res / ss_tot)
    }
}

// ── Helper functions ─────────────────────────────────────────────

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    let ptr = std::ptr::from_ref(&value);
    encoder.set_bytes(index, len, ptr.cast());
}

fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    let len = std::mem::size_of::<f32>() as u64;
    let ptr = std::ptr::from_ref(&value);
    encoder.set_bytes(index, len, ptr.cast());
}

/// Soft-thresholding operator `S(x, a) = sign(x)·max(|x| − a, 0)`, the
/// proximal map of the L1 penalty used by every coordinate-descent update.
fn soft_threshold(x: f32, a: f32) -> f32 {
    if x > a {
        x - a
    } else if x < -a {
        x + a
    } else {
        0.0
    }
}
