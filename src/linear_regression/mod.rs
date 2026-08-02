use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/linear.metal");

// ── Kernel dispatch selection ─────────────────────────────────────

/// Cap on the per-group partials memory (bytes). When the row-chunk grid is
/// larger, groups grid-stride over multiple chunks and accumulate partials
/// in registers, so the partials buffer stays bounded.
const PARTIALS_CAP_BYTES: usize = 32 * 1024 * 1024;

fn pick_gram_kernel_name(d: usize) -> &'static str {
    // The tiled kernel keeps d²/256 register slots per thread; d <= 128
    // keeps that at <= 64 slots.
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

/// Configuration for linear regression training.
///
/// The model is solved in closed form via the normal equations
/// (XᵀX + alpha·I) w = Xᵀy on the GPU with an optional intercept, matching
/// sklearn's `LinearRegression` (alpha = 0) and `Ridge` (alpha > 0)
/// conventions. `max_iterations`, `tol` and `seed` are kept for API
/// symmetry with flashlib but are not used by the closed-form solver.
#[derive(Clone)]
pub struct LinearRegressionConfig {
    /// L2 regularization strength applied to the coefficients only
    /// (0.0 = plain ordinary least squares; sklearn `Ridge` convention).
    pub alpha: f32,
    /// Whether to fit an intercept (bias) term (sklearn default: true).
    pub fit_intercept: bool,
    /// Convergence tolerance (unused; closed-form solve; API symmetry).
    pub tol: f32,
    /// Maximum solver iterations (unused; closed-form solve; API symmetry).
    pub max_iterations: usize,
    /// Random seed (unused; solver is deterministic; API symmetry).
    pub seed: u64,
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            fit_intercept: true,
            tol: 1e-4,
            max_iterations: 100,
            seed: 42,
        }
    }
}

// ── LinearRegression struct ───────────────────────────────────────

/// Linear regression solved exactly on the GPU via the normal equations.
///
/// Two streaming kernels build the augmented Gram matrix
/// ([XᵀX | Xᵀ·1; 1ᵀ·X | n]) and right-hand side ([Xᵀy; Σy]): the XᵀX kernel
/// stages row blocks in shared memory (tiled; d ≤ 128) or uses one thread
/// per element (naive; d > 128), and a deterministic fixed-order reduction
/// kernel combines the per-group partials — no device atomics. The
/// (d+1)×(d+1) system is solved on the host with Gaussian elimination +
/// partial pivoting, and a dedicated predict kernel evaluates the model.
pub struct LinearRegression {
    config: LinearRegressionConfig,
    /// Coefficients (d,).
    pub weights: Vec<f32>,
    /// Intercept (scalar; 0.0 when `fit_intercept` is false).
    pub bias: f32,
    /// Number of features.
    pub d: usize,
    /// Pipeline cache.
    pipelines: PipelineCache,
    /// Solver iterations (1 = closed-form; API symmetry with SGD/L-BFGS).
    pub n_iter: usize,
    /// Mean squared error on the training data after fit.
    pub final_loss: f32,
}

impl LinearRegression {
    pub fn new(config: LinearRegressionConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            bias: 0.0,
            d: 0,
            pipelines: PipelineCache::new(),
            n_iter: 0,
            final_loss: 0.0,
        }
    }

    /// Return a reference to the fitted coefficients.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Alias of [`LinearRegression::weights`] (sklearn-style name).
    pub fn coef(&self) -> &[f32] {
        &self.weights
    }

    /// Return the intercept term.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Alias of [`LinearRegression::bias`] (sklearn-style name).
    pub fn intercept(&self) -> f32 {
        self.bias
    }

    /// Number of features the model was fitted with.
    pub fn n_features(&self) -> usize {
        self.d
    }

    // ── Fit ────────────────────────────────────────────────────────

    /// Fit the linear regression model on the GPU.
    ///
    /// Builds the augmented Gram system with two streaming kernels (XᵀX via
    /// shared-memory row tiles for d ≤ 128, Xᵀy / column sums via chunked
    /// element columns), reduces the per-group partials deterministically on
    /// the GPU, then solves the (d+1)×(d+1) system on the host. Two
    /// CPU↔GPU syncs total (Gram build + predict for the training MSE).
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `data` - Flat row-major data array (n × d)
    /// * `y` - Targets (n,)
    /// * `n` - Number of samples
    /// * `d` - Number of features
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

        let gram_xtx = self.pipelines.get(ctx, pick_gram_kernel_name(d))?;
        let gram_xty = self.pipelines.get(ctx, "linreg_gram_xty")?;
        let reduce_gram = self.pipelines.get(ctx, "linreg_reduce_gram")?;
        let predict = self.pipelines.get(ctx, "linreg_predict")?;

        const TG: u64 = 256;
        let tiled = d <= 128;

        // ── Grid geometry ──
        // XᵀX tiled path: B rows per chunk (shared tile <= 30 KB), groups
        // capped so the partials buffer stays bounded.
        let b_xtx = if tiled {
            ((7680 / d).max(16)).min(256)
        } else {
            256
        };
        let n_chunks_xtx = (n + b_xtx - 1) / b_xtx;
        let g_xtx = if tiled {
            n_chunks_xtx.min((PARTIALS_CAP_BYTES / (d * d * 4)).max(1))
        } else {
            0 // naive path writes XᵀX directly: no partials, no reduction
        };
        // Xᵀy path: 256-row chunks.
        let n_chunks_xty = (n + 255) / 256;
        let g_xty = n_chunks_xty.min((PARTIALS_CAP_BYTES / ((2 * d + 2) * 4)).max(1));

        // ── GPU buffers ──
        let x_buf = ctx.new_buffer(data);
        let y_buf = ctx.new_buffer(y);
        let xtx_buf = ctx.new_buffer_uninitialized((d * d * 4) as u64);
        let xtx_partials_buf = ctx.new_buffer_uninitialized(((g_xtx * d * d).max(1) * 4) as u64);
        let xty_partials_buf =
            ctx.new_buffer_uninitialized(((g_xty * (2 * d + 2)).max(1) * 4) as u64);
        let xty_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let colsum_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let stats_buf = ctx.new_buffer_uninitialized(8);
        let pred_buf = ctx.new_buffer_uninitialized((n * 4) as u64);

        // ── Pass 1: build the augmented Gram system (one command buffer) ──
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

        // ── Host solve: (d+1) or d dimensional augmented system ──
        let dim = if self.config.fit_intercept { d + 1 } else { d };
        let mut a = vec![0.0f32; dim * dim];
        let mut b = vec![0.0f32; dim];

        // Coefficients block: XᵀX + alpha·I (+ tiny ridge for rank-deficient
        // data, matching flashlib's diagonal regulariser).
        let mut max_diag = 0.0f32;
        for j in 0..d {
            let g = xtx[j * d + j];
            max_diag = max_diag.max(g);
        }
        let eps = 1e-7f32 * (1.0 + max_diag);
        for j in 0..d {
            for k in 0..d {
                a[j * dim + k] = xtx[j * d + k];
            }
            a[j * dim + j] += self.config.alpha + eps;
            b[j] = xty[j];
        }

        if self.config.fit_intercept {
            for j in 0..d {
                a[j * dim + d] = colsum[j];
                a[d * dim + j] = colsum[j];
            }
            a[d * dim + d] = stats[0] + eps; // n (+ ridge for the intercept)
            b[d] = stats[1]; // Σy
        }

        let x = solve_linear_system(&mut a, &mut b, dim)?;
        self.weights = x[..d].to_vec();
        self.bias = if self.config.fit_intercept { x[d] } else { 0.0 };
        self.n_iter = 1;

        // ── Pass 2: predict on the training data to report the MSE ──
        let w_buf = ctx.new_buffer(&self.weights);
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
            // Constant target: perfect score iff predictions are exact.
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

/// Solve `A x = b` in place with Gaussian elimination + partial pivoting.
/// `a` is a row-major `dim × dim` matrix, `b` is the right-hand side; the
/// solution is returned in `b`. O(dim³) — the augmented system is at most
/// (d+1)² with d in the low hundreds, so a host-side solve is trivial.
fn solve_linear_system(a: &mut [f32], b: &mut [f32], dim: usize) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        a.len() == dim * dim && b.len() == dim,
        "solver dimension mismatch"
    );

    // Forward elimination with partial pivoting.
    for col in 0..dim {
        // Find the pivot row (largest |a[r][col]| at or below the diagonal).
        let mut pivot = col;
        let mut best = a[col * dim + col].abs();
        for r in (col + 1)..dim {
            let v = a[r * dim + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if pivot != col {
            for c in 0..dim {
                a.swap(col * dim + c, pivot * dim + c);
            }
            b.swap(col, pivot);
        }

        let pv = a[col * dim + col];
        if pv.abs() < 1e-30 {
            anyhow::bail!(
                "linear system is singular (pivot {:.3e}); data may be degenerate",
                pv
            );
        }

        for r in (col + 1)..dim {
            let f = a[r * dim + col] / pv;
            if f == 0.0 {
                continue;
            }
            for c in col..dim {
                a[r * dim + c] -= f * a[col * dim + c];
            }
            b[r] -= f * b[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0f32; dim];
    for r in (0..dim).rev() {
        let mut s = b[r];
        for c in (r + 1)..dim {
            s -= a[r * dim + c] * x[c];
        }
        x[r] = s / a[r * dim + r];
    }
    Ok(x)
}
