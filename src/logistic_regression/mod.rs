use crate::metal::MetalContext;
use metal::*;
use std::eprintln;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/logistic.metal");

// ── Kernel dispatch selection ─────────────────────────────────────

fn pick_kernel_name(d: usize) -> &'static str {
    if d >= 8 && d % 8 == 0 {
        return "logreg_fused_fwd_bwd_simdgroup";
    }
    if d > 128 {
        return "logreg_fused_fwd_bwd_splitd";
    }
    "logreg_fused_fwd_bwd_naive"
}

// ── Pipeline cache ────────────────────────────────────────────────

struct PipelineCache {
    fwd_bwd_naive: OnceLock<ComputePipelineState>,
    fwd_bwd_simdgroup: OnceLock<ComputePipelineState>,
    fwd_bwd_splitd: OnceLock<ComputePipelineState>,
    reduce: OnceLock<ComputePipelineState>,
    predict: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            fwd_bwd_naive: OnceLock::new(),
            fwd_bwd_simdgroup: OnceLock::new(),
            fwd_bwd_splitd: OnceLock::new(),
            reduce: OnceLock::new(),
            predict: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "logreg_fused_fwd_bwd_naive" => &self.fwd_bwd_naive,
            "logreg_fused_fwd_bwd_simdgroup" => &self.fwd_bwd_simdgroup,
            "logreg_fused_fwd_bwd_splitd" => &self.fwd_bwd_splitd,
            "logreg_reduce_partials" => &self.reduce,
            "logreg_predict" => &self.predict,
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

/// Configuration for logistic regression training.
/// Follows sklearn conventions: C = inverse regularization strength.
///
/// Training uses full-batch L-BFGS (m = 10) on the GPU. `learning_rate`,
/// `momentum`, `batch_size` and `seed` are kept for API symmetry with
/// flashlib but are not used by the optimizer.
#[derive(Clone)]
pub struct LogisticRegressionConfig {
    /// Inverse regularization strength (larger C = weaker regularization).
    pub c: f32,
    /// Learning rate (unused by L-BFGS; kept for API symmetry).
    pub learning_rate: f32,
    /// Momentum factor (unused by L-BFGS; kept for API symmetry).
    pub momentum: f32,
    /// Maximum number of L-BFGS iterations.
    pub max_epochs: usize,
    /// Mini-batch size (unused; L-BFGS is full-batch).
    pub batch_size: usize,
    /// Convergence tolerance on gradient sup-norm.
    pub tol: f32,
    /// Random seed (unused; optimizer is deterministic).
    pub seed: u64,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            learning_rate: 0.01,
            momentum: 0.9,
            max_epochs: 100,
            batch_size: 256,
            tol: 1e-4,
            seed: 42,
        }
    }
}

// ── LogisticRegression struct ─────────────────────────────────────

/// Binary logistic regression trained on GPU with full-batch L-BFGS (m = 10).
/// Fused forward/backward/loss kernels evaluate the full dataset in one
/// launch per iteration; the host runs the L-BFGS update.
pub struct LogisticRegression {
    config: LogisticRegressionConfig,
    /// Weights (D,).
    pub weights: Vec<f32>,
    /// Bias (scalar).
    pub bias: f32,
    /// Number of features.
    pub d: usize,
    /// Pipeline cache.
    pipelines: PipelineCache,
    /// Number of epochs trained.
    pub n_epochs: usize,
    /// Final loss value.
    pub final_loss: f32,
}

impl LogisticRegression {
    pub fn new(config: LogisticRegressionConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            bias: 0.0,
            d: 0,
            pipelines: PipelineCache::new(),
            n_epochs: 0,
            final_loss: 0.0,
        }
    }

    /// Return a reference to the trained weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Return the bias term.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    // ── Fit ────────────────────────────────────────────────────────

    /// Fit the logistic regression model using full-batch L-BFGS (m = 10) on
    /// the GPU. Each iteration evaluates loss + gradient with a single fused
    /// kernel launch over the whole dataset (one CPU↔GPU sync per iteration).
    ///
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `data` - Flat row-major data array (n × d)
    /// * `y` - Binary labels (n,) with values in {0.0, 1.0}
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
            "Labels length mismatch: expected {} got {}",
            n,
            y.len()
        );

        self.d = d;
        self.weights = vec![0.0f32; d];
        self.bias = 0.0;

        // L-BFGS memory (flashlib uses m = 10).
        let m_lbfgs = 10usize;
        // Inverse regularization on the averaged loss: matches sklearn/flashlib.
        let c_inv = if self.config.c > 0.0 {
            1.0 / (self.config.c * n as f32)
        } else {
            0.0
        };
        let inv_n = 1.0 / n as f32;

        // Pick kernel variant and compile pipelines once (cached).
        let kernel_name = pick_kernel_name(d);
        let fwd_bwd = self.pipelines.get(ctx, kernel_name)?;
        let reduce = self.pipelines.get(ctx, "logreg_reduce_partials")?;
        let predict = self.pipelines.get(ctx, "logreg_predict")?;

        const TG: u64 = 256;
        let n_tg = (n as u64 + TG - 1) / TG;

        // ── Persistent GPU buffers (allocated once, reused every iteration) ──
        let x_buf = ctx.new_buffer(data);
        let y_buf = ctx.new_buffer(y);
        let w_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let gw_buf = ctx.new_buffer_uninitialized((d * 4) as u64);
        let gb_buf = ctx.new_buffer_uninitialized(4);
        let loss_buf = ctx.new_buffer_uninitialized(4);
        let partials_buf = ctx.new_buffer_uninitialized((n_tg * (d as u64 + 2) * 4) as u64);
        let logits_buf = ctx.new_buffer_uninitialized((n * 4) as u64);

        // CPU mirrors of GPU state.
        let mut w = vec![0.0f32; d];
        let mut b = 0.0f32;
        let mut grad_w = vec![0.0f32; d];
        let mut grad_b = 0.0f32;
        let mut loss_val = 0.0f32;

        // ── Single GPU evaluation: loss + gradient at (w, b) ──
        // Fused kernel over the FULL dataset + tiny reduction kernel, both in
        // one command buffer -> one wait + small readback per evaluation.
        let eval = |w: &[f32], b: f32, gw: &mut Vec<f32>, gb: &mut f32, loss: &mut f32| {
            ctx.write_buffer(&w_buf, w);

            let cmd_buf = ctx.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();

            // Kernel A: fused forward/backward/loss -> per-threadgroup partials.
            enc.set_compute_pipeline_state(&fwd_bwd);
            enc.set_buffer(0, Some(&x_buf), 0);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            enc.set_buffer(3, Some(&partials_buf), 0);
            set_u32(&enc, 4, n as u32);
            set_u32(&enc, 5, d as u32);
            set_f32(&enc, 6, inv_n);
            set_f32(&enc, 7, b);
            set_u32(&enc, 8, n_tg as u32);
            enc.set_threadgroup_memory_length(0, (8 * (d + 2) * std::mem::size_of::<f32>()) as u64);

            let tg = MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            };
            let grp = MTLSize {
                width: n_tg,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grp, tg);

            // Kernel B: deterministic reduction of the partials.
            enc.set_compute_pipeline_state(&reduce);
            enc.set_buffer(0, Some(&partials_buf), 0);
            enc.set_buffer(1, Some(&gw_buf), 0);
            enc.set_buffer(2, Some(&gb_buf), 0);
            enc.set_buffer(3, Some(&loss_buf), 0);
            set_u32(&enc, 4, n_tg as u32);
            set_u32(&enc, 5, d as u32);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                tg,
            );

            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            gw.copy_from_slice(&ctx.read_buffer::<f32>(&gw_buf, d));
            *gb = ctx.read_buffer::<f32>(&gb_buf, 1)[0];
            let loss_sum = ctx.read_buffer::<f32>(&loss_buf, 1)[0];

            // L2 gradient + normalized loss (matches sklearn/flashlib).
            if c_inv > 0.0 {
                for j in 0..d {
                    gw[j] += c_inv * w[j];
                }
            }
            *loss = loss_sum * inv_n + 0.5 * c_inv * w.iter().map(|v| v * v).sum::<f32>();
        };

        // ── Iteration 0: analytical Newton step from w = 0 (flashlib) ──
        // One Newton iteration on a zero-mean Gaussian gives the exact
        // regularized L2 minimizer; its gradient seeds the L-BFGS history.
        eval(&w, b, &mut grad_w, &mut grad_b, &mut loss_val);
        let mut grad_aug = vec![0.0f32; d + 1];
        for j in 0..d {
            grad_aug[j] = grad_w[j];
        }
        grad_aug[d] = grad_b;
        let grad0 = grad_aug.clone();

        // Curvature: Lpp = 0.25 * ||X@g + g_b||^2 / n + C_inv * ||g_w||^2
        // (predict kernel reused as a GPU matvec for X @ grad_w).
        ctx.write_buffer(&w_buf, &grad_w);
        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&predict);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&logits_buf), 0);
        set_u32(&enc, 3, n as u32);
        set_u32(&enc, 4, d as u32);
        set_f32(&enc, 5, grad_b);
        let tg = MTLSize {
            width: TG,
            height: 1,
            depth: 1,
        };
        let grp = MTLSize {
            width: ((n as u64 + TG - 1) / TG),
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grp, tg);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        let logits: Vec<f32> = ctx.read_buffer(&logits_buf, n);
        let ssq: f32 = logits.iter().map(|v| v * v).sum();
        let g_sq: f32 = grad_aug.iter().map(|v| v * v).sum();
        let lpp = 0.25 * ssq * inv_n + c_inv * grad_w.iter().map(|v| v * v).sum::<f32>();
        let step = g_sq / (lpp + 1e-20);

        let mut w_aug = vec![0.0f32; d + 1];
        for j in 0..(d + 1) {
            w_aug[j] = -step * grad_aug[j];
        }

        // Evaluate at the analytical step (seeds the L-BFGS history).
        w.copy_from_slice(&w_aug[..d]);
        b = w_aug[d];
        eval(&w, b, &mut grad_w, &mut grad_b, &mut loss_val);
        for j in 0..d {
            grad_aug[j] = grad_w[j];
        }
        grad_aug[d] = grad_b;
        self.n_epochs = 1;
        self.final_loss = loss_val;
        eprintln!("  epoch {:3}: loss={:.6}", 0, loss_val);

        let mut s_list: Vec<Vec<f32>> = vec![w_aug.clone()];
        let mut y_list: Vec<Vec<f32>> = vec![grad_aug
            .iter()
            .zip(grad0.iter())
            .map(|(g, g0)| g - g0)
            .collect()];
        let sy0 = dot(&s_list[0], &y_list[0]);
        let mut rho_list: Vec<f32> = vec![1.0 / sy0.max(1e-10)];

        let mut gmax = grad_aug.iter().fold(0.0f32, |m, g| m.max(g.abs()));

        // ── L-BFGS (m = 10) iterations with backtracking line search ──
        // flashlib's optimizer takes the two-loop direction unmodified; on
        // this data that diverges, so each iteration backs off the step
        // until the Armijo condition holds on the GPU-evaluated loss.
        for it in 1..self.config.max_epochs {
            if gmax < self.config.tol {
                break;
            }

            let mut dir = lbfgs_two_loop(&grad_aug, &s_list, &y_list, &rho_list);
            let mut dir_grad = dot(&grad_aug, &dir);
            if !(dir_grad < 0.0) {
                // Not a descent direction: fall back to steepest descent.
                dir = grad_aug.iter().map(|g| -g).collect();
                dir_grad = -grad_aug.iter().map(|g| g * g).sum::<f32>();
            }
            let prev_loss = loss_val;

            // ── Backtracking line search (Armijo) on the GPU loss ──
            let mut alpha = 1.0f32;
            let mut w_cand = vec![0.0f32; d + 1];
            let mut grad_new = vec![0.0f32; d + 1];
            let mut loss_new = f32::INFINITY;
            let mut accepted = false;
            for _ in 0..32 {
                for j in 0..(d + 1) {
                    w_cand[j] = w_aug[j] + alpha * dir[j];
                }
                w.copy_from_slice(&w_cand[..d]);
                b = w_cand[d];
                eval(&w, b, &mut grad_w, &mut grad_b, &mut loss_new);
                for j in 0..d {
                    grad_new[j] = grad_w[j];
                }
                grad_new[d] = grad_b;
                if loss_new.is_finite() && loss_new <= prev_loss + 1e-4 * alpha * dir_grad {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                eprintln!("  line search failed at iteration {}; stopping", it);
                break;
            }

            let s: Vec<f32> = w_cand
                .iter()
                .zip(w_aug.iter())
                .map(|(x, y)| x - y)
                .collect();
            let yd: Vec<f32> = grad_new
                .iter()
                .zip(grad_aug.iter())
                .map(|(x, y)| x - y)
                .collect();
            let sy = dot(&s, &yd);
            if sy > 1e-12 {
                if s_list.len() >= m_lbfgs {
                    s_list.remove(0);
                    y_list.remove(0);
                    rho_list.remove(0);
                }
                s_list.push(s);
                y_list.push(yd);
                rho_list.push(1.0 / sy);
            }

            w_aug = w_cand;
            grad_aug = grad_new;
            loss_val = loss_new;
            self.n_epochs = it + 1;
            self.final_loss = loss_val;

            gmax = grad_aug.iter().fold(0.0f32, |m, g| m.max(g.abs()));
            if !gmax.is_finite() {
                eprintln!(
                    "  diverged at iteration {} (grad={:.2e}); stopping",
                    it, gmax
                );
                break;
            }
            if it % 10 == 0 || it == self.config.max_epochs - 1 {
                eprintln!("  epoch {:3}: loss={:.6} grad={:.2e}", it, loss_val, gmax);
            }
        }

        self.weights = w_aug[..d].to_vec();
        self.bias = w_aug[d];
        Ok(())
    }

    // ── Predict probabilities ──────────────────────────────────────

    /// Predict class probabilities for input data using GPU.
    pub fn predict_proba(
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

        let pipeline = self.pipelines.get(ctx, "logreg_predict")?;
        let x_buf = ctx.new_buffer(data);
        let w_buf = ctx.new_buffer(&self.weights);
        let out_buf = ctx.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

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
        let tg = MTLSize {
            width: TG,
            height: 1,
            depth: 1,
        };
        let grp = MTLSize {
            width: ((n as u64 + TG - 1) / TG),
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grp, tg);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let probs: Vec<f32> = ctx.read_buffer(&out_buf, n);
        Ok(probs)
    }

    /// Predict hard class labels (0 or 1) using GPU.
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let probs = self.predict_proba(ctx, data, n, d)?;
        Ok(probs
            .into_iter()
            .map(|p| if p >= 0.5 { 1.0 } else { 0.0 })
            .collect())
    }

    /// Compute accuracy score.
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        let y_pred = self.predict(ctx, data, n, d)?;
        let correct: usize = y_pred
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 0.01)
            .count();
        Ok(correct as f32 / n as f32)
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

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L-BFGS two-loop recursion (Nocedal & Wright Alg. 7.4) on the
/// augmented (d + 1)-vector [w | b]. `rho[i] = 1 / (s_i · y_i)`.
/// Returns the descent search direction `-H · ∇f`.
fn lbfgs_two_loop(
    grad: &[f32],
    s_list: &[Vec<f32>],
    y_list: &[Vec<f32>],
    rho_list: &[f32],
) -> Vec<f32> {
    let m = s_list.len();
    let mut q = grad.to_vec();

    // First loop: q = g_k; alpha_i = rho_i * s_i^T q; q -= alpha_i * y_i
    let mut alpha = vec![0.0f32; m];
    for i in (0..m).rev() {
        alpha[i] = rho_list[i] * dot(&s_list[i], &q);
        for j in 0..q.len() {
            q[j] -= alpha[i] * y_list[i][j];
        }
    }

    // Initial Hessian approximation: gamma = s^T y / y^T y
    let last = m - 1;
    let yy = dot(&y_list[last], &y_list[last]);
    let gamma = dot(&s_list[last], &y_list[last]) / (yy + 1e-20);

    // Second loop: r = gamma * q; beta_i = rho_i * y_i^T r; r += s_i * (alpha_i - beta_i)
    let mut r = vec![0.0f32; q.len()];
    for j in 0..q.len() {
        r[j] = gamma * q[j];
    }
    for i in 0..m {
        let beta = rho_list[i] * dot(&y_list[i], &r);
        for j in 0..r.len() {
            r[j] += s_list[i][j] * (alpha[i] - beta);
        }
    }
    // r = H_k · ∇f_k; the search direction is its negation.
    for v in r.iter_mut() {
        *v = -*v;
    }
    r
}
