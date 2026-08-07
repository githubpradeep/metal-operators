//! Non-negative Matrix Factorization (NMF) with GPU matmul kernels.
//!
//! Factorizes a non-negative data matrix `V` (N×D) into `V ≈ W·H` with
//! `W` (N×K) and `H` (K×D) both non-negative, using the classic
//! Lee–Seung multiplicative-update rules:
//!
//! ```text
//! H ← H ⊙ (Wᵀ·V)   / (Wᵀ·W·H + ε)
//! W ← W ⊙ (V·Hᵀ)   / (W·H·Hᵀ + ε)
//! ```
//!
//! Every O(N·D·K) step is a matrix multiply; the GPU shader
//! (`shaders/nmf.metal`) therefore provides a single generic `nmf_mm`
//! matmul kernel (with a transpose flag on either operand) plus one
//! elementwise `nmf_update` kernel that applies the multiplicative rule.
//! `Wᵀ·V`, `Wᵀ·W`, `Wᵀ·W·H`, `V·Hᵀ`, `H·Hᵀ` and `W·H·Hᵀ` are all
//! dispatched through the same matmul kernel, and the final reconstruction
//! error `‖V − W·H‖_F` is computed with the `nmf_diff` kernel.

use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/nmf.metal");

const DEFAULT_EPS: f32 = 1e-10;

// ── Pipeline cache ───────────────────────────────────────────────

struct PipelineCache {
    mm: OnceLock<ComputePipelineState>,
    update: OnceLock<ComputePipelineState>,
    diff: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            mm: OnceLock::new(),
            update: OnceLock::new(),
            diff: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "nmf_mm" => &self.mm,
            "nmf_update" => &self.update,
            "nmf_diff" => &self.diff,
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

// ── Config ───────────────────────────────────────────────────────

/// Configuration for non-negative matrix factorization.
#[derive(Clone, Debug)]
pub struct NMFConfig {
    /// Number of latent components (rank `K`). Clamped at runtime to
    /// `min(n, d)`.
    pub n_components: usize,
    /// Maximum number of multiplicative-update iterations.
    pub max_iterations: usize,
    /// Early-stop threshold on the relative Frobenius change of `W`
    /// between consecutive iterations (`0.0` disables early stopping).
    pub tolerance: f32,
    /// RNG seed for the non-negative random init of `W` and `H`.
    pub seed: u64,
    /// Small additive constant guarding division by zero in the update.
    pub eps: f32,
}

impl Default for NMFConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iterations: 200,
            tolerance: 1e-4,
            seed: 42,
            eps: DEFAULT_EPS,
        }
    }
}

// ── NMF ──────────────────────────────────────────────────────────

/// Non-negative matrix factorization: `V ≈ W·H` with `W, H ≥ 0`.
///
/// `fit` runs the multiplicative updates with every matmul offloaded to
/// the GPU (`nmf_mm`); the small per-iteration convergence check (relative
/// change of `W`) is done on the host after reading `W` back.
pub struct NMF {
    config: NMFConfig,
    /// Fitted components `H` (K×D, row-major).
    components: Vec<f32>,
    /// Coefficient matrix `W` (N×K, row-major) — the latent embedding of
    /// the training data.
    coeff: Vec<f32>,
    /// Final reconstruction error `‖V − W·H‖_F`.
    reconstruction_error: f32,
    /// Number of multiplicative-update iterations actually run.
    n_iter: usize,
    n: usize,
    d: usize,
    k: usize,
    pipelines: PipelineCache,
}

impl NMF {
    pub fn new(config: NMFConfig) -> Self {
        Self {
            config,
            components: Vec::new(),
            coeff: Vec::new(),
            reconstruction_error: 0.0,
            n_iter: 0,
            n: 0,
            d: 0,
            k: 0,
            pipelines: PipelineCache::new(),
        }
    }

    /// Fitted components `H` (K×D, row-major).
    pub fn components(&self) -> &[f32] {
        &self.components
    }

    /// Coefficient matrix `W` (N×K) — latent representation of the
    /// training data (the `fit_transform` output).
    pub fn coeff(&self) -> &[f32] {
        &self.coeff
    }

    /// Frobenius norm of the reconstruction residual `V − W·H`.
    pub fn reconstruction_error(&self) -> f32 {
        self.reconstruction_error
    }

    /// Number of iterations run by the last `fit`.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    pub fn n_samples(&self) -> usize {
        self.n
    }

    pub fn n_features(&self) -> usize {
        self.d
    }

    /// Fit NMF by multiplicative updates; every matmul runs on the GPU.
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
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
        anyhow::ensure!(
            data.iter().all(|&x| x >= 0.0),
            "NMF requires a non-negative data matrix (found negative values)"
        );
        let k = self.config.n_components.clamp(1, n.min(d));
        anyhow::ensure!(k > 0, "n_components must be >= 1");
        let max_iterations = self.config.max_iterations.max(1);
        let eps = self.config.eps;

        self.n = n;
        self.d = d;
        self.k = k;

        // Non-negative random init, scaled like sklearn (`random` init).
        let mut rng = fastrand::Rng::with_seed(self.config.seed);
        let scale = (data.iter().sum::<f32>() / (n * d) as f32)
            .max(1e-12)
            .sqrt()
            / (k as f32).sqrt();
        let mut w = vec![0.0f32; n * k];
        let mut h = vec![0.0f32; k * d];
        for v in w.iter_mut() {
            *v = rng.f32() * scale;
        }
        for v in h.iter_mut() {
            *v = rng.f32() * scale;
        }

        // ── Allocate GPU buffers once (reused across iterations) ──
        let v_buf = ctx.new_buffer(data);
        let w_buf = ctx.new_buffer(&w);
        let h_buf = ctx.new_buffer(&h);
        let wtv_buf = ctx.new_buffer_uninitialized((k * d * 4) as u64);
        let wtw_buf = ctx.new_buffer_uninitialized((k * k * 4) as u64);
        let wtwh_buf = ctx.new_buffer_uninitialized((k * d * 4) as u64);
        let vht_buf = ctx.new_buffer_uninitialized((n * k * 4) as u64);
        let hht_buf = ctx.new_buffer_uninitialized((k * k * 4) as u64);
        let whht_buf = ctx.new_buffer_uninitialized((n * k * 4) as u64);

        let mm = self.pipelines.get(ctx, "nmf_mm")?;
        let update = self.pipelines.get(ctx, "nmf_update")?;

        let mut n_iter = 0usize;
        let mut w_prev = w.clone();

        for it in 0..max_iterations {
            // H update: WᵀV, WᵀW, (WᵀW)·H, then H ← H⊙WᵀV/(WᵀWH+ε)
            dispatch_mm(ctx, mm, &w_buf, &v_buf, &wtv_buf, k, d, n, true, false);
            dispatch_mm(ctx, mm, &w_buf, &w_buf, &wtw_buf, k, k, n, true, false);
            dispatch_mm(ctx, mm, &wtw_buf, &h_buf, &wtwh_buf, k, d, k, false, false);
            dispatch_update(ctx, update, &h_buf, &wtv_buf, &wtwh_buf, eps, k, d);

            // W update: V·Hᵀ, H·Hᵀ, W·(H·Hᵀ), then W ← W⊙VHᵀ/(WHHᵀ+ε)
            dispatch_mm(ctx, mm, &v_buf, &h_buf, &vht_buf, n, k, d, false, true);
            dispatch_mm(ctx, mm, &h_buf, &h_buf, &hht_buf, k, k, d, false, true);
            dispatch_mm(ctx, mm, &w_buf, &hht_buf, &whht_buf, n, k, k, false, false);
            dispatch_update(ctx, update, &w_buf, &vht_buf, &whht_buf, eps, n, k);

            n_iter = it + 1;

            // Convergence: relative change of W's Frobenius norm.
            let w_new: Vec<f32> = ctx.read_buffer(&w_buf, n * k);
            let mut denom = 0.0f64;
            let mut delta = 0.0f64;
            for i in 0..w_new.len() {
                denom += w_prev[i] as f64 * w_prev[i] as f64;
                let dd = (w_new[i] - w_prev[i]) as f64;
                delta += dd * dd;
            }
            if self.config.tolerance > 0.0 && denom > 0.0 {
                let rel = (delta.sqrt() / denom.sqrt()) as f32;
                if rel < self.config.tolerance {
                    break;
                }
            }
            w_prev = w_new;
        }

        // Read final factors back.
        let final_w: Vec<f32> = ctx.read_buffer(&w_buf, n * k);
        let final_h: Vec<f32> = ctx.read_buffer(&h_buf, k * d);

        // Reconstruction error via the nmf_diff kernel.
        let err_buf = ctx.new_buffer_uninitialized((n * d * 4) as u64);
        {
            let diff = self.pipelines.get(ctx, "nmf_diff")?;
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(diff);
            enc.set_buffer(0, Some(&v_buf), 0);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&h_buf), 0);
            enc.set_buffer(3, Some(&err_buf), 0);
            set_u32(&enc, 4, n as u32);
            set_u32(&enc, 5, d as u32);
            set_u32(&enc, 6, k as u32);
            dispatch_2d(&enc, n, d);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let errs: Vec<f32> = ctx.read_buffer(&err_buf, n * d);
        let reconstruction_error = errs.iter().map(|&e| e * e).sum::<f32>().sqrt();

        self.coeff = final_w;
        self.components = final_h;
        self.reconstruction_error = reconstruction_error;
        self.n_iter = n_iter;

        Ok(())
    }

    /// Project new data into the latent space: solve `X ≈ W_new·H` with the
    /// fitted components `H` kept fixed (multiplicative updates on `W_new`).
    pub fn transform(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(!self.components.is_empty(), "NMF must be fitted first");
        anyhow::ensure!(
            d == self.d,
            "Feature count mismatch: expected {}, got {}",
            self.d,
            d
        );
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        anyhow::ensure!(
            data.iter().all(|&x| x >= 0.0),
            "NMF requires non-negative data"
        );
        let k = self.k;
        let eps = self.config.eps;

        let mut rng = fastrand::Rng::with_seed(self.config.seed ^ 0x9e3779b9);
        let scale = (data.iter().sum::<f32>() / (n * d) as f32)
            .max(1e-12)
            .sqrt()
            / (k as f32).sqrt();
        let mut w_new = vec![0.0f32; n * k];
        for v in w_new.iter_mut() {
            *v = rng.f32() * scale;
        }

        let x_buf = ctx.new_buffer(data);
        let w_buf = ctx.new_buffer(&w_new);
        let h_buf = ctx.new_buffer(&self.components);
        let xht_buf = ctx.new_buffer_uninitialized((n * k * 4) as u64);
        let hht_buf = ctx.new_buffer_uninitialized((k * k * 4) as u64);
        let whht_buf = ctx.new_buffer_uninitialized((n * k * 4) as u64);

        let mm = self.pipelines.get(ctx, "nmf_mm")?;
        let update = self.pipelines.get(ctx, "nmf_update")?;

        // X·Hᵀ and H·Hᵀ are constant across iterations.
        dispatch_mm(ctx, mm, &x_buf, &h_buf, &xht_buf, n, k, d, false, true);
        dispatch_mm(ctx, mm, &h_buf, &h_buf, &hht_buf, k, k, d, false, true);

        let max_iterations = self.config.max_iterations.max(1);
        for _ in 0..max_iterations {
            dispatch_mm(ctx, mm, &w_buf, &hht_buf, &whht_buf, n, k, k, false, false);
            dispatch_update(ctx, update, &w_buf, &xht_buf, &whht_buf, eps, n, k);
        }

        Ok(ctx.read_buffer(&w_buf, n * k))
    }
}

// ── Dispatch helpers ─────────────────────────────────────────────

/// C (M×N) = A (M×K) · B (K×N), with optional per-operand transpose.
fn dispatch_mm(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    a_trans: bool,
    b_trans: bool,
) {
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(c), 0);
    set_u32(&enc, 3, m as u32);
    set_u32(&enc, 4, n as u32);
    set_u32(&enc, 5, k as u32);
    set_u32(&enc, 6, a_trans as u32);
    set_u32(&enc, 7, b_trans as u32);
    dispatch_2d(&enc, m, n);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// X (M×N) ← X ⊙ num / (den + eps) in place.
fn dispatch_update(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    x: &Buffer,
    num: &Buffer,
    den: &Buffer,
    eps: f32,
    m: usize,
    n: usize,
) {
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(num), 0);
    enc.set_buffer(2, Some(den), 0);
    set_f32(&enc, 3, eps);
    set_u32(&enc, 4, m as u32);
    set_u32(&enc, 5, n as u32);
    dispatch_2d(&enc, m, n);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn dispatch_2d(enc: &ComputeCommandEncoderRef, rows: usize, cols: usize) {
    let tg = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
    let groups = MTLSize {
        width: (cols as u64 + 15) / 16,
        height: (rows as u64 + 15) / 16,
        depth: 1,
    };
    enc.dispatch_thread_groups(groups, tg);
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
}

fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    let len = std::mem::size_of::<f32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
}
