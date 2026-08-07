//! t-Distributed Stochastic Neighbor Embedding (t-SNE) with GPU-accelerated
//! exact (O(N²)) affinity and gradient computation.
//!
//! Exact t-SNE has two dominant costs:
//!
//! 1. Building the pairwise Gaussian affinities `P` (N×N) from the input —
//!    an O(N²·D) cost split into a batched squared-L2 distance kernel
//!    (`tsne_distances`) and a per-row, embarrassingly-parallel perplexity
//!    bisection kernel (`tsne_perplexity`).
//! 2. The O(N²) t-SNE gradient on the low-dimensional embedding, computed once
//!    per iteration by `tsne_grad`. Every thread owns one point's output row,
//!    so there are no intra-kernel races; the `÷Z` Student-t normalization is
//!    folded in on the host before the (CPU, O(N)) momentum update.
//!
//! Semantics mirror scikit-learn's exact `TSNE` (`method="exact"`): symmetric
//! affinities `P = (P_{j|i} + P_{i|j}) / (2·N)` (diagonal zero), early
//! exaggeration by `early_exaggeration` for the first `exaggeration_iter`
//! iterations, and the classic velocity momentum update
//! `v ← momentum·v − lr·grad; Y ← Y + v`. The embedding is a random Gaussian
//! (0, 1e-4) initialized from a caller-seeded PRNG for reproducibility.

use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/tsne.metal");

// ── Pipeline cache ────────────────────────────────────────────────

struct PipelineCache {
    distances: OnceLock<ComputePipelineState>,
    perplexity: OnceLock<ComputePipelineState>,
    grad: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            distances: OnceLock::new(),
            perplexity: OnceLock::new(),
            grad: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "tsne_distances" => &self.distances,
            "tsne_perplexity" => &self.perplexity,
            "tsne_grad" => &self.grad,
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

/// Configuration for t-SNE.
#[derive(Clone, Debug)]
pub struct TSNEConfig {
    /// Number of embedding dimensions (1..=8; 2 is the classic visualisation
    /// choice). Kept small so the gradient kernel loops over a bounded
    /// component range.
    pub n_components: usize,
    /// Target perplexity: the (effective) number of neighbors-per-point
    /// balanced by the per-row Gaussian bandwidth. Must satisfy
    /// `1 <= perplexity < n_samples`.
    pub perplexity: f32,
    /// Gradient descent step size.
    pub learning_rate: f32,
    /// Number of gradient descent iterations.
    pub n_iter: usize,
    /// Early-exaggeration multiplier applied to `P` for the first
    /// `exaggeration_iter` iterations (>= 1.0; 1.0 disables exaggeration).
    pub early_exaggeration: f32,
    /// Number of iterations for which `early_exaggeration` is active.
    pub exaggeration_iter: usize,
    /// Momentum (velocity) coefficient of the update.
    pub momentum: f32,
    /// Seed for the Gaussian-initialization PRNG (reproducibility).
    pub seed: u64,
    /// Early-stopping threshold on the gradient L2 norm (0.0 disables).
    pub min_grad_norm: f32,
}

impl Default for TSNEConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            early_exaggeration: 12.0,
            exaggeration_iter: 250,
            momentum: 0.8,
            seed: 42,
            min_grad_norm: 1e-7,
        }
    }
}

// ── Operator ──────────────────────────────────────────────────────

/// t-SNE: nonlinear dimensionality reduction / embedding.
pub struct TSNE {
    config: TSNEConfig,
    /// Low-dimensional embedding, row-major `(n, n_components)`.
    embedding: Vec<f32>,
    n: usize,
    d: usize,
    /// Number of gradient iterations actually performed.
    n_iter_: usize,
    /// Final KL divergence (KL(P ‖ Q)) computed on the host.
    kl_divergence: f32,
}

impl TSNE {
    pub fn new(config: TSNEConfig) -> Self {
        Self {
            config,
            embedding: Vec::new(),
            n: 0,
            d: 0,
            n_iter_: 0,
            kl_divergence: 0.0,
        }
    }

    /// Low-dimensional embedding, row-major `(n, n_components)`.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Number of global iterations performed during the last `fit`.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Final KL(P ‖ Q) divergence of the embedding (smaller is better).
    pub fn kl_divergence(&self) -> f32 {
        self.kl_divergence
    }

    /// Fit the embedding on `data` (`n` samples × `d` features).
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        let nc = self.config.n_components;
        anyhow::ensure!(n > 1, "t-SNE needs at least 2 samples, got {}", n);
        anyhow::ensure!(d > 0, "t-SNE needs d >= 1, got {}", d);
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        anyhow::ensure!(
            (1..=8).contains(&nc),
            "n_components must be in 1..=8, got {}",
            nc
        );
        let perplexity = self.config.perplexity;
        anyhow::ensure!(
            perplexity >= 1.0 && perplexity < n as f32,
            "perplexity must satisfy 1 <= perplexity < n (n={}), got {}",
            n,
            perplexity
        );
        let lr = self.config.learning_rate;
        anyhow::ensure!(lr > 0.0 && lr.is_finite(), "learning_rate must be > 0");
        anyhow::ensure!(self.config.n_iter > 0, "n_iter must be > 0");
        let exag = self.config.early_exaggeration;
        anyhow::ensure!(
            exag >= 1.0 && exag.is_finite(),
            "early_exaggeration must be >= 1.0"
        );

        self.n = n;
        self.d = d;

        let cache = PipelineCache::new();
        let dist_p = cache.get(ctx, "tsne_distances")?;
        let perp_p = cache.get(ctx, "tsne_perplexity")?;
        let grad_p = cache.get(ctx, "tsne_grad")?;

        // 1) Full squared-L2 distance matrix D (N×N) on the GPU.
        let data_buf = ctx.new_buffer(data);
        let d_buf = ctx.new_buffer_uninitialized((n * n * 4) as u64);
        {
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&dist_p);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(&d_buf), 0);
            set_u32(&enc, 2, n as u32);
            set_u32(&enc, 3, d as u32);
            dispatch_1d(&enc, n * n);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // 2) Perplexity bisection -> conditional P (N×N).
        let p_buf = ctx.new_buffer_uninitialized((n * n * 4) as u64);
        {
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(perp_p);
            enc.set_buffer(0, Some(&d_buf), 0);
            enc.set_buffer(1, Some(&p_buf), 0);
            set_u32(&enc, 2, n as u32);
            set_f32(&enc, 3, perplexity);
            dispatch_1d(&enc, n);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let p_cond: Vec<f32> = ctx.read_buffer(&p_buf, n * n);

        // 3) Symmetrize + normalize: P = (P_{j|i} + P_{i|j}) / (2n), diag 0.
        let mut pmat = vec![0.0f32; n * n];
        let inv2n = 1.0 / (2.0 * n as f32);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                pmat[i * n + j] = (p_cond[i * n + j] + p_cond[j * n + i]) * inv2n;
            }
        }
        ctx.write_buffer(&p_buf, &pmat);

        // 4) Random Gaussian (0, 1e-4) embedding init + zero velocity.
        let mut rng = fastrand::Rng::with_seed(self.config.seed);
        let mut y = vec![0.0f32; n * nc];
        for v in y.iter_mut() {
            *v = 1e-4 * gaussian(&mut rng);
        }
        let mut vel = vec![0.0f32; n * nc];
        let zeros = vec![0.0f32; n * nc];

        let y_buf = ctx.new_buffer(&y);
        let a1_buf = ctx.new_buffer_uninitialized((n * nc * 4) as u64);
        let a2_buf = ctx.new_buffer_uninitialized((n * nc * 4) as u64);
        let z_buf = ctx.new_buffer_uninitialized((n * 4) as u64);

        let exag_iters = self.config.exaggeration_iter;
        let momentum = self.config.momentum;
        let min_grad_norm = self.config.min_grad_norm;

        let mut n_iter_run = self.config.n_iter;
        for t in 0..self.config.n_iter {
            let e: f32 = if t < exag_iters { exag } else { 1.0 };
            ctx.write_buffer(&y_buf, &y);
            ctx.write_buffer(&a1_buf, &zeros);
            ctx.write_buffer(&a2_buf, &zeros);

            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&grad_p);
            enc.set_buffer(0, Some(&y_buf), 0);
            enc.set_buffer(1, Some(&p_buf), 0);
            enc.set_buffer(2, Some(&a1_buf), 0);
            enc.set_buffer(3, Some(&a2_buf), 0);
            enc.set_buffer(4, Some(&z_buf), 0);
            set_u32(&enc, 5, n as u32);
            set_u32(&enc, 6, nc as u32);
            set_f32(&enc, 7, e);
            dispatch_1d(&enc, n);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let a1: Vec<f32> = ctx.read_buffer(&a1_buf, n * nc);
            let a2: Vec<f32> = ctx.read_buffer(&a2_buf, n * nc);
            let zbuf: Vec<f32> = ctx.read_buffer(&z_buf, n);
            let z: f32 = zbuf.iter().copied().sum::<f32>().max(1e-12);
            let inv_z = 1.0 / z;

            let mut grad_norm_sq = 0.0f32;
            for i in 0..n {
                for c in 0..nc {
                    let idx = i * nc + c;
                    let g = 4.0 * (a1[idx] - a2[idx] * inv_z);
                    grad_norm_sq += g * g;
                    vel[idx] = momentum * vel[idx] - lr * g;
                }
            }
            for v in y.iter_mut().zip(vel.iter()) {
                // y[idx] += vel[idx]
                *(v.0) += *v.1;
            }

            if min_grad_norm > 0.0 && grad_norm_sq.sqrt() < min_grad_norm {
                n_iter_run = t + 1;
                break;
            }
        }

        self.embedding = y.clone();
        self.n_iter_ = n_iter_run;
        self.kl_divergence = compute_kl(&pmat, &y, n, nc);
        Ok(())
    }
}

/// Final KL(P ‖ Q) for diagnostics: Σ_{i≠j} P_ij · ln(P_ij / Q_ij),
/// with `Q_ij = q_ij / Z`, `q_ij = (1 + ‖y_i−y_j‖²)⁻¹`, `Z = Σ_{i≠j} q_ij`.
fn compute_kl(p: &[f32], y: &[f32], n: usize, nc: usize) -> f32 {
    let mut z = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            z += 1.0 / (1.0 + sq_dist(y, i, j, nc));
        }
    }
    let mut kl = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let pij = p[i * n + j];
            if pij <= 0.0 {
                continue;
            }
            let q = 1.0 / (1.0 + sq_dist(y, i, j, nc));
            let qij = q / z;
            kl += pij * (pij / qij).ln();
        }
    }
    kl
}

fn sq_dist(m: &[f32], i: usize, j: usize, nc: usize) -> f32 {
    let mut s = 0.0f32;
    for c in 0..nc {
        let dc = m[i * nc + c] - m[j * nc + c];
        s += dc * dc;
    }
    s
}

/// Standard normal sample via Box-Muller over fastrand uniform draws.
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = (rng.f32()).max(1e-38);
    let u2 = rng.f32();
    let mag = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    mag * theta.cos()
}

// ── Dispatch helpers ──────────────────────────────────────────────

fn dispatch_1d(enc: &ComputeCommandEncoderRef, total: usize) {
    let groups = MTLSize {
        width: (total as u64 + 255) / 256,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
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
