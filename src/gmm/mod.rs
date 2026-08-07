//! Gaussian Mixture Model (GMM) clustering / density estimation with a
//! GPU-accelerated EM loop.
//!
//! Semantics mirror scikit-learn's `GaussianMixture` (full covariance, EM):
//!
//! 1. **Initialization** — k-means++ selects `n_components` seed means; each
//!    component's covariance starts as the empirical covariance of its nearest
//!    points (falling back to the global covariance for empty/small clusters),
//!    plus `reg_covar` on the diagonal.
//! 2. **E-step (GPU)** — `gmm_e_step` (`shaders/gmm.metal`) computes the
//!    (n × k) log-likelihood matrix `L[i][c] = log w_c + log N(x_i | μ_c, Σ_c)`
//!    in one launch. The host turns it into responsibilities
//!    `r_ic = softmax_c(L_i)` and the average log-likelihood lower bound.
//! 3. **M-step (host)** — weights, means, and full covariances are re-estimated
//!    from the responsibilities; a per-component Cholesky factorization
//!    rebuilds the precision `Σ_c⁻¹` and `log|Σ_c|` for the next E-step.
//! 4. Loop until `|lower_bound - previous| < tolerance` or `max_iterations`.
//!
//! `predict` / `predict_proba` / `score` reuse the same GPU E-step kernel
//! against the fitted parameters (a single launch each).

use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/gmm.metal");

// ── Pipeline cache ────────────────────────────────────────────────────────

struct PipelineCache {
    e_step: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            e_step: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "gmm_e_step" => &self.e_step,
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

/// Configuration for the Gaussian Mixture Model.
#[derive(Clone, Debug)]
pub struct GMMConfig {
    /// Number of mixture components (clusters).
    pub n_components: usize,
    /// Maximum number of EM iterations.
    pub max_iterations: usize,
    /// Convergence threshold on the change in the average log-likelihood
    /// lower bound between iterations.
    pub tolerance: f32,
    /// Seed for the k-means++ initialization PRNG (reproducible).
    pub seed: u64,
    /// Non-negative regularization added to the diagonal of every covariance
    /// matrix to keep it positive definite (mirrors sklearn's `reg_covar`).
    pub reg_covar: f32,
}

impl Default for GMMConfig {
    fn default() -> Self {
        Self {
            n_components: 3,
            max_iterations: 100,
            tolerance: 1e-3,
            seed: 42,
            reg_covar: 1e-6,
        }
    }
}

// ── Operator ──────────────────────────────────────────────────────────────

/// Gaussian Mixture Model fitted with GPU-accelerated EM.
pub struct GMM {
    config: GMMConfig,
    /// Per-component mixture weights (k,).
    weights: Vec<f32>,
    /// Per-component means, row-major (k × d).
    means: Vec<f32>,
    /// Per-component covariance matrices, row-major (k × d × d).
    covariances: Vec<f32>,
    /// Per-component precision matrices (Σ⁻¹), row-major (k × d × d).
    precisions: Vec<f32>,
    /// Per-component log-determinants log|Σ_c| (k,).
    log_dets: Vec<f32>,
    /// Final responsibilities from `fit`, row-major (n × k).
    responsibilities: Vec<f32>,
    /// Final average log-likelihood lower bound (matches sklearn `score`).
    lower_bound: f32,
    /// EM iterations actually performed by the last `fit`.
    n_iter_: usize,
    k: usize,
    d: usize,
    n: usize,
    pipelines: PipelineCache,
}

impl GMM {
    pub fn new(config: GMMConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
            precisions: Vec::new(),
            log_dets: Vec::new(),
            responsibilities: Vec::new(),
            lower_bound: 0.0,
            n_iter_: 0,
            k: 0,
            d: 0,
            n: 0,
            pipelines: PipelineCache::new(),
        }
    }

    pub fn n_components(&self) -> usize {
        self.k
    }
    pub fn n_features(&self) -> usize {
        self.d
    }
    pub fn n_samples(&self) -> usize {
        self.n
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn means(&self) -> &[f32] {
        &self.means
    }
    pub fn covariances(&self) -> &[f32] {
        &self.covariances
    }
    pub fn precisions(&self) -> &[f32] {
        &self.precisions
    }
    pub fn responsibilities(&self) -> &[f32] {
        &self.responsibilities
    }
    /// Average log-likelihood lower bound of the last `fit`.
    pub fn lower_bound(&self) -> f32 {
        self.lower_bound
    }
    /// EM iterations performed by the last `fit`.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Fit the mixture model with EM.
    ///
    /// * `ctx` - Metal context
    /// * `data` - Flat row-major data array (n × d)
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        let k = self.config.n_components;
        anyhow::ensure!(n > 0 && d > 0, "Data must be non-empty");
        anyhow::ensure!(
            k >= 1 && k <= n,
            "n_components must be in 1..=n, got {} (n={})",
            k,
            n
        );
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        anyhow::ensure!(
            self.config.tolerance >= 0.0 && self.config.tolerance.is_finite(),
            "tolerance must be a finite, non-negative value"
        );
        anyhow::ensure!(
            self.config.reg_covar >= 0.0 && self.config.reg_covar.is_finite(),
            "reg_covar must be a finite, non-negative value"
        );
        anyhow::ensure!(self.config.max_iterations > 0, "max_iterations must be > 0");

        self.k = k;
        self.d = d;
        self.n = n;

        // ── Initialization: k-means++ means, empirical per-cluster covariances ──
        self.init_params(ctx, data, n, d)?;

        let e_step_p = self.pipelines.get(ctx, "gmm_e_step")?.clone();
        let mut prev_lb = f32::NEG_INFINITY;
        let mut iter = 0usize;
        while iter < self.config.max_iterations {
            // E-step on the GPU: (n × k) log-likelihood matrix.
            let loglik = self.e_step(ctx, &e_step_p, data, n, d)?;

            // Responsibilities + lower bound on the host.
            let mut logsumexp = vec![0.0f32; n];
            let mut resp = vec![0.0f32; n * k];
            let mut lb = 0.0f32;
            for i in 0..n {
                let row = &loglik[i * k..(i + 1) * k];
                let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse = maxv + row.iter().map(|v| (*v - maxv).exp()).sum::<f32>().ln();
                logsumexp[i] = lse;
                lb += lse;
                for c in 0..k {
                    resp[i * k + c] = (loglik[i * k + c] - lse).exp();
                }
            }
            lb /= n as f32;

            // Convergence: |Δ lower bound| < tolerance (mirrors sklearn).
            if iter > 0 && (lb - prev_lb).abs() < self.config.tolerance {
                self.lower_bound = lb;
                self.responsibilities = resp;
                self.n_iter_ = iter + 1;
                return Ok(());
            }
            prev_lb = lb;

            // M-step on the host: weights, means, covariances.
            self.m_step(data, n, d, &resp);
            // Rebuild precision + logdet from the new covariances.
            self.rebuild_precisions()?;

            iter += 1;
        }

        // Final E-step so stored responsibilities match the fitted parameters.
        let loglik = self.e_step(ctx, &e_step_p, data, n, d)?;
        let mut lb = 0.0f32;
        let mut resp = vec![0.0f32; n * k];
        for i in 0..n {
            let row = &loglik[i * k..(i + 1) * k];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = maxv + row.iter().map(|v| (*v - maxv).exp()).sum::<f32>().ln();
            lb += lse;
            for c in 0..k {
                resp[i * k + c] = (loglik[i * k + c] - lse).exp();
            }
        }
        self.lower_bound = lb / n as f32;
        self.responsibilities = resp;
        self.n_iter_ = iter;
        Ok(())
    }

    /// Predict the most likely component for each sample (single GPU launch).
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<usize>> {
        anyhow::ensure!(self.k > 0, "Model not fitted");
        anyhow::ensure!(
            d == self.d,
            "Feature count mismatch: model fitted with d={}, got d={}",
            self.d,
            d
        );
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        let resp = self.responsibilities_for(ctx, data, n, d)?;
        let mut preds = vec![0usize; n];
        for i in 0..n {
            let row = &resp[i * self.k..(i + 1) * self.k];
            let mut best = 0usize;
            let mut best_v = row[0];
            for c in 1..self.k {
                if row[c] > best_v {
                    best_v = row[c];
                    best = c;
                }
            }
            preds[i] = best;
        }
        Ok(preds)
    }

    /// Posterior component probabilities (n × k, row-major), matching sklearn.
    pub fn predict_proba(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.responsibilities_for(ctx, data, n, d)
    }

    /// Average log-likelihood of `data` under the fitted model (sklearn `score`).
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(self.k > 0, "Model not fitted");
        anyhow::ensure!(
            d == self.d,
            "Feature count mismatch: model fitted with d={}, got d={}",
            self.d,
            d
        );
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        let loglik = self.e_step(ctx, self.pipelines.get(ctx, "gmm_e_step")?, data, n, d)?;
        let mut lb = 0.0f32;
        for i in 0..n {
            let row = &loglik[i * self.k..(i + 1) * self.k];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = maxv + row.iter().map(|v| (*v - maxv).exp()).sum::<f32>().ln();
            lb += lse;
        }
        Ok(lb / n as f32)
    }

    // ── internals ──────────────────────────────────────────────────────────

    /// k-means++ means + per-cluster empirical covariances (with reg_covar).
    fn init_params(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        let k = self.k;
        let mut means = kmeanspp(data, n, d, k, self.config.seed);

        // Global mean + covariance (fallback for degenerate clusters).
        let mut gmean = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d {
                gmean[j] += data[i * d + j];
            }
        }
        for j in 0..d {
            gmean[j] /= n as f32;
        }
        let mut gcov = vec![0.0f32; d * d];
        for i in 0..n {
            for a in 0..d {
                for b in 0..d {
                    gcov[a * d + b] += (data[i * d + a] - gmean[a]) * (data[i * d + b] - gmean[b]);
                }
            }
        }
        for a in 0..d {
            for b in 0..d {
                gcov[a * d + b] /= n as f32;
            }
        }

        // Nearest-assignment per cluster; empty clusters fall back to global.
        let mut counts = vec![0usize; k];
        let mut sums = vec![0.0f32; k * d];
        let mut sumsq = vec![0.0f32; k * d * d];
        for i in 0..n {
            let c = nearest(&means, data, i, n, d);
            counts[c] += 1;
            for a in 0..d {
                sums[c * d + a] += data[i * d + a];
                for b in 0..d {
                    sumsq[(c * d + a) * d + b] += data[i * d + a] * data[i * d + b];
                }
            }
        }

        let mut covs = vec![0.0f32; k * d * d];
        let reg = self.config.reg_covar;
        for c in 0..k {
            if counts[c] >= d + 1 {
                // Empirical covariance of the cluster.
                for a in 0..d {
                    let ma = sums[c * d + a] / counts[c] as f32;
                    for b in 0..d {
                        let mb = sums[c * d + b] / counts[c] as f32;
                        let e_ab = sumsq[(c * d + a) * d + b] / counts[c] as f32;
                        covs[(c * d + a) * d + b] = e_ab - ma * mb;
                    }
                }
            } else {
                covs[c * d * d..(c + 1) * d * d].copy_from_slice(&gcov);
            }
            for a in 0..d {
                covs[(c * d + a) * d + a] += reg;
            }
            // Cluster mean (recompute rather than the raw seed point).
            if counts[c] > 0 {
                for a in 0..d {
                    means[c * d + a] = sums[c * d + a] / counts[c] as f32;
                }
            }
        }

        self.weights = vec![1.0 / k as f32; k];
        self.means = means;
        self.covariances = covs;
        // First precision/logdet pass (used by the first E-step).
        self.rebuild_precisions()?;
        // Warm up the GPU pipeline so the fit loop reuses it.
        let _ = self.pipelines.get(ctx, "gmm_e_step")?;
        Ok(())
    }

    /// M-step: weighted mean / covariance re-estimation from responsibilities.
    fn m_step(&mut self, data: &[f32], n: usize, d: usize, resp: &[f32]) {
        let k = self.k;
        let reg = self.config.reg_covar;
        let mut nk = vec![0.0f32; k];
        for i in 0..n {
            for c in 0..k {
                nk[c] += resp[i * k + c];
            }
        }
        let mut weights = vec![0.0f32; k];
        let mut means = vec![0.0f32; k * d];
        let mut covs = vec![0.0f32; k * d * d];
        for c in 0..k {
            let nc = nk[c];
            weights[c] = nc / n as f32;
            if nc <= 0.0 {
                continue;
            }
            for a in 0..d {
                let mut s = 0.0f32;
                for i in 0..n {
                    s += resp[i * k + c] * data[i * d + a];
                }
                means[c * d + a] = s / nc;
            }
            for a in 0..d {
                for b in 0..d {
                    let mut s = 0.0f32;
                    for i in 0..n {
                        let dxa = data[i * d + a] - means[c * d + a];
                        let dxb = data[i * d + b] - means[c * d + b];
                        s += resp[i * k + c] * dxa * dxb;
                    }
                    covs[(c * d + a) * d + b] = s / nc;
                }
            }
            for a in 0..d {
                covs[(c * d + a) * d + a] += reg;
            }
        }
        self.weights = weights;
        self.means = means;
        self.covariances = covs;
    }

    /// Cholesky each covariance; store precision (Σ⁻¹) and log|Σ|. Degenerate
    /// (non-PD) covariances get progressively more diagonal regularization.
    fn rebuild_precisions(&mut self) -> anyhow::Result<()> {
        let k = self.k;
        let d = self.d;
        let mut precisions = vec![0.0f32; k * d * d];
        let mut log_dets = vec![0.0f32; k];
        for c in 0..k {
            let mut cov = self.covariances[c * d * d..(c + 1) * d * d].to_vec();
            let mut extra = self.config.reg_covar;
            let mut l = cholesky(&cov, d);
            while l.is_none() && extra < 1e3 {
                extra *= 10.0;
                for a in 0..d {
                    cov[a * d + a] += extra;
                }
                l = cholesky(&cov, d);
            }
            let l = l.ok_or_else(|| {
                anyhow::anyhow!("component {} covariance not positive definite", c)
            })?;
            let (prec, logdet) = precision_and_logdet(&l, d);
            precisions[c * d * d..(c + 1) * d * d].copy_from_slice(&prec);
            log_dets[c] = logdet;
        }
        self.precisions = precisions;
        self.log_dets = log_dets;
        Ok(())
    }

    /// GPU E-step: (n × k) log-likelihood matrix for the given data.
    fn e_step(
        &self,
        ctx: &MetalContext,
        e_step_p: &ComputePipelineState,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let k = self.k;
        let logw: Vec<f32> = self.weights.iter().map(|w| w.ln()).collect();

        let data_buf = ctx.new_buffer(data);
        let means_buf = ctx.new_buffer(&self.means);
        let prec_buf = ctx.new_buffer(&self.precisions);
        let logdet_buf = ctx.new_buffer(&self.log_dets);
        let logw_buf = ctx.new_buffer(&logw);
        let out_buf = ctx.new_buffer_uninitialized((n * k * std::mem::size_of::<f32>()) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(e_step_p);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&means_buf), 0);
        enc.set_buffer(2, Some(&prec_buf), 0);
        enc.set_buffer(3, Some(&logdet_buf), 0);
        enc.set_buffer(4, Some(&logw_buf), 0);
        enc.set_buffer(5, Some(&out_buf), 0);
        set_u32(&enc, 6, n as u32);
        set_u32(&enc, 7, k as u32);
        set_u32(&enc, 8, d as u32);
        const TG: u64 = 256;
        let n_groups = (n as u64 + TG - 1) / TG;
        enc.dispatch_thread_groups(
            MTLSize {
                width: n_groups,
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

        Ok(ctx.read_buffer::<f32>(&out_buf, n * k))
    }

    /// Responsibilities (posterior probabilities, n × k) for arbitrary data.
    fn responsibilities_for(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let e_step_p = self.pipelines.get(ctx, "gmm_e_step")?;
        let loglik = self.e_step(ctx, &e_step_p, data, n, d)?;
        let k = self.k;
        let mut resp = vec![0.0f32; n * k];
        for i in 0..n {
            let row = &loglik[i * k..(i + 1) * k];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = maxv + row.iter().map(|v| (*v - maxv).exp()).sum::<f32>().ln();
            for c in 0..k {
                resp[i * k + c] = (loglik[i * k + c] - lse).exp();
            }
        }
        Ok(resp)
    }
}

// ── host helpers ──────────────────────────────────────────────────────────

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(index, 4, &value as *const u32 as *const std::ffi::c_void);
}

/// k-means++ centroid initialization (seeded, deterministic).
fn kmeanspp(data: &[f32], n: usize, d: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut means = vec![0.0f32; k * d];
    let mut dist2 = vec![f32::MAX; n];

    // First center: uniform random sample.
    let first = (rng.usize(0..n)) as usize;
    means[0..d].copy_from_slice(&data[first * d..(first + 1) * d]);
    for i in 0..n {
        dist2[i] = sq_dist(&data[i * d..i * d + d], &means[0..d], d);
    }

    for c in 1..k {
        // Pick next center with probability proportional to dist².
        let total: f32 = dist2.iter().sum();
        let mut target = rng.f32() * total;
        let mut pick = 0usize;
        for (i, &dd) in dist2.iter().enumerate() {
            target -= dd;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        means[c * d..(c + 1) * d].copy_from_slice(&data[pick * d..(pick + 1) * d]);
        for i in 0..n {
            let dd = sq_dist(&data[i * d..i * d + d], &means[c * d..(c + 1) * d], d);
            if dd < dist2[i] {
                dist2[i] = dd;
            }
        }
    }
    means
}

fn sq_dist(x: &[f32], y: &[f32], d: usize) -> f32 {
    let mut s = 0.0f32;
    for j in 0..d {
        let diff = x[j] - y[j];
        s += diff * diff;
    }
    s
}

/// Index of the nearest mean for sample `i`.
fn nearest(means: &[f32], data: &[f32], i: usize, n: usize, d: usize) -> usize {
    let _ = n;
    let mut best = 0usize;
    let mut best_d = sq_dist(&data[i * d..i * d + d], &means[0..d], d);
    for c in 1..means.len() / d {
        let dd = sq_dist(&data[i * d..i * d + d], &means[c * d..(c + 1) * d], d);
        if dd < best_d {
            best_d = dd;
            best = c;
        }
    }
    best
}

/// Lower-triangular Cholesky factor L (row-major d×d) of a symmetric matrix
/// `a` (row-major, only the lower triangle is read). `None` if not PD.
fn cholesky(a: &[f32], d: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for kk in 0..j {
                s -= l[i * d + kk] * l[j * d + kk];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

/// From the lower-triangular Cholesky factor L of Σ, compute the precision
/// Σ⁻¹ = L⁻ᵀ·L⁻¹ and log|Σ| = 2·Σ log L_ii.
fn precision_and_logdet(l: &[f32], d: usize) -> (Vec<f32>, f32) {
    // L⁻¹ (lower triangular), column-by-column forward substitution.
    let mut linv = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            if i == j {
                linv[i * d + j] = 1.0 / l[i * d + j];
            } else {
                let mut s = 0.0f32;
                for kk in j..i {
                    s += l[i * d + kk] * linv[kk * d + j];
                }
                linv[i * d + j] = -s / l[i * d + i];
            }
        }
    }
    // Σ⁻¹ = L⁻ᵀ L⁻¹ (symmetric).
    let mut prec = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = 0.0f32;
            for kk in 0..d {
                s += linv[kk * d + i] * linv[kk * d + j];
            }
            prec[i * d + j] = s;
            prec[j * d + i] = s;
        }
    }
    let mut logdet = 0.0f32;
    for i in 0..d {
        logdet += 2.0 * l[i * d + i].ln();
    }
    (prec, logdet)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 2-blob synthetic data (pure PRNG, no Metal dependency).
    fn blobs(seed: u64, n_per: usize, k: usize, d: usize, spread: f32) -> Vec<f32> {
        let n = n_per * k;
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut data = vec![0.0f32; n * d];
        let mut idx = 0usize;
        for c in 0..k {
            for _ in 0..n_per {
                for dim in 0..d {
                    let center = if dim == c { 8.0 } else { 0.0 };
                    data[idx * d + dim] = center + (rng.f32() - 0.5) * spread;
                }
                idx += 1;
            }
        }
        data
    }

    #[test]
    fn cholesky_roundtrip_recovers_covariance() {
        // Symmetric PD matrix [[4, 1], [1, 3]].
        let a = [4.0f32, 1.0, 1.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        let (prec, logdet) = precision_and_logdet(&l, 2);
        // L Lᵀ == a
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0f32;
                for kk in 0..2 {
                    s += l[i * 2 + kk] * l[j * 2 + kk];
                }
                assert!(
                    (s - a[i * 2 + j]).abs() < 1e-5,
                    "LL^T[{}][{}] = {}",
                    i,
                    j,
                    s
                );
            }
        }
        // prec * a == I
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0f32;
                for kk in 0..2 {
                    s += prec[i * 2 + kk] * a[kk * 2 + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-4, "P*A[{}][{}] = {}", i, j, s);
            }
        }
        // det = 4*3 - 1*1 = 11
        assert!((logdet.exp() - 11.0).abs() < 1e-4, "det = {}", logdet.exp());
    }

    #[test]
    fn gmm_recovers_two_blobs_and_separates() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => return, // no Metal device available
        };
        let n = 240;
        let d = 3;
        let data = blobs(7, 120, 2, d, 1.0);

        let mut gmm = GMM::new(GMMConfig {
            n_components: 2,
            max_iterations: 100,
            tolerance: 1e-4,
            seed: 42,
            reg_covar: 1e-4,
        });
        gmm.fit(&ctx, &data, n, d).unwrap();
        assert!(gmm.n_iter() > 0);

        // Means should cover the two blob centers (8,0,0) and (0,8,0), in
        // either component order (GMM component labeling is arbitrary).
        let m0 = &gmm.means()[0..d];
        let m1 = &gmm.means()[d..2 * d];
        let c0 = [8.0f32, 0.0, 0.0];
        let c1 = [0.0f32, 8.0, 0.0];
        let a = sq_dist(m0, &c0, d) + sq_dist(m1, &c1, d);
        let b = sq_dist(m0, &c1, d) + sq_dist(m1, &c0, d);
        assert!(
            a.min(b) < 1.0,
            "means {:?} / {:?} too far from blob centers",
            m0,
            m1
        );

        // Responsibilities assign each point to the correct blob (>90%),
        // allowing either component ordering (labeling is arbitrary).
        let resp = gmm.responsibilities();
        let mut correct_id = 0usize;
        let mut correct_sw = 0usize;
        for i in 0..n {
            let truth = if i < 120 { 0usize } else { 1usize };
            let pred = if resp[i * 2] >= resp[i * 2 + 1] { 0 } else { 1 };
            if pred == truth {
                correct_id += 1;
            }
            if (1 - pred) == truth {
                correct_sw += 1;
            }
        }
        let purity = correct_id.max(correct_sw) as f32 / n as f32;
        assert!(purity > 0.9, "purity {}", purity);

        // predict / predict_proba / score run on held-out data.
        let preds = gmm.predict(&ctx, &data, n, d).unwrap();
        let proba = gmm.predict_proba(&ctx, &data, n, d).unwrap();
        let sc = gmm.score(&ctx, &data, n, d).unwrap();
        assert!(preds.len() == n && proba.len() == n * 2);
        assert!(sc.is_finite(), "score not finite");
        // proba rows sum to 1
        for i in 0..n {
            let s = proba[i * 2] + proba[i * 2 + 1];
            assert!((s - 1.0).abs() < 1e-4, "proba row {} sums to {}", i, s);
        }
    }

    #[test]
    fn gmm_invalid_inputs_rejected() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut gmm = GMM::new(GMMConfig::default());
        assert!(gmm.fit(&ctx, &[], 0, 2).is_err());
        let data = vec![1.0f32; 6];
        assert!(gmm.fit(&ctx, &data, 3, 2).is_ok());
        // too many components
        let mut bad = GMM::new(GMMConfig {
            n_components: 10,
            ..Default::default()
        });
        assert!(bad.fit(&ctx, &data, 3, 2).is_err());
        // data length mismatch
        let mut gmm2 = GMM::new(GMMConfig::default());
        assert!(gmm2.fit(&ctx, &data, 3, 3).is_err());
    }
}
