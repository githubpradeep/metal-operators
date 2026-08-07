use crate::metal::MetalContext;
use metal::*;
use std::sync::OnceLock;

const SHADER_SRC: &str = include_str!("../../shaders/naive_bayes.metal");

// ── Pipeline cache ────────────────────────────────────────────────────────

struct PipelineCache {
    reduce_partials: OnceLock<ComputePipelineState>,
    reduce_sum: OnceLock<ComputePipelineState>,
    predict_logp: OnceLock<ComputePipelineState>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            reduce_partials: OnceLock::new(),
            reduce_sum: OnceLock::new(),
            predict_logp: OnceLock::new(),
        }
    }

    fn get(&self, ctx: &MetalContext, name: &str) -> anyhow::Result<&ComputePipelineState> {
        let slot: &OnceLock<ComputePipelineState> = match name {
            "gnb_reduce_partials" => &self.reduce_partials,
            "gnb_reduce_sum" => &self.reduce_sum,
            "gnb_predict_logp" => &self.predict_logp,
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

/// Configuration for Gaussian Naive Bayes.
#[derive(Clone)]
pub struct GaussianNBConfig {
    /// Fraction of the largest per-class feature variance added to every class
    /// variance for numerical stability (mirrors sklearn's `var_smoothing`).
    pub var_smoothing: f32,
}

impl Default for GaussianNBConfig {
    fn default() -> Self {
        Self {
            var_smoothing: 1e-9,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────

pub struct GaussianNB {
    config: GaussianNBConfig,
    /// Per-class feature means, row-major (k × d).
    means: Vec<f32>,
    /// Per-class feature variances, row-major (k × d).
    variances: Vec<f32>,
    /// Per-class priors (k,).
    priors: Vec<f32>,
    /// Per-class log priors (k,).
    log_priors: Vec<f32>,
    k: usize,
    d: usize,
    n: usize,
    pipelines: PipelineCache,
}

impl GaussianNB {
    pub fn new(config: GaussianNBConfig) -> Self {
        Self {
            config,
            means: Vec::new(),
            variances: Vec::new(),
            priors: Vec::new(),
            log_priors: Vec::new(),
            k: 0,
            d: 0,
            n: 0,
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
    pub fn means(&self) -> &[f32] {
        &self.means
    }
    pub fn variances(&self) -> &[f32] {
        &self.variances
    }
    pub fn priors(&self) -> &[f32] {
        &self.priors
    }
    pub fn class_prior(&self) -> &[f32] {
        &self.priors
    }

    /// Fit Gaussian Naive Bayes.
    ///
    /// GPU reduction pass computes the per-class feature sum and sum-of-squares
    /// (two kernels, one command buffer -> one CPU-GPU sync), then the host
    /// derives per-class means, variances, and empirical priors. This exactly
    /// mirrors sklearn's `GaussianNB.fit` statistics.
    ///
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `data` - Flat row-major data array (n × d)
    /// * `y` - Class labels (n,) with integer values in `[0, n_classes)`
    /// * `n` - Number of samples
    /// * `d` - Number of features
    /// * `n_classes` - Number of classes
    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        n_classes: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(n > 0 && d > 0, "Data must be non-empty");
        anyhow::ensure!(n_classes >= 2, "n_classes must be at least 2");
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
        anyhow::ensure!(
            self.config.var_smoothing >= 0.0 && self.config.var_smoothing.is_finite(),
            "var_smoothing must be a finite, non-negative value"
        );

        // Threadgroup shared budget: (2*d + 1)*k floats (32 KB shared memory).
        let shared_floats = (2 * d + 1) * n_classes;
        let shared_bytes = shared_floats * std::mem::size_of::<f32>();
        anyhow::ensure!(
            shared_bytes <= 32 * 1024,
            "n_classes*d too large ({}): shared memory for the reduction would exceed 32 KB",
            shared_bytes
        );

        // Convert f32 labels (class ids) to uint32 for the Metal kernel.
        let mut ids = vec![0u32; n];
        for (i, &lab) in y.iter().enumerate() {
            let l = lab as i64;
            anyhow::ensure!(
                lab >= 0.0 && l < n_classes as i64 && lab.fract() == 0.0,
                "label {} at index {} is not an integer class id in [0, {})",
                lab,
                i,
                n_classes
            );
            ids[i] = l as u32;
        }

        self.k = n_classes;
        self.d = d;
        self.n = n;

        let reduce_partials = self.pipelines.get(ctx, "gnb_reduce_partials")?;
        let reduce_sum = self.pipelines.get(ctx, "gnb_reduce_sum")?;

        const TG: u64 = 256;
        const GNB_PTILE: u32 = 128;
        let n_groups = (n as u64 + GNB_PTILE as u64 - 1) / GNB_PTILE as u64;
        let kd = (n_classes * d) as u64;

        // ── GPU buffers ──
        let x_buf = ctx.new_buffer(data);
        let y_buf = ctx.new_buffer(&ids);
        let sum_p_buf = ctx.new_buffer_uninitialized((n_groups * kd * 4) as u64);
        let sumsq_p_buf = ctx.new_buffer_uninitialized((n_groups * kd * 4) as u64);
        let count_p_buf = ctx.new_buffer_uninitialized((n_groups * n_classes as u64 * 4) as u64);
        let sum_buf = ctx.new_buffer_uninitialized((kd * 4) as u64);
        let sumsq_buf = ctx.new_buffer_uninitialized((kd * 4) as u64);
        let cnt_buf = ctx.new_buffer_uninitialized((n_classes as u64 * 4) as u64);

        // ── Pass: reduction (partials -> combine), one command buffer ──
        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Kernel A: per-threadgroup partials.
        enc.set_compute_pipeline_state(&reduce_partials);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&y_buf), 0);
        enc.set_buffer(2, Some(&sum_p_buf), 0);
        enc.set_buffer(3, Some(&sumsq_p_buf), 0);
        enc.set_buffer(4, Some(&count_p_buf), 0);
        set_u32(&enc, 5, n as u32);
        set_u32(&enc, 6, n_classes as u32);
        set_u32(&enc, 7, d as u32);
        enc.set_threadgroup_memory_length(0, shared_bytes as u64);
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

        // Kernel B: deterministic combine of partials.
        enc.set_compute_pipeline_state(&reduce_sum);
        enc.set_buffer(0, Some(&sum_p_buf), 0);
        enc.set_buffer(1, Some(&sumsq_p_buf), 0);
        enc.set_buffer(2, Some(&count_p_buf), 0);
        enc.set_buffer(3, Some(&sum_buf), 0);
        enc.set_buffer(4, Some(&sumsq_buf), 0);
        enc.set_buffer(5, Some(&cnt_buf), 0);
        set_u32(&enc, 6, n_groups as u32);
        set_u32(&enc, 7, n_classes as u32);
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

        let sum = ctx.read_buffer::<f32>(&sum_buf, kd as usize);
        let sumsq = ctx.read_buffer::<f32>(&sumsq_buf, kd as usize);
        let cnt = ctx.read_buffer::<f32>(&cnt_buf, n_classes);

        // ── Host: per-class mean, variance, priors ──
        let mut means = vec![0.0f32; kd as usize];
        let mut variances = vec![0.0f32; kd as usize];
        let mut priors = vec![0.0f32; n_classes];
        // Variance floor follows sklearn: add var_smoothing * largest variance
        // to every class-feature variance.
        let mut max_var = 0.0f32;
        for c in 0..n_classes {
            let count = cnt[c];
            anyhow::ensure!(
                count > 0.0,
                "class {} has no samples (count = {})",
                c,
                count
            );
            priors[c] = count / n as f32;
            for j in 0..d {
                let mean = sum[c * d + j] / count;
                let var = (sumsq[c * d + j] / count) - mean * mean;
                means[c * d + j] = mean;
                variances[c * d + j] = var;
                if var > max_var {
                    max_var = var;
                }
            }
        }
        let floor = self.config.var_smoothing * max_var;
        for v in variances.iter_mut() {
            if *v < floor {
                *v = floor;
            }
        }
        let log_priors = priors.iter().map(|p| p.ln()).collect::<Vec<f32>>();

        self.means = means;
        self.variances = variances;
        self.priors = priors;
        self.log_priors = log_priors;

        Ok(())
    }

    /// Raw per-sample log posteriors (n × k, row-major) from a single GPU
    /// predict launch. The host subtracts the log-sum-exp for the normalized
    /// log-probabilities (sklearn semantics).
    fn predict_logp(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {} got {}",
            n * d,
            data.len()
        );
        anyhow::ensure!(self.k > 0, "Model is not fitted");
        anyhow::ensure!(
            d == self.d,
            "Feature count mismatch: expected {} got {}",
            self.d,
            d
        );

        let predict = self.pipelines.get(ctx, "gnb_predict_logp")?;
        const TG: u64 = 256;
        let n_tg = (n as u64 + TG - 1) / TG;

        let x_buf = ctx.new_buffer(data);
        let mean_buf = ctx.new_buffer(&self.means);
        let var_buf = ctx.new_buffer(&self.variances);
        let logp_buf = ctx.new_buffer(&self.log_priors);
        let out_buf = ctx.new_buffer_uninitialized((n * self.k * 4) as u64);

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&predict);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&mean_buf), 0);
        enc.set_buffer(2, Some(&var_buf), 0);
        enc.set_buffer(3, Some(&logp_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        set_u32(&enc, 5, n as u32);
        set_u32(&enc, 6, self.k as u32);
        set_u32(&enc, 7, d as u32);
        enc.dispatch_thread_groups(
            MTLSize {
                width: n_tg,
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

        Ok(ctx.read_buffer::<f32>(&out_buf, n * self.k))
    }

    /// Predict class labels (argmax over per-class log posteriors).
    pub fn predict(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let logp_un = self.predict_logp(ctx, data, n, d)?;
        let mut preds = vec![0.0f32; n];
        for i in 0..n {
            let mut best = 0usize;
            let mut best_v = logp_un[i * self.k];
            for c in 1..self.k {
                let v = logp_un[i * self.k + c];
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            preds[i] = best as f32;
        }
        Ok(preds)
    }

    /// Normalized log probabilities (n × k, row-major), matching sklearn.
    pub fn predict_log_proba(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let logp_un = self.predict_logp(ctx, data, n, d)?;
        let mut out = vec![0.0f32; n * self.k];
        for i in 0..n {
            // log-sum-exp over classes (per sample).
            let row = &logp_un[i * self.k..(i + 1) * self.k];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = maxv + row.iter().map(|v| (*v - maxv).exp()).sum::<f32>().ln();
            for c in 0..self.k {
                out[i * self.k + c] = logp_un[i * self.k + c] - lse;
            }
        }
        Ok(out)
    }

    /// Class probabilities (n × k, row-major), matching sklearn.
    pub fn predict_proba(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let logp = self.predict_log_proba(ctx, data, n, d)?;
        Ok(logp.iter().map(|v| v.exp()).collect())
    }

    /// Mean accuracy over the given labelled data.
    pub fn score(
        &self,
        ctx: &MetalContext,
        data: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(y.len() == n, "Labels length mismatch");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference (host) Gaussian Naive Bayes statistics + posterior, used to
    /// validate the GPU reduction and predict kernels.
    fn ref_stats(data: &[f32], y: &[f32], n: usize, d: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
        let mut sum = vec![0.0f32; k * d];
        let mut sumsq = vec![0.0f32; k * d];
        let mut cnt = vec![0.0f32; k];
        for i in 0..n {
            let c = y[i] as usize;
            cnt[c] += 1.0;
            for j in 0..d {
                sum[c * d + j] += data[i * d + j];
                sumsq[c * d + j] += data[i * d + j] * data[i * d + j];
            }
        }
        let mut mean = vec![0.0f32; k * d];
        let mut var = vec![0.0f32; k * d];
        for c in 0..k {
            for j in 0..d {
                mean[c * d + j] = sum[c * d + j] / cnt[c];
                var[c * d + j] = sumsq[c * d + j] / cnt[c] - mean[c * d + j] * mean[c * d + j];
            }
        }
        (mean, var)
    }

    #[test]
    fn gnb_reduction_matches_reference() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => return, // no Metal device available
        };
        let n = 24;
        let d = 3;
        let k = 2;
        let data: Vec<f32> = (0..n * d)
            .map(|i| {
                // class 0 centred near -1, class 1 near +1 per feature
                let sample = i / d;
                let feat = i % d;
                let base = if sample < 12 { -1.0 } else { 1.0 };
                base + (i as f32 * 0.37 % 0.5)
            })
            .collect();
        let y: Vec<f32> = (0..n).map(|i| if i < 12 { 0.0 } else { 1.0 }).collect();

        let mut gnb = GaussianNB::new(GaussianNBConfig {
            var_smoothing: 1e-9,
        });
        gnb.fit(&ctx, &data, &y, n, d, k).unwrap();

        let (em, ev) = ref_stats(&data, &y, n, d, k);
        for j in 0..k * d {
            assert!(
                (gnb.means()[j] - em[j]).abs() < 1e-4,
                "mean[{}] {} != {}",
                j,
                gnb.means()[j],
                em[j]
            );
            assert!(
                (gnb.variances()[j] - ev[j]).abs() < 1e-4,
                "var[{}] {} != {}",
                j,
                gnb.variances()[j],
                ev[j]
            );
        }
    }

    #[test]
    fn gnb_predict_classifies_and_normalizes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let n = 20;
        let d = 2;
        let k = 2;
        let data: Vec<f32> = (0..n * d)
            .map(|i| {
                let s = i / d;
                let f = i % d;
                let base = if s < 10 { -2.0 } else { 2.0 };
                base + (f as f32 * 0.1)
            })
            .collect();
        let y: Vec<f32> = (0..n).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();

        let mut gnb = GaussianNB::new(GaussianNBConfig::default());
        gnb.fit(&ctx, &data, &y, n, d, k).unwrap();

        // Test on the training data (separable) -> perfect accuracy.
        let acc = gnb.score(&ctx, &data, &y, n, d).unwrap();
        assert_eq!(acc, 1.0, "separable data should classify perfectly");

        let proba = gnb.predict_proba(&ctx, &data, n, d).unwrap();
        for i in 0..n {
            let row: f32 = (0..k).map(|c| proba[i * k + c]).sum();
            assert!((row - 1.0).abs() < 1e-4, "proba row {} sums to {}", i, row);
            assert!(proba[i * k + 0..i * k + k].iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn gnb_multigroup_reduction_it_three_classes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // n > GNB_PTILE (128) forces the multi-threadgroup partial-combine path.
        let n = 400;
        let d = 4;
        let k = 3;
        let mut data = Vec::with_capacity(n * d);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let c = i % k;
            y.push(c as f32);
            for j in 0..d {
                let center = (c as f32) * 1.7;
                data.push(center + ((i as f32 * 0.21 + j as f32) % 0.3));
            }
        }

        let mut gnb = GaussianNB::new(GaussianNBConfig::default());
        gnb.fit(&ctx, &data, &y, n, d, k).unwrap();
        let (em, ev) = ref_stats(&data, &y, n, d, k);
        for j in 0..k * d {
            assert!((gnb.means()[j] - em[j]).abs() < 1e-4, "mean[{}]", j);
            assert!((gnb.variances()[j] - ev[j]).abs() < 1e-4, "var[{}]", j);
        }
        let acc = gnb.score(&ctx, &data, &y, n, d).unwrap();
        assert_eq!(
            acc, 1.0,
            "three separable classes should classify perfectly"
        );
    }
}
