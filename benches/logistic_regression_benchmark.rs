//! Benchmark for logistic regression — Metal GPU vs CPU (BLAS/Accelerate) vs sklearn.
//!
//! Mirrors the methodology of kmeans_benchmark.rs / pca_benchmark.rs:
//!   - GPU:  LogisticRegression::fit (Metal fused fwd/bwd kernels)
//!   - CPU:  pure-Rust mini-batch SGD reference
//!   - sklearn: via Python subprocess (sklearn.linear_model.LogisticRegression)

use metal_operators::logistic_regression::{LogisticRegression, LogisticRegressionConfig};
use metal_operators::metal::MetalContext;
use std::io::Write;
use std::time::Instant;

// ── BLAS via Accelerate (macOS) ────────────────────────────────────────────
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn sgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
}

// ── CPU reference logistic regression ──────────────────────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Pure-Rust mini-batch SGD logistic regression (independent CPU baseline).
fn cpu_logreg_fit(
    data: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    config: &LogisticRegressionConfig,
) -> (Vec<f32>, f32, f32) {
    let batch_size = config.batch_size.min(n);
    let n_batches = (n + batch_size - 1) / batch_size;
    let inv_batch = 1.0 / batch_size as f32;
    let reg = 1.0 / (config.c * n as f32).max(1e-30);

    let mut weights = vec![0.0f32; d];
    let mut bias = 0.0f32;
    let mut vel_w = vec![0.0f32; d];
    let mut vel_b = 0.0f32;

    let mut rng = fastrand::Rng::with_seed(config.seed);
    let mut epoch_loss = 0.0f32;

    for epoch in 0..config.max_epochs {
        // shuffle
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }

        let mut loss_sum = 0.0f32;
        let mut loss_batches = 0u32;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n);
            let bsize = end - start;
            if bsize == 0 {
                break;
            }

            let mut grad_w = vec![0.0f32; d];
            let mut grad_b = 0.0f32;

            for &idx in &indices[start..end] {
                let xi = &data[idx * d..(idx + 1) * d];
                let mut logit = bias;
                for j in 0..d {
                    logit += xi[j] * weights[j];
                }
                let p = sigmoid(logit);
                let err = p - y[idx];
                for j in 0..d {
                    grad_w[j] += err * xi[j];
                }
                grad_b += err;
            }

            for j in 0..d {
                grad_w[j] = grad_w[j] * inv_batch + reg * weights[j];
            }
            grad_b *= inv_batch;

            for j in 0..d {
                vel_w[j] = config.momentum * vel_w[j] - config.learning_rate * grad_w[j];
                weights[j] += vel_w[j];
            }
            vel_b = config.momentum * vel_b - config.learning_rate * grad_b;
            bias += vel_b;

            // loss
            let mut bl = 0.0f32;
            for &idx in &indices[start..end] {
                let xi = &data[idx * d..(idx + 1) * d];
                let mut logit = bias;
                for j in 0..d {
                    logit += xi[j] * weights[j];
                }
                let p = sigmoid(logit);
                let eps = 1e-15f32;
                let yi = y[idx];
                bl += -yi * (p + eps).ln() - (1.0 - yi) * (1.0 - p + eps).ln();
            }
            let l2 = weights.iter().map(|w| w * w).sum::<f32>() * 0.5 * reg;
            loss_sum += bl * inv_batch + l2;
            loss_batches += 1;
        }

        if loss_batches > 0 {
            epoch_loss = loss_sum / loss_batches as f32;
        }
        if epoch % 10 == 0 || epoch == config.max_epochs - 1 {
            eprintln!("  [cpu] epoch {:3}: loss={:.6}", epoch, epoch_loss);
        }
    }

    (weights, bias, epoch_loss)
}

// ── sklearn reference via Python subprocess ────────────────────────────────

fn sklearn_ms(
    data: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    config: &LogisticRegressionConfig,
) -> f64 {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("sklearn_logreg.py");

    let data_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n * d * 4) };
    let label_bytes = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u8, n * 4) };

    let mut inp = Vec::new();
    writeln!(
        inp,
        "{} {} {} {} {} {} {} {} {}",
        n,
        d,
        config.c,
        config.learning_rate,
        config.momentum,
        config.max_epochs,
        config.batch_size,
        config.tol,
        n * d * 4
    )
    .unwrap();
    inp.extend_from_slice(data_bytes);
    inp.extend_from_slice(label_bytes);

    let mut ch = match std::process::Command::new("python3")
        .arg(script.to_str().unwrap())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return 0.0,
    };

    if let Err(_) = ch.stdin.take().unwrap().write_all(&inp) {
        return 0.0;
    }
    let o = match ch.wait_with_output() {
        Ok(o) => o,
        Err(_) => return 0.0,
    };

    if !o.status.success() || o.stdout.len() < 4 {
        return 0.0;
    }

    let mut r = [0u8; 4];
    r.copy_from_slice(&o.stdout[..4]);
    f32::from_le_bytes(r) as f64
}

// ── data generation ────────────────────────────────────────────────────────

fn gen_data(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    // Create separable data: two Gaussian blobs
    let mut data = Vec::with_capacity(n * d);
    let mut labels = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..n {
        let label = if i < half { 0.0 } else { 1.0 };
        labels.push(label);
        let center = if label == 0.0 { -1.5 } else { 1.5 };
        for _ in 0..d {
            data.push(center + (rng.f32() - 0.5) * 2.0);
        }
    }
    (data, labels)
}

fn time_one(label: &str, metal_ms: f64, cpu_ms: f64, sk_ms: f64) {
    let vs_cpu = if cpu_ms > 0.0 { metal_ms / cpu_ms } else { 0.0 };
    let vs_skl = if sk_ms > 0.0 { metal_ms / sk_ms } else { 0.0 };
    let speedup = format!(
        "{:.1}x/CPU {:.1}x/skl",
        1.0 / vs_cpu.max(0.001),
        1.0 / vs_skl.max(0.001)
    );
    println!(
        "  {:<30} {:>8.2} {:>8.2} {:>8.2}  {}",
        label, metal_ms, cpu_ms, sk_ms, speedup
    );
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Warm up: compile pipelines
    let warm = gen_data(200, 8, 0);
    let mut pw = LogisticRegression::new(LogisticRegressionConfig::default());
    pw.fit(&ctx, &warm.0, &warm.1, 200, 8).unwrap();
    cpu_logreg_fit(
        &warm.0,
        &warm.1,
        200,
        8,
        &LogisticRegressionConfig::default(),
    );

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  Logistic Regression fit — Metal GPU vs CPU vs sklearn                ║");
    println!("  ╚══════════════════════════════════════════════════════════════════════════╝");
    println!(
        "  {:<30} {:>8} {:>8} {:>8}  {}",
        "shape", "metal", "cpu", "sklearn", "speedup"
    );

    let shapes: &[(&str, usize, usize)] = &[
        ("N=10K  D=8", 10_000, 8),
        ("N=10K  D=32", 10_000, 32),
        ("N=10K  D=128", 10_000, 128),
        ("N=100K D=8", 100_000, 8),
        ("N=100K D=32", 100_000, 32),
        ("N=100K D=128", 100_000, 128),
        ("N=1M   D=8", 1_000_000, 8),
        ("N=1M   D=32", 1_000_000, 32),
    ];

    for &(label, n, d) in shapes {
        let (data, labels) = gen_data(n, d, 42);
        let config = LogisticRegressionConfig::default();

        // GPU Metal
        let niter = if n * d > 500_000 { 3usize } else { 10 };
        let t0 = Instant::now();
        for _ in 0..niter {
            let mut model = LogisticRegression::new(config.clone());
            model.fit(&ctx, &data, &labels, n, d).unwrap();
        }
        let metal = t0.elapsed().as_secs_f64() * 1000.0 / niter as f64;

        // CPU
        let cpu = {
            let niter_c = if n * d > 100_000 { 3usize } else { 10 };
            let t0 = Instant::now();
            for _ in 0..niter_c {
                let _ = cpu_logreg_fit(&data, &labels, n, d, &config);
            }
            t0.elapsed().as_secs_f64() * 1000.0 / niter_c as f64
        };

        // sklearn (skip for large data — pipe too large)
        let sk = if n * d * 4 > 200_000_000 {
            0.0
        } else {
            sklearn_ms(&data, &labels, n, d, &config)
        };

        time_one(label, metal, cpu, sk);
    }

    println!();
    println!("  Notes:");
    println!("  - 'sklearn = 0.00' means skipped (data > 200 MB for pipe)");
    println!();
}
