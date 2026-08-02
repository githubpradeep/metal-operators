//! Benchmark for linear regression — Metal GPU vs CPU (normal equations) vs sklearn.
//!
//! Mirrors the methodology of logistic_regression_benchmark.rs:
//!   - GPU:  LinearRegression::fit (Metal Gram kernels + host solve)
//!   - CPU:  pure-Rust normal-equations reference
//!   - sklearn: via Python subprocess (sklearn.linear_model.LinearRegression)

use metal_operators::linear_regression::{LinearRegression, LinearRegressionConfig};
use metal_operators::metal::MetalContext;
use std::io::Write;
use std::time::Instant;

// ── CPU reference linear regression (normal equations) ─────────────────────

/// Solve `A x = b` (row-major `dim × dim`) with Gaussian elimination +
/// partial pivoting.
fn cpu_solve(a: &mut [f32], b: &mut [f32], dim: usize) -> Vec<f32> {
    for col in 0..dim {
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
        for r in (col + 1)..dim {
            let f = a[r * dim + col] / a[col * dim + col];
            for c in col..dim {
                a[r * dim + c] -= f * a[col * dim + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f32; dim];
    for r in (0..dim).rev() {
        let mut s = b[r];
        for c in (r + 1)..dim {
            s -= a[r * dim + c] * x[c];
        }
        x[r] = s / a[r * dim + r];
    }
    x
}

/// Pure-Rust OLS via the normal equations (independent CPU baseline).
fn cpu_ols_fit(data: &[f32], y: &[f32], n: usize, d: usize) -> (Vec<f32>, f32) {
    let dim = d + 1;
    let mut a = vec![0.0f32; dim * dim];
    let mut b = vec![0.0f32; dim];

    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for j in 0..d {
            for k in 0..d {
                a[j * dim + k] += row[j] * row[k];
            }
            b[j] += row[j] * y[i];
        }
        for j in 0..d {
            a[j * dim + d] += row[j];
            a[d * dim + j] += row[j];
        }
        a[d * dim + d] += 1.0;
        b[d] += y[i];
    }

    let x = cpu_solve(&mut a, &mut b, dim);
    (x[..d].to_vec(), x[d])
}

// ── sklearn reference via Python subprocess ────────────────────────────────

fn sklearn_ms(data: &[f32], y: &[f32], n: usize, d: usize, alpha: f32) -> f64 {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("sklearn_linear.py");

    let data_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n * d * 4) };
    let label_bytes = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u8, n * 4) };

    let mut inp = Vec::new();
    writeln!(inp, "{} {} {} {} {}", n, d, alpha, 1, n * d * 4).unwrap();
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
    let w_true: Vec<f32> = (0..d).map(|j| (j as f32 + 1.0) * 0.5).collect();
    let mut data = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let mut z = 1.25f32;
        for j in 0..d {
            let x = rng.f32() * 2.0 - 1.0;
            data.push(x);
            z += w_true[j] * x;
        }
        y.push(z + (rng.f32() - 0.5) * 0.2);
    }
    (data, y)
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
    let mut pw = LinearRegression::new(LinearRegressionConfig::default());
    pw.fit(&ctx, &warm.0, &warm.1, 200, 8).unwrap();
    let _ = cpu_ols_fit(&warm.0, &warm.1, 200, 8);

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  Linear Regression fit — Metal GPU vs CPU vs sklearn                      ║");
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
        let (data, y) = gen_data(n, d, 42);

        // GPU Metal
        let niter = if n * d > 500_000 { 3usize } else { 10 };
        let t0 = Instant::now();
        for _ in 0..niter {
            let mut model = LinearRegression::new(LinearRegressionConfig::default());
            model.fit(&ctx, &data, &y, n, d).unwrap();
        }
        let metal = t0.elapsed().as_secs_f64() * 1000.0 / niter as f64;

        // CPU
        let cpu = {
            let niter_c = if n * d > 100_000 { 3usize } else { 10 };
            let t0 = Instant::now();
            for _ in 0..niter_c {
                let _ = cpu_ols_fit(&data, &y, n, d);
            }
            t0.elapsed().as_secs_f64() * 1000.0 / niter_c as f64
        };

        // sklearn (skip for large data — pipe too large)
        let sk = if n * d * 4 > 200_000_000 {
            0.0
        } else {
            sklearn_ms(&data, &y, n, d, 0.0)
        };

        time_one(label, metal, cpu, sk);
    }

    println!();
    println!("  Notes:");
    println!("  - 'sklearn = 0.00' means skipped (data > 200 MB for pipe)");
    println!("  - Metal solve is closed-form normal equations: XᵀX built with");
    println!("    shared-memory row tiles (d<=128), Xᵀy/colsum chunked, then a");
    println!("    tiny fixed-order reduction + host (d+1)² Gaussian solve.");
    println!();
}
