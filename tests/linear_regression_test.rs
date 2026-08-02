use metal_operators::linear_regression::{LinearRegression, LinearRegressionConfig};
use metal_operators::metal::MetalContext;

// ── CPU reference (for accuracy comparison only) ───────────────────────────

/// Solve `A x = b` (row-major `dim × dim`) with Gaussian elimination +
/// partial pivoting on the CPU.
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

/// CPU ordinary least squares via the normal equations (with optional
/// intercept), matching the GPU formulation.
fn cpu_ols_fit(
    data: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    alpha: f32,
    fit_intercept: bool,
) -> (Vec<f32>, f32) {
    let dim = if fit_intercept { d + 1 } else { d };
    let mut a = vec![0.0f32; dim * dim];
    let mut b = vec![0.0f32; dim];

    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        // XᵀX / Xᵀy
        for j in 0..d {
            for k in 0..d {
                a[j * dim + k] += row[j] * row[k];
            }
            b[j] += row[j] * y[i];
        }
        if fit_intercept {
            for j in 0..d {
                a[j * dim + d] += row[j];
                a[d * dim + j] += row[j];
            }
            a[d * dim + d] += 1.0;
            b[d] += y[i];
        }
    }
    for j in 0..d {
        a[j * dim + j] += alpha;
    }

    let x = cpu_solve(&mut a, &mut b, dim);
    (x[..d].to_vec(), if fit_intercept { x[d] } else { 0.0 })
}

fn cpu_predict(weights: &[f32], bias: f32, data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let row = &data[i * d..(i + 1) * d];
            bias + row
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
        })
        .collect()
}

fn cpu_r2(preds: &[f32], y: &[f32]) -> f32 {
    let n = y.len();
    let y_mean = y.iter().sum::<f32>() / n as f32;
    let ss_res: f32 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum();
    let ss_tot: f32 = y.iter().map(|t| (t - y_mean) * (t - y_mean)).sum();
    1.0 - ss_res / ss_tot
}

// ── Test helpers ────────────────────────────────────────────────────────────

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// y = X·w_true + b_true + noise·ε
fn make_regression(
    data: &[f32],
    n: usize,
    d: usize,
    w_true: &[f32],
    b_true: f32,
    noise: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n)
        .map(|i| {
            let row = &data[i * d..(i + 1) * d];
            b_true
                + row
                    .iter()
                    .zip(w_true.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                + noise * (rng.f32() * 2.0 - 1.0)
        })
        .collect()
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|x| x * x).sum();
    (num / (den + 1e-12)).sqrt()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_linear_basic_fit() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (300, 4);
    let data = make_data(n, d, 3);
    let w_true = vec![1.0f32, -2.0, 0.5, 3.0];
    let y = make_regression(&data, n, d, &w_true, 0.7, 0.05, 4);

    let mut lr = LinearRegression::new(LinearRegressionConfig::default());
    lr.fit(&ctx, &data, &y, n, d).expect("fit");

    assert_eq!(lr.weights().len(), d, "weights length mismatch");
    assert_eq!(lr.n_features(), d);
    assert!(lr.weights().iter().all(|w| w.is_finite()), "weights finite");
    assert!(lr.bias().is_finite(), "bias finite");
    assert!(lr.final_loss.is_finite(), "mse finite");
}

#[test]
fn test_linear_predict_before_fit_fails() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (n, d) = (10, 2);
    let data = make_data(n, d, 1);
    let y = make_data(n, 1, 2);

    let lr = LinearRegression::new(LinearRegressionConfig::default());
    assert!(
        lr.predict(&ctx, &data, n, d).is_err(),
        "predict should fail before fit"
    );
    assert!(
        lr.score(&ctx, &data, &y, n, d).is_err(),
        "score should fail before fit"
    );
}

#[test]
fn test_linear_recovers_known_coefficients() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Low noise: recovered coefficients should be very close to ground truth.
    let (n, d) = (2000, 6);
    let data = make_data(n, d, 11);
    let w_true = vec![0.5f32, -1.5, 2.0, 0.25, -3.0, 1.0];
    let b_true = 1.25f32;
    let y = make_regression(&data, n, d, &w_true, b_true, 0.01, 12);

    let mut lr = LinearRegression::new(LinearRegressionConfig::default());
    lr.fit(&ctx, &data, &y, n, d).expect("fit");

    let werr = rel_err(&lr.weights(), &w_true);
    assert!(
        werr < 1e-2,
        "weights too far from ground truth: rel_err={}",
        werr
    );
    assert!(
        (lr.bias() - b_true).abs() < 0.05,
        "bias too far from ground truth: {} vs {}",
        lr.bias(),
        b_true
    );
}

#[test]
fn test_linear_accuracy_matches_cpu() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (800, 12);
    let data = make_data(n, d, 42);
    let y = make_data(n, 1, 43);

    // GPU
    let mut gpu = LinearRegression::new(LinearRegressionConfig::default());
    gpu.fit(&ctx, &data, &y, n, d).expect("GPU fit");
    let gpu_preds = gpu.predict(&ctx, &data, n, d).expect("GPU predict");

    // CPU
    let (cpu_w, cpu_b) = cpu_ols_fit(&data, &y, n, d, 0.0, true);
    let cpu_preds = cpu_predict(&cpu_w, cpu_b, &data, n, d);

    let gpu_r2 = cpu_r2(&gpu_preds, &y);
    let cpu_r2 = cpu_r2(&cpu_preds, &y);
    let diff = (gpu_r2 - cpu_r2).abs();
    assert!(
        diff < 1e-4,
        "R² gap between GPU ({}) and CPU ({}) too large: {}",
        gpu_r2,
        cpu_r2,
        diff
    );
}

#[test]
fn test_linear_score_r2() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Noise-free data: R² ≈ 1.
    let (n, d) = (500, 4);
    let data = make_data(n, d, 21);
    let w_true = vec![1.0f32, 2.0, -1.0, 0.5];
    let y = make_regression(&data, n, d, &w_true, 0.0, 0.0, 22);

    let mut lr = LinearRegression::new(LinearRegressionConfig::default());
    lr.fit(&ctx, &data, &y, n, d).expect("fit");
    let r2 = lr.score(&ctx, &data, &y, n, d).expect("score");
    assert!(
        (r2 - 1.0).abs() < 1e-3,
        "expected R² ≈ 1 on noise-free data, got {}",
        r2
    );
}

#[test]
fn test_linear_ridge_shrinks_coefficients() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (300, 6);
    let data = make_data(n, d, 31);
    let w_true = vec![5.0f32, -5.0, 5.0, -5.0, 5.0, -5.0];
    let y = make_regression(&data, n, d, &w_true, 0.0, 0.5, 32);

    // OLS
    let mut ols = LinearRegression::new(LinearRegressionConfig::default());
    ols.fit(&ctx, &data, &y, n, d).expect("ols fit");
    let norm_ols: f32 = ols.weights().iter().map(|w| w * w).sum();

    // Ridge with strong alpha
    let mut ridge = LinearRegression::new(LinearRegressionConfig {
        alpha: 50.0,
        ..Default::default()
    });
    ridge.fit(&ctx, &data, &y, n, d).expect("ridge fit");
    let norm_ridge: f32 = ridge.weights().iter().map(|w| w * w).sum();

    assert!(
        norm_ridge < norm_ols,
        "ridge should shrink coefficients: {} vs {}",
        norm_ridge,
        norm_ols
    );
}

#[test]
fn test_linear_no_intercept() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (400, 3);
    let data = make_data(n, d, 51);
    let w_true = vec![2.0f32, -1.0, 0.5];
    let y = make_regression(&data, n, d, &w_true, 0.0, 0.05, 52);

    let mut lr = LinearRegression::new(LinearRegressionConfig {
        fit_intercept: false,
        ..Default::default()
    });
    lr.fit(&ctx, &data, &y, n, d).expect("fit");

    assert_eq!(lr.bias(), 0.0, "bias should be zero without intercept");
    let werr = rel_err(&lr.weights(), &w_true);
    assert!(
        werr < 1e-2,
        "weights too far from ground truth without intercept: rel_err={}",
        werr
    );
}

#[test]
fn test_linear_deterministic() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (200, 8);
    let data = make_data(n, d, 61);
    let y = make_data(n, 1, 62);

    let config = LinearRegressionConfig::default();

    let mut m1 = LinearRegression::new(config.clone());
    m1.fit(&ctx, &data, &y, n, d).expect("fit 1");
    let w1 = m1.weights().to_vec();
    let b1 = m1.bias();

    let mut m2 = LinearRegression::new(config.clone());
    m2.fit(&ctx, &data, &y, n, d).expect("fit 2");
    let w2 = m2.weights().to_vec();
    let b2 = m2.bias();

    assert_eq!(b1, b2, "bias differs between runs");
    for j in 0..d {
        assert!(
            (w1[j] - w2[j]).abs() < 1e-6,
            "weight[{}] differs: {} vs {}",
            j,
            w1[j],
            w2[j]
        );
    }

    let p1 = m1.predict(&ctx, &data, n, d).expect("predict 1");
    let p2 = m2.predict(&ctx, &data, n, d).expect("predict 2");
    assert_eq!(p1, p2, "predictions differ between runs");
}

#[test]
fn test_linear_kernel_variants() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Sweep shapes across the kernel dispatch ranges (n > d so the system
    // is well-posed):
    //  - d=2   -> tiny Gram (1 threadgroup)
    //  - d=16  -> one threadgroup of the Xᵀy element grid
    //  - d=64  -> multi-threadgroup Gram
    //  - d=128 -> large Gram grid
    //  - d=256 -> element grid spans 2 threadgroups (2d+2 = 514)
    let cases: &[(usize, usize)] = &[
        (400, 2),
        (400, 16),
        (400, 64),
        (300, 128),
        (600, 256),
        (300, 3),
        (400, 8),
    ];

    for &(n, d) in cases {
        let data = make_data(n, d, 100);
        let w_true: Vec<f32> = (0..d).map(|j| (j as f32 + 1.0) * 0.25).collect();
        let y = make_regression(&data, n, d, &w_true, 1.0, 0.1, 101);

        let mut lr = LinearRegression::new(LinearRegressionConfig::default());
        lr.fit(&ctx, &data, &y, n, d)
            .unwrap_or_else(|e| panic!("fit failed for n={} d={}: {}", n, d, e));

        assert_eq!(lr.weights().len(), d, "wrong number of weights");
        for &w in lr.weights() {
            assert!(w.is_finite(), "non-finite weight for n={} d={}", n, d);
        }

        let preds = lr
            .predict(&ctx, &data, n, d)
            .unwrap_or_else(|e| panic!("predict failed for n={} d={}: {}", n, d, e));
        assert_eq!(preds.len(), n, "wrong number of predictions");
        for &p in &preds {
            assert!(p.is_finite(), "non-finite prediction for n={} d={}", n, d);
        }

        let r2 = lr
            .score(&ctx, &data, &y, n, d)
            .unwrap_or_else(|e| panic!("score failed for n={} d={}: {}", n, d, e));
        assert!(
            r2 >= 0.8,
            "low R² {} for low-noise data at n={} d={}",
            r2,
            n,
            d
        );
    }
}

#[test]
fn test_linear_singular_data() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Duplicate columns -> XᵀX singular; the tiny ridge must keep the
    // solve stable (no panic, finite weights).
    let (n, d) = (100, 4);
    let data = make_data(n, d, 71);
    // Make columns 2 and 3 identical to column 0.
    let mut degenerate = data.clone();
    for i in 0..n {
        degenerate[i * d + 2] = degenerate[i * d];
        degenerate[i * d + 3] = degenerate[i * d];
    }
    let y = make_regression(&degenerate, n, d, &[1.0, 1.0, 0.0, 0.0], 0.0, 0.1, 72);

    let mut lr = LinearRegression::new(LinearRegressionConfig::default());
    lr.fit(&ctx, &degenerate, &y, n, d)
        .expect("fit on singular data");
    assert!(
        lr.weights().iter().all(|w| w.is_finite()),
        "weights must stay finite on degenerate data"
    );
    assert!(lr.bias().is_finite(), "bias must stay finite");
}
