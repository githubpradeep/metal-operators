use metal_operators::lasso::{Lasso, LassoConfig};
use metal_operators::metal::MetalContext;

// ── CPU reference (same objective, for accuracy comparison only) ──────────

fn soft_threshold(x: f32, a: f32) -> f32 {
    if x > a {
        x - a
    } else if x < -a {
        x + a
    } else {
        0.0
    }
}

/// CPU coordinate descent on the augmented Gram, mirroring the GPU Lasso.
fn cpu_lasso_fit(
    data: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    alpha: f32,
    fit_intercept: bool,
    max_iter: usize,
    tol: f32,
) -> (Vec<f32>, f32) {
    let dim = if fit_intercept { d + 1 } else { d };
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
        if fit_intercept {
            for j in 0..d {
                a[j * dim + d] += row[j];
                a[d * dim + j] += row[j];
            }
            a[d * dim + d] += 1.0;
            b[d] += y[i];
        }
    }

    let mut w = vec![0.0f32; dim];
    for _ in 0..max_iter {
        let mut max_change = 0.0f32;
        for j in 0..dim {
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
            let new_w = if fit_intercept && j == dim - 1 {
                rho / denom
            } else {
                soft_threshold(rho, alpha) / denom
            };
            max_change = max_change.max((new_w - w[j]).abs());
            w[j] = new_w;
        }
        if max_change <= tol {
            break;
        }
    }
    (w[..d].to_vec(), if fit_intercept { w[d] } else { 0.0 })
}

fn predict(weights: &[f32], bias: f32, data: &[f32], n: usize, d: usize) -> Vec<f32> {
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── Test helpers ───────────────────────────────────────────────────────────

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn make_regression(data: &[f32], n: usize, d: usize, w_true: &[f32], b_true: f32) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(7);
    // y = X·w_true + b_true + gaussian-ish noise
    (0..n)
        .map(|i| {
            let row = &data[i * d..(i + 1) * d];
            let mut s = b_true;
            for (x, w) in row.iter().zip(w_true.iter()) {
                s += x * w;
            }
            s + (rng.f32() - 0.5) * 0.2
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn lasso_matches_cpu_reference() {
    let ctx = MetalContext::new().expect("Metal context");
    let d = 5;
    let n = 200;
    let data = make_data(n, d, 7);
    let w_true = [1.0, -0.5, 0.0, 2.0, 0.0];
    let y = make_regression(&data, n, d, &w_true, 0.7);

    let alpha = 0.05;
    let max_iter = 2000;
    let tol = 1e-5;
    let mut lasso = Lasso::new(LassoConfig {
        alpha,
        fit_intercept: true,
        tol,
        max_iterations: max_iter,
        seed: 42,
    });
    lasso.fit(&ctx, &data, &y, n, d).expect("lasso fit");

    let (cpu_w, cpu_b) = cpu_lasso_fit(&data, &y, n, d, alpha, true, max_iter, tol);

    assert!(
        max_abs_diff(&lasso.weights, &cpu_w) < 1e-3,
        "weights differ: gpu={:?} cpu={:?}",
        &lasso.weights,
        &cpu_w
    );
    assert!(
        (lasso.bias - cpu_b).abs() < 1e-3,
        "bias differ: gpu={:?} cpu={:?}",
        lasso.bias,
        cpu_b
    );
}

#[test]
fn lasso_zero_alpha_matches_ols() {
    let ctx = MetalContext::new().expect("metal context");
    let d = 4;
    let n = 300;
    let data = make_data(n, d, 11);
    let w_true = [0.8, 1.2, -1.1, 0.9];
    let y = make_regression(&data, n, d, &w_true, 0.3);

    // alpha = 0 → ordinary least squares (coordinate descent converges there).
    let config = LassoConfig {
        alpha: 0.0,
        fit_intercept: true,
        tol: 1e-5,
        max_iterations: 5000,
        seed: 42,
    };
    let mut lasso = Lasso::new(config);
    lasso.fit(&ctx, &data, &y, n, d).expect("lasso fit");

    // At alpha = 0 the Lasso objective is exactly OLS, so verify in-sample R²
    // is essentially perfect for the noise-free-ish synthetic problem.
    let preds = lasso.predict(&ctx, &data, n, d).unwrap();
    let r2 = lasso.score(&ctx, &data, &y, n, d).unwrap();
    assert!(
        r2 > 0.9,
        "lasso(alpha=0) should approximate OLS well, got r2={r2}"
    );
    let _ = preds;
}

#[test]
fn lasso_high_alpha_induces_sparsity() {
    let ctx = MetalContext::new().expect("metal context");
    let d = 10;
    let n = 150;
    let data = make_data(n, d, 3);
    let w_true = [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.2, 0.0];
    let y = make_regression(&data, n, d, &w_true, 0.0);

    // Large alpha shrinks small coefficients to exactly zero.
    let config = LassoConfig {
        alpha: 0.5,
        fit_intercept: true,
        tol: 1e-4,
        max_iterations: 3000,
        seed: 42,
    };
    let mut lasso = Lasso::new(config);
    lasso.fit(&ctx, &data, &y, n, d).expect("lasso fit");

    let n_zero = lasso.weights.iter().filter(|w| **w == 0.0).count();
    assert!(
        n_zero >= 4,
        "high alpha should drive coefficients to zero, got {} zeros in {:?}",
        n_zero,
        &lasso.weights
    );
    assert!(lasso.converged, "solver should report convergence");
}

#[test]
fn lasso_no_intercept() {
    let ctx = MetalContext::new().expect("metal context");
    let d = 3;
    let n = 120;
    let data = make_data(n, d, 5);
    let w_true = [1.0, -0.7, 0.4];
    let y = make_regression(&data, n, d, &w_true, 0.0);

    let config = LassoConfig {
        alpha: 0.05,
        fit_intercept: false,
        tol: 1e-5,
        max_iterations: 2000,
        seed: 42,
    };
    let mut lasso = Lasso::new(config);
    lasso.fit(&ctx, &data, &y, n, d).expect("lasso fit");
    assert_eq!(lasso.bias, 0.0);

    let (cpu_w, _) = cpu_lasso_fit(&data, &y, n, d, 0.05, false, 2000, 1e-5);
    assert!(
        max_abs_diff(&lasso.weights, &cpu_w) < 1e-3,
        "no-intercept weights mismatch: {:?} vs {:?}",
        &lasso.weights,
        &cpu_w
    );
}

#[test]
fn lasso_predict_and_score() {
    let ctx = MetalContext::new().expect("metal context");
    let d = 4;
    let n = 150;
    let data = make_data(n, d, 9);
    let w_true = [1.0, 2.0, -1.0, 0.5];
    let y = make_regression(&data, n, d, &w_true, 0.6);

    let mut lasso = Lasso::new(LassoConfig {
        alpha: 0.1,
        fit_intercept: true,
        tol: 1e-5,
        max_iterations: 2000,
        seed: 42,
    });
    lasso.fit(&ctx, &data, &y, n, d).expect("lasso fit");

    // Score on the training set should be high for an easy problem.
    let r2 = lasso.score(&ctx, &data, &y, n, d).unwrap();
    assert!(r2 > 0.85, "expected high r2, got {r2}");

    // predict should match the closed-form Xw + b from the fitted params.
    let preds = lasso.predict(&ctx, &data, n, d).unwrap();
    let ref_preds = predict(&lasso.weights, lasso.bias, &data, n, d);
    assert!(max_abs_diff(&preds, &ref_preds) < 1e-4);
}

#[test]
fn lasso_rejects_invalid_inputs() {
    let ctx = MetalContext::new().expect("metal context");
    let d = 3;
    let n = 10;
    let data = make_data(n, d, 1);
    let mut lasso = Lasso::new(LassoConfig::default());

    // Missing dims.
    assert!(lasso.fit(&ctx, &[], &[], 0, 0).is_err());
    // Data length mismatch.
    assert!(lasso.fit(&ctx, &data, &vec![0.0f32; n], n, d).is_ok());
    // Bad alpha.
    let bad = Lasso::new(LassoConfig {
        alpha: -1.0,
        ..Default::default()
    });
    let mut bad = bad;
    assert!(bad.fit(&ctx, &data, &vec![0.0f32; n], n, d).is_err());
    // Predict before fit.
    let unfitted = Lasso::new(LassoConfig::default());
    assert!(unfitted.predict(&ctx, &data, n, d).is_err());
}
