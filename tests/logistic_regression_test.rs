use metal_operators::logistic_regression::{LogisticRegression, LogisticRegressionConfig};
use metal_operators::metal::MetalContext;

// ── CPU reference (for accuracy comparison only) ───────────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cpu_predict_proba(weights: &[f32], bias: f32, data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut probs = Vec::with_capacity(n);
    for i in 0..n {
        let mut logit = bias;
        for j in 0..d {
            logit += data[i * d + j] * weights[j];
        }
        probs.push(sigmoid(logit));
    }
    probs
}

fn cpu_predict(weights: &[f32], bias: f32, data: &[f32], n: usize, d: usize) -> Vec<f32> {
    cpu_predict_proba(weights, bias, data, n, d)
        .into_iter()
        .map(|p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect()
}

fn cpu_score(weights: &[f32], bias: f32, data: &[f32], y: &[f32], n: usize, d: usize) -> f32 {
    let pred = cpu_predict(weights, bias, data, n, d);
    let correct: usize = pred
        .iter()
        .zip(y.iter())
        .filter(|(p, t)| (*p - *t).abs() < 0.01)
        .count();
    correct as f32 / n as f32
}

/// CPU mini-batch SGD with momentum (independent baseline for accuracy comparison).
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
    let mut final_loss = 0.0f32;

    for _epoch in 0..config.max_epochs {
        // Fisher-Yates shuffle
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

            // Batch loss (for tracking only)
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
            final_loss = loss_sum / loss_batches as f32;
        }
    }

    (weights, bias, final_loss)
}

// ── Test helpers ────────────────────────────────────────────────────────────

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// Generate linearly separable labels: sign of (bias + sum of feature_i * (i+1))
fn make_labels_linear(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut val = 0.3;
            for j in 0..d {
                val += data[i * d + j] * (j as f32 + 1.0);
            }
            if val > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_lr_rejects_invalid() {
    let ctx = MetalContext::new().expect("No Metal device");
    let mut lr = LogisticRegression::new(LogisticRegressionConfig::default());

    // Zero samples
    assert!(lr.fit(&ctx, &[], &[], 0, 4).is_err());
    // Zero features
    assert!(lr.fit(&ctx, &[1.0, 2.0], &[0.0, 1.0], 2, 0).is_err());
    // Data length mismatch
    assert!(lr.fit(&ctx, &[1.0, 2.0, 3.0], &[0.0, 1.0], 2, 1).is_err());
    // Labels length mismatch
    assert!(lr.fit(&ctx, &[1.0, 2.0], &[0.0], 2, 1).is_err());
    // Non-binary labels (multiclass) must be rejected, not silently mis-trained
    assert!(
        lr.fit(&ctx, &[1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0], 3, 1)
            .is_err(),
        "multiclass labels should be rejected"
    );
    assert!(
        lr.fit(&ctx, &[1.0, 2.0], &[-1.0, 1.0], 2, 1).is_err(),
        "negative labels should be rejected"
    );
}

#[test]
fn test_lr_predict_before_fit_errors() {
    let ctx = MetalContext::new().expect("No Metal device");
    let lr = LogisticRegression::new(LogisticRegressionConfig::default());

    let data = make_data(10, 2, 0);
    assert!(
        lr.predict_proba(&ctx, &data, 10, 2).is_err(),
        "predict_proba should fail before fit"
    );
    assert!(
        lr.predict(&ctx, &data, 10, 2).is_err(),
        "predict should fail before fit"
    );
}

#[test]
fn test_lr_predict_proba_range() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (200, 4);
    let data = make_data(n, d, 5);
    let y = make_labels_linear(&data, n, d);

    let mut lr = LogisticRegression::new(LogisticRegressionConfig {
        max_epochs: 30,
        batch_size: 64,
        ..Default::default()
    });
    lr.fit(&ctx, &data, &y, n, d).expect("fit");
    let probs = lr.predict_proba(&ctx, &data, n, d).expect("predict_proba");

    for &p in &probs {
        assert!(p >= 0.0 && p <= 1.0, "probability out of range: {}", p);
    }
}

#[test]
fn test_lr_convergence() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (500, 8);
    let data = make_data(n, d, 10);
    let y = make_labels_linear(&data, n, d);

    let mut lr = LogisticRegression::new(LogisticRegressionConfig {
        max_epochs: 10,
        batch_size: 64,
        learning_rate: 0.02,
        ..Default::default()
    });
    lr.fit(&ctx, &data, &y, n, d).expect("fit");

    // Check that final_loss is finite and reasonable
    assert!(
        lr.final_loss.is_finite(),
        "final loss is not finite: {}",
        lr.final_loss
    );
    assert!(
        lr.final_loss < 0.7,
        "expected final loss < 0.7, got {}",
        lr.final_loss
    );
}

#[test]
fn test_lr_weight_bias_accessors() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (50, 3);
    let data = make_data(n, d, 9);
    let y = make_labels_linear(&data, n, d);

    let mut lr = LogisticRegression::new(LogisticRegressionConfig {
        max_epochs: 10,
        batch_size: 32,
        ..Default::default()
    });
    lr.fit(&ctx, &data, &y, n, d).expect("fit");

    let w = lr.weights();
    let b = lr.bias();

    assert_eq!(w.len(), d, "weights length mismatch");
    assert!(
        w.iter().any(|x| x.abs() > 0.0),
        "some weights should be non-zero after fit"
    );
    assert!(b.is_finite(), "bias should be finite");
}

#[test]
fn test_lr_high_accuracy_on_separable_data() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Generate well-separated binary classes
    let n = 300;
    let d = 4;
    let mut data = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        if i < n / 2 {
            // Class 0: all features near -2
            for _ in 0..d {
                data.push(-2.0 + (i as f32 % 5.0) * 0.02);
            }
            y.push(0.0);
        } else {
            // Class 1: all features near +2
            for _ in 0..d {
                data.push(2.0 - (i as f32 % 5.0) * 0.02);
            }
            y.push(1.0);
        }
    }

    let mut lr = LogisticRegression::new(LogisticRegressionConfig {
        max_epochs: 50,
        batch_size: 32,
        learning_rate: 0.05,
        tol: 1e-6,
        ..Default::default()
    });
    lr.fit(&ctx, &data, &y, n, d).expect("fit");

    let acc = lr.score(&ctx, &data, &y, n, d).expect("score");
    assert!(
        acc > 0.95,
        "accuracy should be high on well-separated data: {}",
        acc
    );
}

#[test]
fn test_lr_accuracy_matches_cpu() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (500, 16);
    let data = make_data(n, d, 42);
    let y = make_labels_linear(&data, n, d);

    let config = LogisticRegressionConfig {
        max_epochs: 40,
        batch_size: 64,
        learning_rate: 0.02,
        seed: 1,
        ..Default::default()
    };

    // GPU
    let mut gpu = LogisticRegression::new(config.clone());
    gpu.fit(&ctx, &data, &y, n, d).expect("GPU fit");
    let gpu_acc = gpu.score(&ctx, &data, &y, n, d).expect("GPU score");

    // CPU
    let (cpu_w, cpu_b, _) = cpu_logreg_fit(&data, &y, n, d, &config);
    let cpu_acc = cpu_score(&cpu_w, cpu_b, &data, &y, n, d);

    let diff = (gpu_acc - cpu_acc).abs();
    assert!(
        diff < 0.15,
        "accuracy gap between GPU ({}) and CPU ({}) too large: {}",
        gpu_acc,
        cpu_acc,
        diff
    );
}

#[test]
fn test_lr_deterministic() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (n, d) = (100, 4);
    let data = make_data(n, d, 1);
    let y = make_labels_linear(&data, n, d);

    let config = LogisticRegressionConfig {
        max_epochs: 20,
        batch_size: 32,
        seed: 42,
        ..Default::default()
    };

    // First fit
    let mut m1 = LogisticRegression::new(config.clone());
    m1.fit(&ctx, &data, &y, n, d).expect("fit 1");
    let w1 = m1.weights().to_vec();
    let b1 = m1.bias();
    let loss1 = m1.final_loss;

    // Second fit (same seed)
    let mut m2 = LogisticRegression::new(config.clone());
    m2.fit(&ctx, &data, &y, n, d).expect("fit 2");
    let w2 = m2.weights().to_vec();
    let b2 = m2.bias();
    let loss2 = m2.final_loss;

    // Weights, bias, and loss should be identical given same seed
    assert_eq!(loss1, loss2, "final_loss differs between runs");
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

    // Predictions should also be identical
    let p1 = m1.predict(&ctx, &data, n, d).expect("predict 1");
    let p2 = m2.predict(&ctx, &data, n, d).expect("predict 2");
    assert_eq!(p1, p2, "predictions differ between runs");
}

#[test]
fn test_lr_kernel_variants() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Test with d=8 (naive kernel branch: d >= 8 && d % 8 == 0 but shared_f32 not <= 32768?)
    // d=8: shared_f32 = 8*8 + 8 + 16 = 88 bytes → fits, so simdgroup
    // We just verify all dimensions work without error

    let cases: &[(usize, usize)] = &[
        (200, 2),   // naive
        (200, 16),  // simdgroup (d >= 8 && d % 8 == 0, shared fits)
        (200, 8),   // simdgroup
        (200, 256), // splitd (d > 128)
        (100, 48),  // simdgroup (d >= 8 && d % 8 == 0)
        (100, 3),   // naive (d < 8)
    ];

    for &(n, d) in cases {
        let data = make_data(n, d, 100);
        let y = make_labels_linear(&data, n, d);

        let mut lr = LogisticRegression::new(LogisticRegressionConfig {
            max_epochs: 10,
            batch_size: 32,
            seed: 1,
            ..Default::default()
        });
        lr.fit(&ctx, &data, &y, n, d)
            .unwrap_or_else(|e| panic!("fit failed for n={} d={}: {}", n, d, e));

        // Predict should succeed and produce finite probabilities
        let probs = lr
            .predict_proba(&ctx, &data, n, d)
            .unwrap_or_else(|e| panic!("predict_proba failed for n={} d={}: {}", n, d, e));
        assert_eq!(probs.len(), n, "wrong number of probabilities");
        for &p in &probs {
            assert!(
                p.is_finite() && p >= 0.0 && p <= 1.0,
                "invalid probability {} for n={} d={}",
                p,
                n,
                d
            );
        }

        let preds = lr
            .predict(&ctx, &data, n, d)
            .unwrap_or_else(|e| panic!("predict failed for n={} d={}: {}", n, d, e));
        assert_eq!(preds.len(), n, "wrong number of predictions");
        for &pr in &preds {
            assert!(
                pr == 0.0 || pr == 1.0,
                "invalid prediction {} for n={} d={}",
                pr,
                n,
                d
            );
        }

        let acc = lr
            .score(&ctx, &data, &y, n, d)
            .unwrap_or_else(|e| panic!("score failed for n={} d={}: {}", n, d, e));
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "invalid score {} for n={} d={}",
            acc,
            n,
            d
        );
    }
}
