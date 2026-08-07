//! t-SNE integration test: verifies the Metal-backed operator (exact O(N²)
//! `tsne_distances` / `tsne_perplexity` / `tsne_grad` kernels) against an
//! independent CPU reference that runs the *same* deterministic algorithm
//! (same seeded Gaussian init, same perplexity bisection, same velocity
//! update), plus structural checks: separation of well-separated blobs,
//! seed determinism, and input validation.

use metal_operators::metal::MetalContext;
use metal_operators::tsne::{TSNEConfig, TSNE};

/// Generate `k * n_per` samples in `d` dimensions as `k` well-separated
/// Gaussian blobs (pure deterministic PRNG so GPU and CPU runs see the same
/// data). Blob `c` is centred at `10.0` on axis `c` and ~0 on the others.
fn blobs(seed: u64, n_per: usize, k: usize, d: usize, spread: f32) -> (Vec<f32>, Vec<usize>) {
    let n = n_per * k;
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut data = vec![0.0f32; n * d];
    let mut labels = Vec::with_capacity(n);
    let mut idx = 0usize;
    for c in 0..k {
        for _ in 0..n_per {
            for dim in 0..d {
                let center = if dim == c { 10.0 } else { 0.0 };
                data[idx * d + dim] = center + (rng.f32() - 0.5) * spread;
            }
            labels.push(c);
            idx += 1;
        }
    }
    (data, labels)
}

// ── CPU reference (mirrors shaders/tsne.metal + src/tsne/mod.rs) ──

fn ref_distances(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..d {
                let diff = x[i * d + k] - x[j * d + k];
                s += diff * diff;
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn ref_perplexity(d: &[f32], n: usize, perplexity: f32) -> Vec<f32> {
    const ITERS: usize = 50;
    const LO: f32 = -15.0;
    const HI: f32 = 15.0;
    let target = perplexity.ln();
    let mut p = vec![0.0f32; n * n];
    for i in 0..n {
        let (mut lo, mut hi) = (LO, HI);
        for _ in 0..ITERS {
            let mid = 0.5 * (lo + hi);
            let sigma = mid.exp();
            let inv2s2 = 1.0 / (2.0 * sigma * sigma);
            let mut s = 0.0f32;
            for j in 0..n {
                if j == i {
                    continue;
                }
                s += (-d[i * n + j] * inv2s2).exp();
            }
            if s <= 0.0 {
                lo = mid;
                continue;
            }
            let mut h = 0.0f32;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let pj = (-d[i * n + j] * inv2s2).exp() / s;
                h -= pj * pj.ln();
            }
            if h > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let sigma = (0.5 * (lo + hi)).exp();
        let inv2s2 = 1.0 / (2.0 * sigma * sigma);
        let mut s = 0.0f32;
        for j in 0..n {
            if j == i {
                continue;
            }
            s += (-d[i * n + j] * inv2s2).exp();
        }
        let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
        for j in 0..n {
            p[i * n + j] = if j == i {
                0.0
            } else {
                (-d[i * n + j] * inv2s2).exp() * inv_s
            };
        }
    }
    p
}

fn ref_symmetrize(p: &[f32], n: usize) -> Vec<f32> {
    let inv2n = 1.0 / (2.0 * n as f32);
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            out[i * n + j] = (p[i * n + j] + p[j * n + i]) * inv2n;
        }
    }
    out
}

fn ref_gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-38);
    let u2 = rng.f32();
    let mag = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    mag * theta.cos()
}

#[allow(clippy::too_many_arguments)]
fn ref_tsne(
    data: &[f32],
    n: usize,
    d: usize,
    nc: usize,
    perplexity: f32,
    lr: f32,
    n_iter: usize,
    exag: f32,
    exag_iters: usize,
    momentum: f32,
    seed: u64,
) -> Vec<f32> {
    let dmat = ref_distances(data, n, d);
    let p = ref_symmetrize(&ref_perplexity(&dmat, n, perplexity), n);

    let mut rng = fastrand::Rng::with_seed(seed);
    let mut y = vec![0.0f32; n * nc];
    for v in y.iter_mut() {
        *v = 1e-4 * ref_gaussian(&mut rng);
    }
    let mut vel = vec![0.0f32; n * nc];

    for t in 0..n_iter {
        let e = if t < exag_iters { exag } else { 1.0 };
        let mut a1 = vec![0.0f32; n * nc];
        let mut a2 = vec![0.0f32; n * nc];
        let mut z = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut d2 = 0.0f32;
                for c in 0..nc {
                    let dc = y[i * nc + c] - y[j * nc + c];
                    d2 += dc * dc;
                }
                let q = 1.0 / (1.0 + d2);
                let pq = p[i * n + j] * e * q;
                let q2 = q * q;
                for c in 0..nc {
                    let dc = y[i * nc + c] - y[j * nc + c];
                    a1[i * nc + c] += pq * dc;
                    a2[i * nc + c] += q2 * dc;
                }
                z += q;
            }
        }
        let inv_z = 1.0 / z;
        for i in 0..n {
            for c in 0..nc {
                let idx = i * nc + c;
                let g = 4.0 * (a1[idx] - a2[idx] * inv_z);
                vel[idx] = momentum * vel[idx] - lr * g;
                y[idx] += vel[idx];
            }
        }
    }
    y
}

// ── Tests ─────────────────────────────────────────────────────────

#[test]
fn test_smoke_fit() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (data, _labels) = blobs(1, 40, 3, 4, 0.5);
    let n = 120;
    let d = 4;
    let mut tsne = TSNE::new(TSNEConfig {
        n_components: 2,
        perplexity: 15.0,
        learning_rate: 100.0,
        n_iter: 60,
        early_exaggeration: 12.0,
        exaggeration_iter: 20,
        momentum: 0.8,
        seed: 42,
        min_grad_norm: 0.0, // no early stop: exercise the full iteration count
    });
    tsne.fit(&ctx, &data, n, d).expect("fit failed");
    assert_eq!(tsne.embedding().len(), n * 2);
    assert_eq!(tsne.n_iter(), 60, "expected full iteration budget");
    assert!(tsne.kl_divergence().is_finite(), "KL must be finite");
    assert!(tsne.kl_divergence() > 0.0);
    // Embedding must have spread out (not all points collapsed at the origin).
    let y = tsne.embedding();
    let mean0: f32 = y.iter().step_by(2).map(|v| v * v).sum::<f32>() / n as f32;
    let mean1: f32 = y.iter().skip(1).step_by(2).map(|v| v * v).sum::<f32>() / n as f32;
    assert!(mean0 > 1e-6, "dim-0 spread too small: {mean0}");
    assert!(mean1 > 1e-6, "dim-1 spread too small: {mean1}");
}

#[test]
fn test_matches_cpu_reference() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (data, _labels) = blobs(3, 15, 3, 3, 0.6);
    let n = 45;
    let d = 3;
    // Runs with the same seed and 1..3 steps must track the CPU reference
    // closely. The first gradient step is the strongest, most direct check of
    // the P/affinity + gradient kernels: it must agree essentially exactly.
    // Later steps are chaotic (GPU FMA vs strict CPU f32 rounding compounds),
    // so tolerances widen — but a genuinely wrong kernel would blow past even
    // the loosest bound.
    for (iters, tol) in [(1usize, 2e-3f32), (2, 0.2), (3, 3.0)] {
        let mut tsne = TSNE::new(TSNEConfig {
            n_components: 2,
            perplexity: 8.0,
            learning_rate: 120.0,
            n_iter: iters,
            early_exaggeration: 12.0,
            exaggeration_iter: 10,
            momentum: 0.8,
            seed: 7,
            min_grad_norm: 0.0,
        });
        tsne.fit(&ctx, &data, n, d).expect("fit failed");
        let expected = ref_tsne(&data, n, d, 2, 8.0, 120.0, iters, 12.0, 10, 0.8, 7);
        let got = tsne.embedding();
        let mut max_diff = 0.0f32;
        for (a, b) in got.iter().zip(expected.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < tol,
            "iters {iters}: GPU embedding diverged from CPU reference: max diff {max_diff} (tol {tol})"
        );
    }
}

#[test]
fn test_deterministic_seed() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (data, _labels) = blobs(9, 20, 2, 3, 0.5);
    let n = 40;
    let d = 3;
    let config = TSNEConfig {
        n_components: 2,
        perplexity: 10.0,
        learning_rate: 100.0,
        n_iter: 25,
        early_exaggeration: 12.0,
        exaggeration_iter: 8,
        momentum: 0.8,
        seed: 1234,
        min_grad_norm: 0.0,
    };
    let mut a = TSNE::new(config.clone());
    let mut b = TSNE::new(config.clone());
    a.fit(&ctx, &data, n, d).expect("fit a failed");
    b.fit(&ctx, &data, n, d).expect("fit b failed");
    assert_eq!(
        a.embedding(),
        b.embedding(),
        "same seed must reproduce embedding"
    );

    let mut c = TSNE::new(TSNEConfig {
        seed: 9999,
        ..config.clone()
    });
    c.fit(&ctx, &data, n, d).expect("fit c failed");
    let differs = a
        .embedding()
        .iter()
        .zip(c.embedding().iter())
        .any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(differs, "different seeds should give different embeddings");
}

#[test]
fn test_separates_blobs() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (data, labels) = blobs(5, 40, 3, 8, 0.4);
    let n = 120;
    let d = 8;
    let mut tsne = TSNE::new(TSNEConfig {
        n_components: 2,
        perplexity: 15.0,
        learning_rate: 200.0,
        n_iter: 300,
        early_exaggeration: 12.0,
        exaggeration_iter: 100,
        momentum: 0.8,
        seed: 42,
        min_grad_norm: 0.0,
    });
    tsne.fit(&ctx, &data, n, d).expect("fit failed");

    // Nearest-neighbor purity in the embedding: most points' nearest neighbor
    // must belong to the same blob (t-SNE preserves local structure).
    let y = tsne.embedding();
    let mut same = 0usize;
    for i in 0..n {
        let mut nn = usize::MAX;
        let mut bd = f32::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = y[2 * i] - y[2 * j];
            let dy = y[2 * i + 1] - y[2 * j + 1];
            let dd = dx * dx + dy * dy;
            if dd < bd {
                bd = dd;
                nn = j;
            }
        }
        if labels[i] == labels[nn] {
            same += 1;
        }
    }
    let purity = same as f64 / n as f64;
    assert!(
        purity > 0.9,
        "embedding must separate the 3 blobs (purity {purity:.3})"
    );
}

#[test]
fn test_invalid_inputs_rejected() {
    let ctx = MetalContext::new().expect("No Metal device");
    let base = TSNEConfig::default();

    // Data length mismatch.
    let mut t1 = TSNE::new(base.clone());
    assert!(t1.fit(&ctx, &[1.0f32; 10], 3, 2).is_err());

    // perplexity must be < n.
    let mut t2 = TSNE::new(TSNEConfig {
        perplexity: 50.0,
        ..base.clone()
    });
    assert!(t2.fit(&ctx, &[1.0f32; 20], 10, 2).is_err());

    // n_components out of range.
    let mut t3 = TSNE::new(TSNEConfig {
        n_components: 16,
        ..base.clone()
    });
    assert!(t3.fit(&ctx, &[1.0f32; 20], 10, 2).is_err());

    // n < 2.
    let mut t4 = TSNE::new(base.clone());
    assert!(t4.fit(&ctx, &[1.0f32; 2], 1, 2).is_err());

    // learning_rate <= 0.
    let mut t5 = TSNE::new(TSNEConfig {
        learning_rate: 0.0,
        ..base
    });
    assert!(t5.fit(&ctx, &[1.0f32; 20], 10, 2).is_err());
}
