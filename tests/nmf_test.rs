//! NMF integration test: verifies the Metal-backed operator (generic
//! `nmf_mm` matmul kernels + `nmf_update`) against an independent CPU
//! reference implementation of the Lee–Seung multiplicative updates on a
//! low-rank non-negative matrix, and exercises the NMF `transform`.

use metal_operators::metal::MetalContext;
use metal_operators::nmf::{NMFConfig, NMF};

/// Independent CPU reference NMF (multiplicative updates, f64).
fn reference_nmf(
    v: &[f32],
    n: usize,
    d: usize,
    k: usize,
    iters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut w = vec![0.0f64; n * k];
    let mut h = vec![0.0f64; k * d];
    for v in w.iter_mut() {
        *v = rng.f32() as f64;
    }
    for v in h.iter_mut() {
        *v = rng.f32() as f64;
    }
    const EPS: f64 = 1e-10;
    for _ in 0..iters {
        // H update: (WᵀV)/(WᵀWH)
        let mut wtv = vec![0.0f64; k * d];
        let mut wtw = vec![0.0f64; k * k];
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += w[i * k + a] * w[i * k + b];
                }
                wtw[a * k + b] = acc;
            }
            for j in 0..d {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += w[i * k + a] * v[i * d + j] as f64;
                }
                wtv[a * d + j] = acc;
            }
        }
        let mut wtw_h = vec![0.0f64; k * d];
        for a in 0..k {
            for j in 0..d {
                let mut acc = 0.0;
                for b in 0..k {
                    acc += wtw[a * k + b] * h[b * d + j];
                }
                wtw_h[a * d + j] = acc;
            }
        }
        for a in 0..k {
            for j in 0..d {
                h[a * d + j] *= wtv[a * d + j] / (wtw_h[a * d + j] + EPS);
            }
        }

        // W update: (V·Hᵀ)/(W·H·Hᵀ)
        let mut vht = vec![0.0f64; n * k];
        let mut hht = vec![0.0f64; k * k];
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0;
                for j in 0..d {
                    acc += h[a * d + j] * h[b * d + j];
                }
                hht[a * k + b] = acc;
            }
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..d {
                    acc += v[i * d + j] as f64 * h[a * d + j];
                }
                vht[i * k + a] = acc;
            }
        }
        let mut w_hht = vec![0.0f64; n * k];
        for i in 0..n {
            for a in 0..k {
                let mut acc = 0.0;
                for b in 0..k {
                    acc += w[i * k + b] * hht[b * k + a];
                }
                w_hht[i * k + a] = acc;
            }
        }
        for i in 0..n {
            for a in 0..k {
                w[i * k + a] *= vht[i * k + a] / (w_hht[i * k + a] + EPS);
            }
        }
    }
    let wf: Vec<f32> = w.iter().map(|&x| x as f32).collect();
    let hf: Vec<f32> = h.iter().map(|&x| x as f32).collect();
    (wf, hf)
}

fn frob_err(v: &[f32], w: &[f32], h: &[f32], n: usize, d: usize, k: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n {
        for j in 0..d {
            let mut s = 0.0f64;
            for c in 0..k {
                s += w[i * k + c] as f64 * h[c * d + j] as f64;
            }
            let e = v[i * d + j] as f64 - s;
            acc += e * e;
        }
    }
    acc.sqrt()
}

fn frob(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn synth_lowrank(n: usize, d: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();
    let h: Vec<f32> = (0..k * d).map(|_| rng.f32()).collect();
    let mut v = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let mut s = 0.0f32;
            for c in 0..k {
                s += w[i * k + c] * h[c * d + j];
            }
            v[i * d + j] = s + rng.f32() * 0.001;
        }
    }
    v
}

#[test]
fn test_reconstruction_against_reference() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (n, d, k) = (120usize, 25usize, 4usize);
    let v = synth_lowrank(n, d, k, 11);

    let mut nmf = NMF::new(NMFConfig {
        n_components: k,
        max_iterations: 400,
        tolerance: 1e-9,
        seed: 42,
        eps: 1e-10,
    });
    nmf.fit(&ctx, &v, n, d).expect("NMF fit failed");

    let vnorm = frob(&v).max(1e-12);

    // Independent CPU reference matching the GPU's update schedule.
    let (w_ref, h_ref) = reference_nmf(&v, n, d, k, 400, 42);
    let cpu_err = frob_err(&v, &w_ref, &h_ref, n, d, k);

    // GPU and CPU multiplicative updates must arrive at the same solution.
    assert!(
        (cpu_err - nmf.reconstruction_error() as f64).abs() / vnorm < 5e-3,
        "GPU NMF diverged from CPU reference ({:.4} vs {:.4})",
        nmf.reconstruction_error(),
        cpu_err
    );

    // The factorization must actually reconstruct V well (multiplicative
    // updates converge slowly, so allow a modest relative residual).
    let rel = nmf.reconstruction_error() as f64 / vnorm;
    assert!(
        rel < 0.04,
        "GPU NMF reconstruction relative error too large: {:.4}",
        rel
    );
}

#[test]
fn test_transform_returns_latent_embedding() {
    let ctx = MetalContext::new().expect("No Metal device");
    let (n, d, k) = (90usize, 20usize, 3usize);
    let v = synth_lowrank(n, d, k, 5);
    let n_new = 30;

    let mut nmf = NMF::new(NMFConfig {
        n_components: k,
        max_iterations: 100,
        tolerance: 0.0,
        seed: 1,
        eps: 1e-10,
    });
    nmf.fit(&ctx, &v, n, d).expect("fit failed");

    let held = &v[..n_new * d];
    let w_new = nmf
        .transform(&ctx, held, n_new, d)
        .expect("transform failed");
    assert_eq!(w_new.len(), n_new * k);
    assert!(
        w_new.iter().all(|&x| x >= 0.0),
        "transform output must be non-negative"
    );
}

#[test]
fn test_invalid_inputs_rejected() {
    let ctx = MetalContext::new().expect("No Metal device");
    let mut nmf = NMF::new(NMFConfig::default());

    // Length mismatch.
    assert!(nmf.fit(&ctx, &[1.0f32; 10], 3, 2).is_err());
    // Negative values rejected.
    assert!(nmf.fit(&ctx, &[-1.0f32, 2.0, 3.0, 4.0], 2, 2).is_err());
    // transform before fit.
    assert!(nmf.transform(&ctx, &[1.0f32; 4], 2, 2).is_err());
}
