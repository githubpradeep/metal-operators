use fastrand::Rng;
use metal_operators::lda::{LDAConfig, LDA};
use metal_operators::metal::MetalContext;

fn get_context() -> MetalContext {
    MetalContext::new()
        .expect("Failed to create MetalContext - is this running on a Mac with Metal?")
}

/// Standard-normal-ish samples via the Box-Muller transform.
fn gauss(rng: &mut Rng) -> f32 {
    let u1 = rng.f32().max(1e-9);
    let u2 = rng.f32();
    ((-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()) as f32
}

/// Two Gaussian classes separated along axis 0 by `sep`.
fn two_classes(n0: usize, n1: usize, d: usize, sep: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::with_seed(seed);
    let mut x = Vec::with_capacity((n0 + n1) * d);
    let mut y = Vec::with_capacity(n0 + n1);
    for i in 0..(n0 + n1) {
        let c = if i < n0 { 0.0 } else { 1.0 };
        for j in 0..d {
            let off = if j == 0 && c > 0.5 { sep } else { 0.0 };
            x.push((gauss(&mut rng) + off) as f32);
        }
        y.push(c);
    }
    (x, y)
}

#[test]
fn lda_fit_smoke_test() {
    let ctx = get_context();
    let (n0, n1, d, k) = (80, 120, 6, 2);
    let (data, labels) = two_classes(n0, n1, d, 4.0, 42);

    let mut lda = LDA::new(LDAConfig { n_components: k });
    lda.fit(&ctx, &data, &labels, n0 + n1, d).unwrap();

    // k clamped to C-1 = 1 for only 2 classes.
    assert_eq!(lda.n_components(), 1);
    assert_eq!(lda.scalings().len(), 1 * d);
    assert_eq!(lda.eigenvalues().len(), 1);
    assert_eq!(lda.mean().len(), d);
    assert_eq!(lda.classes().len(), 2);
    assert_eq!(lda.class_counts().len(), 2);
    assert_eq!(lda.class_means().len(), 2 * d);

    // Transform shape (n, k).
    let proj = lda.transform(&ctx, &data, n0 + n1, d).unwrap();
    assert_eq!(proj.len(), (n0 + n1) * 1);

    // Well-separated two-class data → score should be near perfect.
    let score = lda.score(&ctx, &data, &labels, n0 + n1, d).unwrap();
    assert!(
        score >= 0.95,
        "expected ~perfect separation, got accuracy {score}"
    );

    let preds = lda.predict(&ctx, &data, n0 + n1, d).unwrap();
    assert_eq!(preds.len(), n0 + n1);
}

#[test]
fn lda_top_scaling_aligns_with_class_separation() {
    let ctx = get_context();
    let (n0, n1, d) = (150, 150, 6);
    // Strong separation only along axis 2 so the dominant direction is clear.
    let mut rng = Rng::with_seed(7);
    let mut data = Vec::with_capacity((n0 + n1) * d);
    let mut labels = Vec::with_capacity(n0 + n1);
    for i in 0..(n0 + n1) {
        let c = if i < n0 { 0.0 } else { 1.0 };
        for j in 0..d {
            let off = if j == 2 && c > 0.5 { 6.0 } else { 0.0 };
            data.push((gauss(&mut rng) + off) as f32);
        }
        labels.push(c);
    }
    let n = n0 + n1;

    let mut lda = LDA::new(LDAConfig { n_components: 1 });
    lda.fit(&ctx, &data, &labels, n, d).unwrap();

    // Expected direction = mean1 - mean0 (≈ axis 2).
    let mut ref_dir = vec![0.0f32; d];
    for j in 0..d {
        let m0 = data.iter().take(n0).skip(j).step_by(d).sum::<f32>() / n0 as f32;
        let m1 = data.iter().skip(n0 * d).skip(j).step_by(d).sum::<f32>() / n1 as f32;
        ref_dir[j] = m1 - m0;
    }
    let rn: f32 = ref_dir.iter().map(|v| v * v).sum::<f32>().sqrt();
    let s = lda.scalings();
    let mut sn = 0.0f32;
    for j in 0..d {
        sn += s[j] * s[j];
    }
    sn = sn.sqrt();
    let cos: f32 = (0..d).map(|j| (s[j] / sn) * (ref_dir[j] / rn)).sum();
    assert!(
        cos.abs() > 0.99,
        "top scaling should align with mean difference, cos={cos}"
    );
}

#[test]
fn lda_rank_clamped_to_classes_minus_one() {
    let ctx = get_context();
    let (n, d) = (200, 5);
    // Three well-separated classes → max informative rank = 2.
    let mut rng = Rng::with_seed(9);
    let mut data = Vec::with_capacity(n * d);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % 3;
        labels.push(c as f32);
        for j in 0..d {
            let off = if c == 1 && j == 0 {
                5.0
            } else if c == 2 && j == 1 {
                5.0
            } else {
                0.0
            };
            data.push((gauss(&mut rng) + off) as f32);
        }
    }

    for nc in [5usize, 10] {
        let mut lda = LDA::new(LDAConfig { n_components: nc });
        lda.fit(&ctx, &data, &labels, n, d).unwrap();
        assert_eq!(lda.n_components(), 2, "n_components must clamp to C-1=2");
        assert_eq!(lda.scalings().len(), 2 * d);
    }
}

#[test]
fn lda_transform_matches_cpu() {
    let ctx = get_context();
    let (n0, n1, d) = (60, 70, 3);
    let (data, labels) = two_classes(n0, n1, d, 2.5, 123);
    let n = n0 + n1;

    let mut lda = LDA::new(LDAConfig { n_components: 1 });
    lda.fit(&ctx, &data, &labels, n, d).unwrap();

    let k = lda.n_components();
    let proj = lda.transform(&ctx, &data, n, d).unwrap();
    let mean = lda.mean();
    let scal = lda.scalings();
    for i in 0..n {
        for r in 0..k {
            let mut cpu = 0.0f32;
            for j in 0..d {
                cpu += (data[i * d + j] - mean[j]) * scal[r * d + j];
            }
            let gpu = proj[i * k + r];
            assert!(
                (cpu - gpu).abs() < 1e-3,
                "transform mismatch at {i},{r}: cpu={cpu} gpu={gpu}"
            );
        }
    }
}

#[test]
fn lda_eigenvalues_descending_nonnegative() {
    let ctx = get_context();
    let n = 300;
    let d = 5;
    let mut rng = Rng::with_seed(5);
    let mut data = Vec::with_capacity(n * d);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % 3;
        labels.push(c as f32);
        for j in 0..d {
            let off = match c {
                1 => 2.0 * gauss(&mut rng) as f32 * (j as f32) * 0.1,
                2 => -2.0 * gauss(&mut rng) as f32,
                _ => 0.0,
            };
            data.push(gauss(&mut rng) as f32 + off);
        }
    }
    let mut lda = LDA::new(LDAConfig { n_components: 2 });
    lda.fit(&ctx, &data, &labels, n, d).unwrap();
    let ev = lda.eigenvalues();
    assert_eq!(ev.len(), 2);
    for w in ev.windows(2) {
        assert!(
            w[0] >= w[1] - 1e-4,
            "eigenvalues must be descending: {ev:?}"
        );
    }
    assert!(ev[0] >= 0.0, "leading eigenvalue should be non-negative");
}
