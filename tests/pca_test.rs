use metal_operators::metal::MetalContext;
use metal_operators::pca::{PCA, PCAConfig};

fn get_context() -> MetalContext {
    MetalContext::new().expect("Failed to create MetalContext - is this running on a Mac with Metal?")
}

fn compute_gram_xtx(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut gram = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += data[k * d + i] * data[k * d + j];
            }
            gram[i * d + j] = sum;
            gram[j * d + i] = sum;
        }
    }
    gram
}

#[test]
fn pca_fit_smoke_test() {
    let ctx = get_context();
    let n = 100;
    let d = 8;
    let k = 3;
    let data = generate_data(n, d, 42);

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d).expect("PCA fit failed");

    assert_eq!(pca.components().len(), k * d, "components should be (K, D)");
    assert_eq!(pca.explained_variance().len(), k, "explained_variance should be (K,)");
    assert_eq!(pca.explained_variance_ratio().len(), k, "explained_variance_ratio should be (K,)");
    assert_eq!(pca.mean().len(), d, "mean should be (D,)");
}

#[test]
fn pca_orthonormal_components() {
    let ctx = get_context();
    let n = 200;
    let d = 5;
    let k = 3;
    let data = generate_data(n, d, 123);

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d).unwrap();

    // Reshape components as (K, D) — already row-major
    let comps = pca.components();

    // Check orthonormality: each component should have unit norm
    for i in 0..k {
        let row = &comps[i * d..(i + 1) * d];
        let norm: f32 = row.iter().map(|x| x * x).sum();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Component {i} norm = {norm}, expected ~1.0"
        );
    }

    // Check orthogonality: different components should have near-zero dot product
    for i in 0..k {
        for j in (i + 1)..k {
            let row_i = &comps[i * d..(i + 1) * d];
            let row_j = &comps[j * d..(j + 1) * d];
            let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() < 1e-5,
                "Component dot({i},{j}) = {dot}, expected 0"
            );
        }
    }
}

#[test]
fn pca_explained_variance_sum() {
    let ctx = get_context();
    let n = 100;
    let d = 6;
    let k = d; // all components
    let data = generate_data(n, d, 77);

    // Compute total variance of centered data
    let means: Vec<f32> = (0..d)
        .map(|col| {
            data.iter()
                .skip(col)
                .step_by(d)
                .sum::<f32>()
                / n as f32
        })
        .collect();
    let centered: Vec<f32> = data
        .iter()
        .enumerate()
        .map(|(idx, &x)| x - means[idx % d])
        .collect();
    let total_var: f32 = centered.iter().map(|x| x * x).sum::<f32>() / n as f32;

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d).unwrap();

    let explained_var_sum: f32 = pca.explained_variance().iter().sum();
    let ratio_sum: f32 = pca.explained_variance_ratio().iter().sum();

    // When K=D, explained variance should sum to total variance
    let ratio = (explained_var_sum - total_var).abs() / total_var.max(1.0);
    assert!(
        ratio < 1e-3,
        "explained variance sum {explained_var_sum} != total variance {total_var}"
    );
    assert!(
        (ratio_sum - 1.0).abs() < 1e-3,
        "variance ratio sum = {ratio_sum}, expected 1.0"
    );
}

#[test]
fn pca_reconstruction_with_all_components() {
    let ctx = get_context();
    let n = 60;
    let d = 4;
    let k = d; // use all components
    let data = generate_data(n, d, 99);

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d).unwrap();

    // Transform to latent space
    let transformed = pca.transform(&ctx, &data, n, d).unwrap();
    assert_eq!(transformed.len(), n * k);

    // Reconstruct: X_hat = transformed @ components + mean
    let mut reconstructed = vec![0.0f32; n * d];
    let comps = pca.components();
    let mean = pca.mean();
    for i in 0..n {
        for j in 0..d {
            let mut val = 0.0;
            for comp in 0..k {
                val += transformed[i * k + comp] * comps[comp * d + j];
            }
            reconstructed[i * d + j] = val + mean[j];
        }
    }

    // Check reconstruction error (should be near zero with all components)
    let mse: f32 = (0..n * d)
        .map(|idx| {
            let diff = data[idx] - reconstructed[idx];
            diff * diff
        })
        .sum::<f32>()
        / (n * d) as f32;

    assert!(
        mse < 1e-5,
        "Reconstruction MSE = {mse}, expected near zero"
    );
}

#[test]
fn pca_deterministic() {
    let ctx = get_context();
    let data = generate_data(80, 5, 42);

    let mut pca1 = PCA::new(PCAConfig { n_components: 3 });
    pca1.fit(&ctx, &data, 80, 5).unwrap();

    let mut pca2 = PCA::new(PCAConfig { n_components: 3 });
    pca2.fit(&ctx, &data, 80, 5).unwrap();

    // Components should be identical (component signs may differ, but dot products should be 1)
    let c1 = pca1.components();
    let c2 = pca2.components();
    for comp in 0..3 {
        let row1 = &c1[comp * 5..(comp + 1) * 5];
        let row2 = &c2[comp * 5..(comp + 1) * 5];
        let dot: f32 = row1.iter().zip(row2.iter()).map(|(a, b)| a * b).sum();
        // Dot should be ~1.0 (or ~-1.0 if sign flip)
        assert!(
            dot.abs() > 0.99,
            "Component {comp} dot = {dot}, expected ±1"
        );
    }

    // Variance should be exactly the same
    for i in 0..3 {
        let v1 = pca1.explained_variance()[i];
        let v2 = pca2.explained_variance()[i];
        assert!(
            (v1 - v2).abs() < 1e-6,
            "Variance mismatch at {i}: {v1} vs {v2}"
        );
    }
}

#[test]
fn pca_n_less_than_d() {
    // Test the transpose trick path (N < D)
    let ctx = get_context();
    let n = 20;
    let d = 50;
    let k = 5;
    let data = generate_data(n, d, 33);

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d).unwrap();

    assert_eq!(pca.components().len(), k * d);
    assert_eq!(pca.explained_variance().len(), k);

    // Components should be orthonormal
    let comps = pca.components();
    for i in 0..k {
        let row = &comps[i * d..(i + 1) * d];
        let norm: f32 = row.iter().map(|x| x * x).sum();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Component {i} norm = {norm}"
        );
    }
}

#[test]
fn pca_single_component() {
    let ctx = get_context();
    let data = generate_data(50, 7, 55);

    let mut pca = PCA::new(PCAConfig { n_components: 1 });
    pca.fit(&ctx, &data, 50, 7).unwrap();

    assert_eq!(pca.components().len(), 1 * 7);
    assert_eq!(pca.explained_variance().len(), 1);
    assert_eq!(pca.explained_variance_ratio().len(), 1);

    // Transform should produce (N, 1)
    let transformed = pca.transform(&ctx, &data, 50, 7).unwrap();
    assert_eq!(transformed.len(), 50);
}

#[test]
fn pca_transform_output_shape() {
    let ctx = get_context();
    let data = generate_data(30, 10, 11);

    let mut pca = PCA::new(PCAConfig { n_components: 4 });
    pca.fit(&ctx, &data, 30, 10).unwrap();

    let transformed = pca.transform(&ctx, &data, 30, 10).unwrap();
    assert_eq!(transformed.len(), 30 * 4);

    // Fit-transform should give same result
    let mut pca2 = PCA::new(PCAConfig { n_components: 4 });
    let transformed2 = pca2.fit_transform(&ctx, &data, 30, 10).unwrap();
    assert_eq!(transformed2.len(), 30 * 4);

    // Results should match
    for i in 0..transformed.len() {
        assert!(
            (transformed[i] - transformed2[i]).abs() < 1e-5,
            "Difference at {i}"
        );
    }
}

#[test]
fn pca_high_dim_low_samples() {
    // N < D with high dimensionality
    let ctx = get_context();
    let n = 30;
    let d = 100;
    let k = 10;
    let data = generate_data(n, d, 2024);

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d).unwrap();

    assert_eq!(pca.components().len(), k * d);
    assert_eq!(pca.explained_variance().len(), k);

    // Orthonormality
    let comps = pca.components();
    for i in 0..k {
        let norm: f32 = comps[i * d..(i + 1) * d].iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-3, "Component {i} norm = {norm}");
    }
}

#[test]
fn pca_variance_ratio_descending() {
    let ctx = get_context();
    let data = generate_data(200, 8, 999);

    let mut pca = PCA::new(PCAConfig { n_components: 5 });
    pca.fit(&ctx, &data, 200, 8).unwrap();

    // Explained variance should be strictly decreasing
    for i in 1..5 {
        assert!(
            pca.explained_variance()[i - 1] >= pca.explained_variance()[i] - 1e-6,
            "Variance not descending at {i}"
        );
        assert!(
            pca.explained_variance_ratio()[i - 1] >= pca.explained_variance_ratio()[i] - 1e-6,
            "Ratio not descending at {i}"
        );
    }
}

#[test]
fn pca_fit_transform_on_subset() {
    // Test that fit on subset and transform on full works
    let ctx = get_context();
    let mut data = generate_data(100, 6, 42);

    // Fit on first 60 samples
    let mut pca = PCA::new(PCAConfig { n_components: 2 });
    pca.fit(&ctx, &data[..60 * 6], 60, 6).unwrap();

    // Transform all 100 samples
    let transformed = pca.transform(&ctx, &data, 100, 6).unwrap();
    assert_eq!(transformed.len(), 100 * 2);
}

#[test]
fn pca_against_known_data() {
    // Test with a simple known dataset: 3 points in 2D
    // After centering, the principal component should align with the direction of max variance
    let ctx = get_context();
    let n = 10;
    let d = 2;
    // Data: points roughly along line y = 2x
    let data: Vec<f32> = (0..n)
        .flat_map(|i| {
            let x = i as f32 - 4.5; // centered around 0
            vec![x, x * 2.0 + ((i as f32) * 0.1)]
        })
        .collect();

    let mut pca = PCA::new(PCAConfig { n_components: 1 });
    pca.fit(&ctx, &data, n, d).unwrap();

    // First component should roughly be in direction (1, 2) / sqrt(5)
    let c = &pca.components()[0..2];
    let expected = 1.0 / (5.0f32).sqrt();
    // Allow sign ambiguity
    let dot = c[0] * expected + c[1] * (2.0 * expected);
    assert!(
        dot.abs() > 0.9,
        "Component [{:.4}, {:.4}] should align with (1, 2), dot = {:.4}",
        c[0],
        c[1],
        dot
    );
}

// ── helpers ────────────────────────────────────────────────────

fn generate_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    use fastrand::Rng;
    let mut rng = Rng::with_seed(seed);
    let mut data = Vec::with_capacity(n * d);
    for _ in 0..n {
        for _ in 0..d {
            data.push(rng.f32() * 10.0 - 5.0);
        }
    }
    data
}
