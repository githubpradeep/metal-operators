// Test: simdgroup kernel with K=16 (2 centroid tiles) assigns correctly.
// Centroids 0..7 are near origin; centroids 8..15 are at x≈100.
// Points 0..7 are near centroids 0..7; points 8..15 are near centroids 8..15.
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

#[test]
fn test_simdgroup_multitile_16() {
    let ctx = MetalContext::new().expect("metal");
    let (n, d, k) = (16usize, 8usize, 16usize);

    let mut centroids = vec![0.0f32; k * d];
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        centroids[i * d] = angle.cos() * 5.0 + if i >= 8 { 100.0 } else { 0.0 };
        centroids[i * d + 1] = angle.sin() * 5.0;
        // dims 2..7 are 0
    }

    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % k;
        data[i * d] = centroids[cluster * d];
        data[i * d + 1] = centroids[cluster * d + 1];
        for dim in 2..d {
            data[i * d + dim] = (i as f32 * 0.1 + dim as f32 * 0.01).fract() - 0.5;
        }
    }

    let mut km = KMeans::new(KMeansConfig {
        k,
        max_iterations: 1,
        tolerance: 0.0,
        seed: 42,
        init_centroids: Some(centroids.clone()),
    });
    km.fit(&ctx, &data, n, d).expect("fit");
    let labels = km.labels();
    eprintln!("Labels: {:?}", &labels);

    for i in 0..n {
        let expected = i;
        assert_eq!(labels[i], expected,
            "Point {} misassigned: expected centroid {}, got {}", i, expected, labels[i]);
    }
    eprintln!("ALL CORRECT");
}
