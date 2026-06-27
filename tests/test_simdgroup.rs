// quick test: run KMeans with K=256 using simdgroup kernel
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

#[test]
fn test_simdgroup_k256() {
    let ctx = MetalContext::new().expect("metal");

    let (n, d, k) = (640, 32, 256);
    let mut rng = fastrand::Rng::with_seed(42);
    let mut centers = vec![0.0f32; k * d];
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        for dim in 0..d {
            centers[i * d + dim] = if dim == 0 { angle.cos() * 5.0 }
                else if dim == 1 { angle.sin() * 5.0 }
                else { rng.f32() * 4.0 - 2.0 };
        }
    }
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % k;
        for dim in 0..d {
            data[i * d + dim] = centers[cluster * d + dim] + (rng.f32() - 0.5) * 1.5;
        }
    }

    let init = centers.clone();

    let mut km = KMeans::new(KMeansConfig {
        k,
        max_iterations: 5,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: Some(init),
    });

    km.fit(&ctx, &data, n, d).expect("fit failed");
    let labels = km.labels();
    assert_eq!(labels.len(), n);
    assert!(labels.iter().all(|&l| l < 256), "labels contain values >= 256: {:?}", &labels[..20]);
}
