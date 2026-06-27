#[test]
fn test_kmeans_k64_d32() {
    use metal_operators::kmeans::{KMeans, KMeansConfig};
    use metal_operators::metal::MetalContext;

    let ctx = MetalContext::new().expect("metal");
    let (n, d, k) = (256, 32, 64);
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
    // CPU reference
    let cpu_centroids = init.clone();
    let mut cpu_labels = vec![0u32; n];
    for _iter in 0..5 {
        for i in 0..n {
            let mut best_d = f32::INFINITY; let mut best = 0u32;
            for j in 0..k {
                let mut d2 = 0.0;
                for dim in 0..d { let diff = data[i*d+dim] - cpu_centroids[j*d+dim]; d2 += diff*diff; }
                if d2 < best_d { best_d = d2; best = j as u32; }
            }
            cpu_labels[i] = best;
        }
        // skip centroid update for this test
    }

    let mut km = KMeans::new(KMeansConfig {
        k, max_iterations: 5, tolerance: 1e-4, seed: 42,
        init_centroids: Some(init),
    });
    km.fit(&ctx, &data, n, d).expect("fit failed");
    let labels = km.labels();
    let mismatches: Vec<_> = (0..n).filter(|&i| labels[i] as u32 != cpu_labels[i]).collect();
    println!("Mismatches: {}/{}", mismatches.len(), n);
    if mismatches.len() > 0 {
        println!("First mismatch: point {}: metal={} cpu={}", mismatches[0], labels[mismatches[0]], cpu_labels[mismatches[0]]);
        println!("CPU labels[..20] = {:?}", &cpu_labels[..20]);
        println!("Metal labels[..20] = {:?}", &labels[..20]);
    }
    assert!(mismatches.len() < n / 2, "too many mismatches");
}
