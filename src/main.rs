use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

fn main() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    println!("Metal device: {}", ctx.device.name());

    // Generate synthetic 2D data with 3 clusters
    let (points, n, d) = generate_data(3000, 2, 3);

    let mut kmeans = KMeans::new(KMeansConfig {
        k: 3,
        max_iterations: 50,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });

    let start = std::time::Instant::now();
    kmeans.fit(&ctx, &points, n, d)?;
    let elapsed = start.elapsed();

    println!("KMeans converged in {} iterations ({:.2?})", kmeans.n_iter(), elapsed);
    println!("Inertia: {:.4}", kmeans.inertia());

    for (i, c) in kmeans.centroids().chunks(d).enumerate() {
        println!("  Centroid {}: [{:.4}, {:.4}]", i, c[0], c[1]);
    }

    let label_counts = count_labels(kmeans.labels(), 3);
    println!("Cluster sizes: {:?}", label_counts);

    Ok(())
}

fn generate_data(n: usize, d: usize, n_clusters: usize) -> (Vec<f32>, usize, usize) {
    use fastrand::Rng;
    let mut rng = Rng::with_seed(123);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|i| {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / n_clusters as f32;
            vec![angle.cos() * 5.0, angle.sin() * 5.0]
        })
        .collect();

    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let cluster = i % n_clusters;
        let center = &centers[cluster];
        for dim in 0..d {
            let noise = rng.f32() * 1.0 - 0.5;
            data.push(center[dim] + noise);
        }
    }

    (data, n, d)
}

fn count_labels(labels: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0usize; k];
    for &l in labels {
        counts[l] += 1;
    }
    counts
}
