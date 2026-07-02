use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::knn::{KNN, KNNConfig};
use metal_operators::metal::MetalContext;
use metal_operators::pca::{PCA, PCAConfig};

fn main() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;

    // ── KMeans example ──
    println!("=== KMeans Example (synthetic data) ===");
    {
        let n = 1000;
        let d = 2;
        let k = 4;
        let mut rng = fastrand::Rng::with_seed(42);
        let mut centers = Vec::with_capacity(k * d);
        for i in 0..k {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
            centers.push(angle.cos() * 5.0);
            centers.push(angle.sin() * 5.0);
        }
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let cluster = i % k;
            let base = cluster * d;
            data[i * d] = centers[base] + (rng.f32() - 0.5) * 1.5;
            data[i * d + 1] = centers[base + 1] + (rng.f32() - 0.5) * 1.5;
        }

        let mut km = KMeans::new(KMeansConfig {
            k,
            max_iterations: 20,
            tolerance: 0.0,
            seed: 42,
            init_centroids: None,
        });
        km.fit(&ctx, &data, n, d)?;
        println!("Inertia: {:.4}", km.inertia());
        println!("Iterations: {}", km.n_iter());
        println!("Centroids: {:?}", km.centroids());
    }

    // ── KNN example ──
    println!("\n=== KNN Example (synthetic data) ===");
    {
        let n = 1000;
        let d = 2;
        let k = 3;
        let mut rng = fastrand::Rng::with_seed(42);
        let mut data = Vec::with_capacity(n * d);
        for _ in 0..n {
            data.push(rng.f32() * 10.0);
            data.push(rng.f32() * 10.0);
        }
        let queries = vec![5.0, 5.0]; // 1 query

        let mut knn = KNN::new(KNNConfig { k });
        knn.fit(&ctx, &data, n, d)?;
        let (distances, indices) = knn.kneighbors(&ctx, &queries, 1)?;
        println!("Query (5.0, 5.0) — nearest {}:", k);
        for j in 0..k {
            let idx = indices[j] as usize;
            println!(
                "  Index {}: pos=({:.2}, {:.2}), dist={:.4}",
                idx, data[idx * d], data[idx * d + 1], distances[j]
            );
        }
    }

    // ── PCA example ──
    println!("\n=== PCA Example (synthetic data) ===");
    {
        let n = 200;
        let d = 10;
        let k = 3;
        let mut rng = fastrand::Rng::with_seed(42);

        // Data with strong 2D structure + noise
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let x = rng.f32() * 100.0 - 50.0; // x variance ~833
            let y = (x * 0.5) + rng.f32() * 20.0 - 10.0; // y correlated with x
            data[i * d] = x;
            data[i * d + 1] = y;
            for dim in 2..d {
                data[i * d + dim] = rng.f32() - 0.5; // noise
            }
        }

        let mut pca = PCA::new(PCAConfig { n_components: k });
        pca.fit(&ctx, &data, n, d)?;

        println!("Data: {} points, {} dimensions", n, d);
        println!("Components ({} principal axes):", k);
        for (i, component) in pca.components().chunks(d).enumerate() {
            let comp_str: Vec<String> =
                component.iter().take(4).map(|v| format!("{:.4}", v)).collect();
            println!("  PC{}: [{}, ...]", i + 1, comp_str.join(", "));
        }
        println!("Explained variance: {:?}", pca.explained_variance());
        println!("Explained variance ratio: {:?}", pca.explained_variance_ratio());

        let transformed = pca.transform(&ctx, &data, n, d)?;
        println!("Transformed shape: {} x {}", n, k);
    }

    Ok(())
}
