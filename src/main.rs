use metal_operators::dbscan::{DBSCANConfig, DBSCAN};
use metal_operators::gmm::{GMMConfig, GMM};
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::knn::{KNNConfig, KNN};
use metal_operators::metal::MetalContext;
use metal_operators::nmf::{NMFConfig, NMF};
use metal_operators::pca::{PCAConfig, PCA};

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
                idx,
                data[idx * d],
                data[idx * d + 1],
                distances[j]
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
            let comp_str: Vec<String> = component
                .iter()
                .take(4)
                .map(|v| format!("{:.4}", v))
                .collect();
            println!("  PC{}: [{}, ...]", i + 1, comp_str.join(", "));
        }
        println!("Explained variance: {:?}", pca.explained_variance());
        println!(
            "Explained variance ratio: {:?}",
            pca.explained_variance_ratio()
        );

        let transformed = pca.transform(&ctx, &data, n, d)?;
        println!("Transformed shape: {} x {}", n, k);
    }

    // ── DBSCAN example ──
    println!("\n=== DBSCAN Example (synthetic blobs + noise) ===");
    {
        let n = 40;
        let d = 2;
        let mut rng = fastrand::Rng::with_seed(42);
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            let cx = if i % 2 == 0 { 0.0 } else { 10.0 };
            let cy = if i % 2 == 0 { 0.0 } else { 10.0 };
            data.push(cx + (rng.f32() - 0.5) * 1.2);
            data.push(cy + (rng.f32() - 0.5) * 1.2);
        }
        // one obvious outlier
        data.push(50.0);
        data.push(50.0);
        let n = n + 1;

        let mut db = DBSCAN::new(DBSCANConfig {
            eps: 2.0,
            min_samples: 4,
        });
        db.fit(&ctx, &data, n, d)?;
        println!("Clusters found: {}", db.n_clusters());
        println!("Labels: {:?}", db.labels());
    }

    // ── NMF example ──
    println!("\n=== NMF Example (non-negative synthetic matrix) ===");
    {
        let n = 500;
        let d = 40;
        let k = 4;
        let mut rng = fastrand::Rng::with_seed(7);
        // Build a low-rank non-negative matrix V = W_true · H_true + noise.
        let mut w_true = Vec::with_capacity(n * k);
        for _ in 0..n * k {
            w_true.push(rng.f32());
        }
        let mut h_true = Vec::with_capacity(k * d);
        for _ in 0..k * d {
            h_true.push(rng.f32());
        }
        let mut v = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let mut s = 0.0f32;
                for c in 0..k {
                    s += w_true[i * k + c] * h_true[c * d + j];
                }
                v[i * d + j] = s + rng.f32() * 0.01; // small non-negative noise
            }
        }
        println!("V: {} x {}, rank {}", n, d, k);

        let mut nmf = NMF::new(NMFConfig {
            n_components: k,
            max_iterations: 200,
            tolerance: 1e-5,
            seed: 42,
            eps: 1e-10,
        });
        nmf.fit(&ctx, &v, n, d)?;
        println!(
            "Reconstruction error (Frobenius): {:.6} after {} iters",
            nmf.reconstruction_error(),
            nmf.n_iter()
        );
        println!("Components (H): shape {} x {}", k, d);

        // Transform a small held-out slice back into latent space.
        let held = &v[..100 * d];
        let latent = nmf.transform(&ctx, held, 100, d)?;
        println!("Transformed (held-out) shape: 100 x {}", latent.len() / 100);
    }

    // ── GMM example ──
    println!("\n=== GMM Example (three Gaussian blobs) ===");
    {
        let n = 600;
        let d = 3;
        let mut rng = fastrand::Rng::with_seed(7);
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let c = i / 200; // blob 0, 1, 2
            for dim in 0..d {
                let center = if dim == c { 6.0 } else { 0.0 };
                data[i * d + dim] = center + (rng.f32() - 0.5) * 1.2;
            }
        }

        let mut gmm = GMM::new(GMMConfig {
            n_components: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            seed: 42,
            reg_covar: 1e-4,
        });
        gmm.fit(&ctx, &data, n, d)?;
        println!(
            "Lower bound (avg log-likelihood): {:.4} after {} iterations",
            gmm.lower_bound(),
            gmm.n_iter()
        );
        println!("Weights: {:?}", gmm.weights());
        for c in 0..3 {
            let m = &gmm.means()[c * d..(c + 1) * d];
            println!(
                "  Component {} mean: ({:.3}, {:.3}, {:.3})",
                c, m[0], m[1], m[2]
            );
        }

        let preds = gmm.predict(&ctx, &data, n, d)?;
        // Best-over-permutation purity: GMM component labeling is arbitrary.
        let mut purity = 0.0f32;
        for a in 0..3 {
            for b in 0..3 {
                if b == a {
                    continue;
                }
                for c2 in 0..3 {
                    if c2 == a || c2 == b {
                        continue;
                    }
                    let perm = [a, b, c2];
                    let matches = (0..n).filter(|&i| preds[i] == perm[i / 200]).count() as f32;
                    purity = purity.max(matches / n as f32);
                }
            }
        }
        println!("Cluster purity on training data: {:.3}", purity);
        println!(
            "Score on training data: {:.4}",
            gmm.score(&ctx, &data, n, d)?
        );
    }

    Ok(())
}
