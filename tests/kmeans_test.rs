use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

fn cpu_kmeans(data: &[f32], n: usize, d: usize, k: usize, max_iter: usize, centroids: &mut [f32]) -> (Vec<usize>, f32) {
    let mut labels = vec![0usize; n];
    let tolerance = 1e-4;

    for _iter in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut min_dist = f32::MAX;
            let mut best = 0;
            for c in 0..k {
                let mut dist = 0.0;
                for dim in 0..d {
                    let diff = data[i * d + dim] - centroids[c * d + dim];
                    dist += diff * diff;
                }
                if dist < min_dist {
                    min_dist = dist;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0u64; k];
        for i in 0..n {
            let label = labels[i];
            counts[label] += 1;
            for dim in 0..d {
                sums[label * d + dim] += data[i * d + dim];
            }
        }

        let mut max_shift = 0.0;
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for dim in 0..d {
                    let old = centroids[c * d + dim];
                    let new = sums[c * d + dim] * inv;
                    centroids[c * d + dim] = new;
                    let shift = (new - old).abs();
                    if shift > max_shift {
                        max_shift = shift;
                    }
                }
            }
        }

        if max_shift < tolerance {
            break;
        }
    }

    let mut inertia = 0.0;
    for i in 0..n {
        let label = labels[i];
        let mut dist = 0.0;
        for dim in 0..d {
            let diff = data[i * d + dim] - centroids[label * d + dim];
            dist += diff * diff;
        }
        inertia += dist;
    }

    (labels, inertia)
}

fn adjusted_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n == 0 {
        return 1.0;
    }

    let n_clusters_true = labels_true.iter().max().unwrap_or(&0) + 1;
    let n_clusters_pred = labels_pred.iter().max().unwrap_or(&0) + 1;

    let mut contingency = vec![vec![0u64; n_clusters_pred]; n_clusters_true];
    for i in 0..n {
        contingency[labels_true[i]][labels_pred[i]] += 1;
    }

    let mut sum_comb_ij = 0u64;
    for i in 0..n_clusters_true {
        for j in 0..n_clusters_pred {
            let v = contingency[i][j];
            if v >= 2 {
                sum_comb_ij += v * (v - 1) / 2;
            }
        }
    }

    let mut sum_comb_a = 0u64;
    for i in 0..n_clusters_true {
        let v: u64 = contingency[i].iter().sum();
        if v >= 2 {
            sum_comb_a += v * (v - 1) / 2;
        }
    }

    let mut sum_comb_b = 0u64;
    for j in 0..n_clusters_pred {
        let v: u64 = (0..n_clusters_true).map(|i| contingency[i][j]).sum();
        if v >= 2 {
            sum_comb_b += v * (v - 1) / 2;
        }
    }

    let total_comb = if n >= 2 {
        (n * (n - 1) / 2) as u64
    } else {
        1
    };

    let expected = (sum_comb_a as f64) * (sum_comb_b as f64) / (total_comb as f64);
    let max_ari = ((sum_comb_a as f64 + sum_comb_b as f64) / 2.0).max(expected);
    let numerator = sum_comb_ij as f64 - expected;
    let denominator = max_ari - expected;

    if denominator.abs() < 1e-12 {
        1.0
    } else {
        numerator / denominator
    }
}

fn generate_blobs(n: usize, d: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut centers = Vec::with_capacity(k * d);
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        for dim in 0..d {
            if dim == 0 {
                centers.push(angle.cos() * 5.0);
            } else if dim == 1 {
                centers.push(angle.sin() * 5.0);
            } else {
                centers.push(rng.f32() * 4.0 - 2.0);
            }
        }
    }

    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let cluster = i % k;
        for dim in 0..d {
            let noise = (rng.f32() - 0.5) * 1.5;
            data.push(centers[cluster * d + dim] + noise);
        }
    }

    (data, centers)
}

#[test]
fn test_kmeans_ari_against_cpu_reference() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let (n, d, k) = (2000, 2, 4);
    let max_iter = 50;
    let (data, _centers) = generate_blobs(n, d, k, 42);

    let mut init_centroids = vec![0.0f32; k * d];
    for i in 0..k {
        let src = i * (n / k);
        for dim in 0..d {
            init_centroids[i * d + dim] = data[src * d + dim];
        }
    }

    let mut metal_kmeans = KMeans::new(KMeansConfig {
        k,
        max_iterations: max_iter,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: Some(init_centroids.clone()),
    });

    metal_kmeans.fit(&ctx, &data, n, d).expect("Metal KMeans fit failed");
    let metal_labels = metal_kmeans.labels().to_vec();

    let mut cpu_centroids = init_centroids;
    let (cpu_labels, cpu_inertia) = cpu_kmeans(&data, n, d, k, max_iter, &mut cpu_centroids);

    let ari = adjusted_rand_index(&cpu_labels, &metal_labels);
    assert!(
        ari >= 0.95,
        "ARI between Metal and CPU KMeans too low: {:.4}",
        ari
    );

    let metal_inertia: f32 = (0..n)
        .map(|i| {
            let label = metal_labels[i];
            (0..d).map(|dim| {
                let diff = data[i * d + dim] - cpu_centroids[label * d + dim];
                diff * diff
            }).sum::<f32>()
        })
        .sum();

    let inertia_ratio = (metal_inertia - cpu_inertia).abs() / cpu_inertia.max(1e-6);
    assert!(
        inertia_ratio < 0.1,
        "Inertia mismatch: Metal={:.1}, CPU={:.1}, ratio={:.4}",
        metal_inertia,
        cpu_inertia,
        inertia_ratio
    );
}

#[test]
fn test_kmeans_predict_matches_fit_labels() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let (n, d, k) = (1000, 2, 3);
    let (data, _centers) = generate_blobs(n, d, k, 7);

    let mut kmeans = KMeans::new(KMeansConfig {
        k,
        max_iterations: 30,
        tolerance: 1e-4,
        seed: 7,
        init_centroids: None,
    });

    kmeans.fit(&ctx, &data, n, d).expect("fit failed");
    let fit_labels = kmeans.labels().to_vec();

    let predict_labels = kmeans.predict(&ctx, &data, n, d).expect("predict failed");

    let agreement: f64 = fit_labels
        .iter()
        .zip(&predict_labels)
        .filter(|(a, b)| a == b)
        .count() as f64
        / fit_labels.len() as f64;

    assert!(
        agreement >= 0.99,
        "Predict labels disagree with fit labels: agreement={:.4}",
        agreement
    );
}

#[test]
fn test_kmeans_k1() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let (n, d, k) = (500, 4, 1);
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        for dim in 0..d {
            data[i * d + dim] = (i as f32) * 0.1;
        }
    }

    let mut kmeans = KMeans::new(KMeansConfig {
        k,
        max_iterations: 10,
        tolerance: 1e-4,
        seed: 0,
        init_centroids: None,
    });

    kmeans.fit(&ctx, &data, n, d).expect("KMeans k=1 fit failed");

    assert_eq!(kmeans.labels().len(), n);
    assert!(kmeans.labels().iter().all(|&l| l == 0));
    assert_eq!(kmeans.centroids().len(), d);
}

#[test]
fn test_kmeans_empty_cluster_does_not_panic() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let d = 2;
    let data = vec![0.0f32, 0.0, 10.0, 10.0, 20.0, 20.0];
    let n = 3;
    let k = 3;

    let mut kmeans = KMeans::new(KMeansConfig {
        k,
        max_iterations: 10,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });

    kmeans.fit(&ctx, &data, n, d).expect("fit with k=n failed");
    assert_eq!(kmeans.labels().len(), n);
    assert_eq!(kmeans.centroids().len(), k * d);
}

#[test]
fn test_kmeans_large_dimensions() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let (n, d, k) = (500, 128, 5);
    let (data, _centers) = generate_blobs(n, d, k, 99);

    let mut kmeans = KMeans::new(KMeansConfig {
        k,
        max_iterations: 30,
        tolerance: 1e-3,
        seed: 99,
        init_centroids: None,
    });

    kmeans.fit(&ctx, &data, n, d).expect("high-dim KMeans fit failed");

    assert_eq!(kmeans.labels().len(), n);
    assert_eq!(kmeans.centroids().len(), k * d);
}

#[test]
fn test_kmeans_deterministic_with_same_seed() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let (n, d, k) = (500, 4, 5);
    let (data, _centers) = generate_blobs(n, d, k, 123);

    let mut km1 = KMeans::new(KMeansConfig {
        k,
        max_iterations: 30,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });
    km1.fit(&ctx, &data, n, d).expect("first fit failed");

    let mut km2 = KMeans::new(KMeansConfig {
        k,
        max_iterations: 30,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });
    km2.fit(&ctx, &data, n, d).expect("second fit failed");

    let ari = adjusted_rand_index(km1.labels(), km2.labels());
    assert!(
        ari >= 0.99,
        "Determinism check: same seed gave different results (ARI={:.4})",
        ari
    );
}

#[test]
fn test_cpu_kmeans_smoke() {
    let (n, d, k) = (100, 2, 3);
    let (data, _centers) = generate_blobs(n, d, k, 42);

    let mut centroids: Vec<f32> = (0..k * d).map(|i| data[i]).collect();
    let (labels, inertia) = cpu_kmeans(&data, n, d, k, 50, &mut centroids);

    assert_eq!(labels.len(), n);
    assert!(inertia > 0.0);
    assert_eq!(centroids.len(), k * d);
}

#[test]
fn test_kmeans_rejects_invalid_params() {
    let ctx = MetalContext::new().expect("No Metal device available");
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let mut km = KMeans::new(KMeansConfig {
        k: 0,
        max_iterations: 10,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });
    assert!(km.fit(&ctx, &data, 2, 2).is_err());

    let mut km = KMeans::new(KMeansConfig {
        k: 10,
        max_iterations: 10,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });
    assert!(km.fit(&ctx, &data, 2, 2).is_err());
}

#[test]
fn test_adjusted_rand_index_identical() {
    let labels = vec![0, 0, 1, 1, 2, 2];
    let ari = adjusted_rand_index(&labels, &labels);
    assert!((ari - 1.0).abs() < 1e-10, "ARI of identical labels should be 1.0");
}

#[test]
fn test_adjusted_rand_index_perfect() {
    let labels_a = vec![0, 0, 1, 1, 2, 2];
    let labels_b = vec![2, 2, 1, 1, 0, 0];
    let ari = adjusted_rand_index(&labels_a, &labels_b);
    assert!((ari - 1.0).abs() < 1e-10, "ARI of permuted labels should be 1.0, got {ari}");
}

#[test]
fn test_adjusted_rand_index_random() {
    let labels_a = vec![0, 0, 0, 1, 1, 1];
    let labels_b = vec![0, 1, 0, 1, 0, 1];
    let ari = adjusted_rand_index(&labels_a, &labels_b);
    assert!(ari.abs() < 0.5, "ARI of near-random should be near 0, got {ari}");
}

/// Quick timing benchmark for fit throughput (not a correctness test).
/// Run with `-- --nocapture` to see timing output.
#[test]
fn test_fit_timing() {
    let ctx = MetalContext::new().expect("No Metal device available");

    let (n, d, k) = (1_000_000usize, 32, 16);
    let (data, centers) = generate_blobs(n, d, k, 42);

    // ── just the assign kernel ──
    use metal_operators::kmeans::{KMeansConfig, KMeans};
    use std::time::Instant;

    // Time a single assign pass using the public KMeans internals
    let init = centers.clone();
    let mut km = KMeans::new(KMeansConfig {
        k, max_iterations: 1, tolerance: 0.0, seed: 42,
        init_centroids: Some(init),
    });
    let start = Instant::now();
    // We'll time iteration 0 only (first assign + centroid update)
    km.fit(&ctx, &data, n, d).expect("fit");
    let t1 = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  N=1M_D=32_K=16 single iter: {:.2}ms", t1);
}
