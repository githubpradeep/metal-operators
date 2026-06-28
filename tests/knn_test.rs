use metal_operators::knn::{KNN, KNNConfig};
use metal_operators::metal::MetalContext;

fn cpu_knn(
    corpus: &[f32], nc: usize, d: usize,
    queries: &[f32], nq: usize, k: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut distances = vec![0.0f32; nq * k];
    let mut indices = vec![0u32; nq * k];

    for q in 0..nq {
        // Brute-force all distances
        let mut pairs: Vec<(f32, u32)> = (0..nc as u32)
            .map(|c| {
                let mut dist = 0.0;
                for dim in 0..d {
                    let diff = queries[q * d + dim] - corpus[c as usize * d + dim];
                    dist += diff * diff;
                }
                (dist, c)
            })
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for j in 0..k {
            distances[q * k + j] = pairs[j].0;
            indices[q * k + j] = pairs[j].1;
        }
    }
    (distances, indices)
}

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

#[test]
fn test_knn_naive_path() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (nc, nq, d, k) = (100, 20, 3, 3);
    let corpus = make_data(nc, d, 1);
    let queries = make_data(nq, d, 2);

    let mut knn = KNN::new(KNNConfig { k });
    knn.fit(&ctx, &corpus, nc, d).expect("fit");
    let (gpu_d, gpu_i) = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let (cpu_d, cpu_i) = cpu_knn(&corpus, nc, d, &queries, nq, k);

    for q in 0..nq {
        for j in 0..k {
            assert_eq!(
                gpu_i[q * k + j], cpu_i[q * k + j],
                "Indices differ at q={}, j={}: GPU={}, CPU={}",
                q, j, gpu_i[q * k + j], cpu_i[q * k + j],
            );
            assert!(
                (gpu_d[q * k + j] - cpu_d[q * k + j]).abs() < 1e-3,
                "Distance diff at q={}, j={}: GPU={}, CPU={}",
                q, j, gpu_d[q * k + j], cpu_d[q * k + j],
            );
        }
    }
}

#[test]
fn test_knn_simdgroup_path() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (nc, nq, d, k) = (200, 32, 16, 5);
    let corpus = make_data(nc, d, 10);
    let queries = make_data(nq, d, 20);

    let mut knn = KNN::new(KNNConfig { k });
    knn.fit(&ctx, &corpus, nc, d).expect("fit");
    let (gpu_d, gpu_i) = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let (cpu_d, cpu_i) = cpu_knn(&corpus, nc, d, &queries, nq, k);

    for q in 0..nq {
        for j in 0..k {
            assert_eq!(
                gpu_i[q * k + j], cpu_i[q * k + j],
                "Indices differ at q={}, j={}", q, j,
            );
            assert!(
                (gpu_d[q * k + j] - cpu_d[q * k + j]).abs() < 1e-3,
                "Distance diff at q={}, j={}", q, j,
            );
        }
    }
}

#[test]
fn test_knn_k10_d32() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (nc, nq, d, k) = (500, 50, 32, 10);
    let corpus = make_data(nc, d, 30);
    let queries = make_data(nq, d, 40);

    let mut knn = KNN::new(KNNConfig { k });
    knn.fit(&ctx, &corpus, nc, d).expect("fit");
    let (gpu_d, gpu_i) = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let (cpu_d, cpu_i) = cpu_knn(&corpus, nc, d, &queries, nq, k);

    for q in 0..nq {
        for j in 0..k {
            assert_eq!(
                gpu_i[q * k + j], cpu_i[q * k + j],
                "Indices differ at q={}, j={}", q, j,
            );
            assert!(
                (gpu_d[q * k + j] - cpu_d[q * k + j]).abs() < 1e-3,
                "Distance diff at q={}, j={}", q, j,
            );
        }
    }
}

#[test]
fn test_knn_deterministic() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (nc, nq, d, k) = (50, 10, 8, 3);
    let corpus = make_data(nc, d, 100);
    let queries = make_data(nq, d, 200);

    let mut knn1 = KNN::new(KNNConfig { k });
    knn1.fit(&ctx, &corpus, nc, d).expect("fit");
    let (d1, i1) = knn1.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let mut knn2 = KNN::new(KNNConfig { k });
    knn2.fit(&ctx, &corpus, nc, d).expect("fit");
    let (d2, i2) = knn2.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    assert_eq!(i1, i2, "Indices differ between runs");
    assert_eq!(d1, d2, "Distances differ between runs");
}

#[test]
fn test_knn_rejects_invalid() {
    let ctx = MetalContext::new().expect("No Metal device");
    let mut knn = KNN::new(KNNConfig { k: 5 });

    // zero dims
    assert!(knn.fit(&ctx, &[1.0, 2.0], 2, 0).is_err());
    // zero points
    assert!(knn.fit(&ctx, &[], 0, 2).is_err());
}

#[test]
fn test_knn_k1() {
    let ctx = MetalContext::new().expect("No Metal device");

    let (nc, nq, d, k) = (100, 10, 4, 1);
    let corpus = make_data(nc, d, 50);
    let queries = make_data(nq, d, 60);

    let mut knn = KNN::new(KNNConfig { k });
    knn.fit(&ctx, &corpus, nc, d).expect("fit");
    let (_gpu_d, gpu_i) = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let (_cpu_d, cpu_i) = cpu_knn(&corpus, nc, d, &queries, nq, k);
    assert_eq!(gpu_i, cpu_i, "k=1 indices differ");
}
