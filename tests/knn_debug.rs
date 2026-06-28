use metal_operators::metal::MetalContext;
use metal_operators::knn::{KNN, KNNConfig};

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

fn cpu_knn(corpus: &[f32], nc: usize, d: usize, queries: &[f32], nq: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut distances = vec![0.0f32; nq * k];
    let mut indices = vec![0u32; nq * k];
    for q in 0..nq {
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

#[test]
fn test_knn_debug_tiny() {
    let ctx = MetalContext::new().expect("Metal");

    // Tiny: 10 corpus, 2 queries, D=3, K=2 — forces Dense path
    let (nc, nq, d, k) = (10, 2, 3, 2);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);

    eprintln!("corpus: {:?}", &corpus[..30]);
    eprintln!("queries: {:?}", &queries[..6]);

    let mut knn = KNN::new(KNNConfig { k });
    knn.fit(&ctx, &corpus, nc, d).expect("fit");
    let (gpu_d, gpu_i) = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let (cpu_d, cpu_i) = cpu_knn(&corpus, nc, d, &queries, nq, k);

    eprintln!("GPU dists: {:?}", gpu_d);
    eprintln!("GPU idxs: {:?}", gpu_i);
    eprintln!("CPU dists: {:?}", cpu_d);
    eprintln!("CPU idxs: {:?}", cpu_i);

    for q in 0..nq {
        for j in 0..k {
            assert_eq!(gpu_i[q*k+j], cpu_i[q*k+j], "idx q={} j={}", q, j);
            assert!((gpu_d[q*k+j] - cpu_d[q*k+j]).abs() < 1e-3, "dist q={} j={}", q, j);
        }
    }
}
