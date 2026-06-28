use metal_operators::knn::{KNN, KNNConfig};
use metal_operators::metal::MetalContext;
use std::time::Instant;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

#[test]
fn test_knn_timing() {
    let ctx = MetalContext::new().expect("Metal");

    let (nc, nq, d, k) = (10000, 1000, 8, 5);
    let corpus = make_data(nc, d, 1);
    let queries = make_data(nq, d, 2);

    let mut knn = KNN::new(KNNConfig { k });

    let t0 = Instant::now();
    knn.fit(&ctx, &corpus, nc, d).expect("fit");
    eprintln!("fit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // warmup
    let _ = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");

    let t0 = Instant::now();
    let _ = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");
    let t = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("kneighbors: {:.1}ms", t);

    // batch
    let t0 = Instant::now();
    for _ in 0..20 {
        let _ = knn.kneighbors(&ctx, &queries, nq).expect("kneighbors");
    }
    let t = t0.elapsed().as_secs_f64() / 20.0 * 1000.0;
    eprintln!("kneighbors avg 20: {:.1}ms", t);
}
