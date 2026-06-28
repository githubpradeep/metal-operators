use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use metal_operators::knn::{KNN, KNNConfig};
use metal_operators::metal::MetalContext;
use std::process::Command;
use std::io::Write;

// ── BLAS via Accelerate (macOS) ────────────────────────────────────────────
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn sgemm_(
        transa: *const u8, transb: *const u8,
        m: *const i32, n: *const i32, k: *const i32,
        alpha: *const f32, a: *const f32, lda: *const i32,
        b: *const f32, ldb: *const i32,
        beta: *const f32, c: *mut f32, ldc: *const i32,
    );
}

fn blas_pairwise_dot(queries: &[f32], corpus: &[f32], nq: usize, nc: usize, d: usize) -> Vec<f32> {
    let mut dots = vec![0.0f32; nq * nc];
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = nc as i32;
    let n_ = nq as i32;
    let k_ = d as i32;
    let lda = d as i32;
    let ldb = d as i32;
    let ldc = nc as i32;
    let trans: u8 = b'T';
    let notrans: u8 = b'N';
    unsafe {
        sgemm_(&trans, &notrans, &m, &n_, &k_, &alpha, corpus.as_ptr(), &lda, queries.as_ptr(), &ldb, &beta, dots.as_mut_ptr(), &ldc);
    }
    dots
}

fn cpu_brute_knn(
    queries: &[f32], corpus: &[f32], nq: usize, nc: usize, d: usize, k: usize,
) -> (Vec<f32>, Vec<usize>) {
    let dots = blas_pairwise_dot(queries, corpus, nq, nc, d);

    let mut qnorms = Vec::with_capacity(nq);
    for q in 0..nq {
        let mut s = 0.0;
        for dim in 0..d { s += queries[q * d + dim] * queries[q * d + dim]; }
        qnorms.push(s);
    }
    let mut cnorms = Vec::with_capacity(nc);
    for c in 0..nc {
        let mut s = 0.0;
        for dim in 0..d { s += corpus[c * d + dim] * corpus[c * d + dim]; }
        cnorms.push(s);
    }

    let mut out_d = vec![0.0f32; nq * k];
    let mut out_i = vec![0usize; nq * k];

    for q in 0..nq {
        let base = q * nc;
        let mut pairs: Vec<(f32, usize)> = (0..nc)
            .map(|c| {
                let dist = qnorms[q] + cnorms[c] - 2.0 * dots[base + c];
                (dist, c)
            })
            .collect();
        pairs.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        pairs[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for j in 0..k {
            out_d[q * k + j] = pairs[j].0;
            out_i[q * k + j] = pairs[j].1;
        }
    }
    (out_d, out_i)
}

fn generate_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

fn bench_knn_assign(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("No Metal device available");

    let mut group = c.benchmark_group("knn_assign");
    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("Q=1K_C=10K_D=8_K=5",    1_000, 10_000, 8,   5),
        ("Q=1K_C=10K_D=32_K=5",   1_000, 10_000, 32,  5),
        ("Q=1K_C=10K_D=128_K=5",  1_000, 10_000, 128, 5),
        ("Q=1K_C=50K_D=8_K=5",    1_000, 50_000, 8,   5),
        ("Q=1K_C=50K_D=32_K=5",   1_000, 50_000, 32,  5),
        ("Q=10K_C=10K_D=8_K=5",  10_000, 10_000, 8,   5),
        ("Q=10K_C=10K_D=32_K=5", 10_000, 10_000, 32,  5),
        ("Q=10K_C=50K_D=8_K=5",  10_000, 50_000, 8,   5),
        ("Q=10K_C=50K_D=32_K=5", 10_000, 50_000, 32,  5),
    ];

    for &(label, nq, nc, d, k) in shapes {
        let corpus = generate_data(nc, d, 1);
        let queries = generate_data(nq, d, 2);

        // Metal
        let mut knn = KNN::new(KNNConfig { k });
        knn.fit(&ctx, &corpus, nc, d).expect("fit");
        group.bench_with_input(BenchmarkId::new(label, "metal"), &nq, |b, _| {
            b.iter(|| {
                let _ = knn.kneighbors(black_box(&ctx), black_box(&queries), black_box(nq));
            });
        });

        // BLAS-CPU
        let qc = queries.clone();
        let cc = corpus.clone();
        group.bench_with_input(BenchmarkId::new(label, "blas-cpu"), &nq, |b, _| {
            b.iter(|| {
                black_box(cpu_brute_knn(&qc, &cc, nq, nc, d, k));
            });
        });
    }
    group.finish();
}

fn bench_knn_kneighbors(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("No Metal device available");

    // Compare full kneighbors: Metal vs sklearn (brute).
    struct Shape { label: &'static str, nq: usize, nc: usize, d: usize, k: usize }
    let shapes: &[Shape] = &[
        Shape { label: "Q=1K_C=10K_D=8_K=5",    nq: 1_000,  nc: 10_000, d: 8,   k: 5 },
        Shape { label: "Q=1K_C=10K_D=32_K=5",   nq: 1_000,  nc: 10_000, d: 32,  k: 5 },
        Shape { label: "Q=1K_C=50K_D=32_K=5",   nq: 1_000,  nc: 50_000, d: 32,  k: 5 },
        Shape { label: "Q=5K_C=10K_D=32_K=10",  nq: 5_000,  nc: 10_000, d: 32,  k: 10 },
    ];

    let mut group = c.benchmark_group("knn_kneighbors");
    group.sample_size(10);

    for s in shapes {
        let corpus = generate_data(s.nc, s.d, 1);
        let queries = generate_data(s.nq, s.d, 2);

        let label = s.label;
        let nq = s.nq; let nc = s.nc; let d = s.d; let k = s.k;

        // Metal
        let mut knn = KNN::new(KNNConfig { k });
        knn.fit(&ctx, &corpus, nc, d).expect("fit");
        group.bench_with_input(BenchmarkId::new(label, "metal"), &nq, |b, _| {
            b.iter(|| {
                let _ = knn.kneighbors(black_box(&ctx), black_box(&queries), black_box(nq));
            });
        });

        // sklearn
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("benches")
            .join("sklearn_knn.py");
        let q_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(queries.as_ptr() as *const u8, nq * d * 4)
        };
        let c_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(corpus.as_ptr() as *const u8, nc * d * 4)
        };
        let mut input_data = Vec::new();
        writeln!(input_data, "{} {} {} {} {}", nq, nc, d, k, 1).unwrap();
        input_data.extend_from_slice(q_bytes);
        input_data.extend_from_slice(c_bytes);

        let nbytes = (nq + nc) * d * 4;
        if nbytes > 250_000_000 {
            eprintln!("  sklearn {}: SKIPPED ({} MB — pipe too large)", label, nbytes >> 20);
        } else {
            group.bench_with_input(BenchmarkId::new(label, "sklearn"), &nq, |b, _| {
                b.iter(|| {
                    let mut child = Command::new("python3")
                        .arg(script.to_str().unwrap())
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::null())
                        .spawn()
                        .expect("failed to launch python3 — is sklearn installed?");

                    let inp = input_data.clone();
                    child.stdin.take().unwrap().write_all(&inp).expect("stdin write");
                    let out = child.wait_with_output().expect("subprocess failed");
                    assert!(out.status.success(), "sklearn failed: {:?}", String::from_utf8_lossy(&out.stderr));
                    let mut buf = [0u8; 4];
                    buf.copy_from_slice(&out.stdout[..4]);
                    black_box(f32::from_le_bytes(buf));
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_knn_assign, bench_knn_kneighbors);
criterion_main!(benches);
