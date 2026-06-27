use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use metal_operators::kmeans::{KMeans, KMeansConfig};
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

fn blas_pairwise_distances(x: &[f32], c: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut dists = vec![0.0f32; n * k];
    let alpha: f32 = -2.0;
    let beta: f32 = 0.0;
    let m = k as i32;
    let n_ = n as i32;
    let k_ = d as i32;
    let lda = d as i32;
    let ldb = d as i32;
    let ldc = k as i32;
    let trans: u8 = b'T';
    let notrans: u8 = b'N';
    unsafe {
        sgemm_(&trans, &notrans, &m, &n_, &k_, &alpha, c.as_ptr(), &lda, x.as_ptr(), &ldb, &beta, dists.as_mut_ptr(), &ldc);
    }
    let mut xnorms = vec![0.0f32; n];
    for i in 0..n {
        let mut s = 0.0;
        for dd in 0..d { s += x[i * d + dd] * x[i * d + dd]; }
        xnorms[i] = s;
    }
    for i in 0..n { for j in 0..k { dists[i * k + j] += xnorms[i]; } }
    let mut cnorms = vec![0.0f32; k];
    for j in 0..k {
        let mut s = 0.0;
        for dd in 0..d { s += c[j * d + dd] * c[j * d + dd]; }
        cnorms[j] = s;
    }
    for i in 0..n { for j in 0..k { dists[i * k + j] += cnorms[j]; } }
    dists
}

// ── data generation ───────────────────────────────────────────────────────
fn generate_data(n: usize, d: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut centers = Vec::with_capacity(k * d);
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        for dim in 0..d {
            centers.push(if dim == 0 { angle.cos() * 5.0 }
                else if dim == 1 { angle.sin() * 5.0 }
                else { rng.f32() * 4.0 - 2.0 });
        }
    }
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let cluster = i % k;
        for dim in 0..d {
            data.push(centers[cluster * d + dim] + (rng.f32() - 0.5) * 1.5);
        }
    }
    (data, centers)
}

// ── sklearn reference via Python subprocess ────────────────────────────────
fn time_sklearn_kmeans(data: &[f32], n: usize, d: usize, k: usize, max_iter: usize,
                        tol: f32, init: Option<&[f32]>) -> f64 {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("sklearn_kmeans.py");

    let byte_len = (n * d * 4) as i32;
    let data_bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, n * d * 4)
    };
    let init_bytes = init.map(|c| unsafe {
        std::slice::from_raw_parts(c.as_ptr() as *const u8, k * d * 4)
    });
    let input_data = {
        let mut buf = Vec::new();
        writeln!(buf, "{} {} {} {} {} {}", n, d, k, max_iter, tol as i32, byte_len).unwrap();
        buf.extend_from_slice(data_bytes);
        if let Some(ib) = init_bytes {
            buf.extend_from_slice(b"1\n");
            buf.extend_from_slice(ib);
        } else {
            buf.push(b'\n');
        }
        buf
    };

    let mut child = Command::new("python3")
        .arg(script.to_str().unwrap())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to launch python3 — is sklearn installed?");

    child.stdin.take().unwrap().write_all(&input_data).expect("write to sklearn stdin");
    let output = child.wait_with_output().expect("sklearn subprocess failed");

    assert!(output.status.success(), "sklearn failed: {:?}", String::from_utf8_lossy(&output.stderr));

    let mut result = [0u8; 4];
    result.copy_from_slice(&output.stdout[..4]);
    f32::from_le_bytes(result) as f64
}

// ── assign kernel benchmarks ──────────────────────────────────────────────
fn bench_kmeans_assign_vs_blas(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("No Metal device available");
    let lib_src = include_str!("../shaders/kmeans.metal");
    let pipeline = ctx.compile_kernel(lib_src, "kmeans_assign").expect("compile assign kernel");

    let mut group = c.benchmark_group("kmeans_assign");
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("N=10K_D=2_K=8", 10_000, 2, 8),
        ("N=100K_D=2_K=8", 100_000, 2, 8),
        ("N=1M_D=2_K=8", 1_000_000, 2, 8),
        ("N=10K_D=32_K=16", 10_000, 32, 16),
        ("N=10K_D=64_K=16", 10_000, 64, 16),
        ("N=10K_D=64_K=256", 10_000, 64, 256),
        ("N=10K_D=128_K=32", 10_000, 128, 32),
        ("N=10K_D=128_K=256", 10_000, 128, 256),
        ("N=100K_D=64_K=16", 100_000, 64, 16),
        ("N=100K_D=32_K=256", 100_000, 32, 256),
    ];

    for &(label, n, d, k) in shapes {
        let (data, centers) = generate_data(n, d, k, 42);
        let point_buffer = ctx.new_buffer(&data);
        let centroids_buffer = ctx.new_buffer(&centers);
        let assign_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<u32>()) as u64);
        let dist_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<f32>()) as u64);
        let tg_size = 256u64;
        let groups = metal::MTLSize { width: (n as u64 + tg_size - 1) / tg_size, height: 1, depth: 1 };
        let tg_mtl = metal::MTLSize { width: tg_size, height: 1, depth: 1 };
        let n_u32 = n as u32; let k_u32 = k as u32; let d_u32 = d as u32;

        group.bench_with_input(BenchmarkId::new(label, "metal"), &n, |b, _| {
            b.iter(|| {
                let cmd_buffer = ctx.queue.new_command_buffer();
                let encoder = cmd_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&point_buffer), 0);
                encoder.set_buffer(1, Some(&centroids_buffer), 0);
                encoder.set_buffer(2, Some(&assign_buffer), 0);
                encoder.set_buffer(3, Some(&dist_buffer), 0);
                encoder.set_bytes(4, 4, std::ptr::from_ref(&n_u32).cast());
                encoder.set_bytes(5, 4, std::ptr::from_ref(&k_u32).cast());
                encoder.set_bytes(6, 4, std::ptr::from_ref(&d_u32).cast());
                encoder.dispatch_thread_groups(groups, tg_mtl);
                encoder.end_encoding();
                cmd_buffer.commit();
                cmd_buffer.wait_until_completed();
                black_box(());
            });
        });

        let data_c = data.clone();
        let centers_c = centers.clone();
        group.bench_with_input(BenchmarkId::new(label, "blas-cpu"), &n, |b, _| {
            b.iter(|| {
                black_box(blas_pairwise_distances(&data_c, &centers_c, n, d, k));
            });
        });
    }
    group.finish();
}

// ── full fit benchmark: Metal vs BLAS-CPU vs sklearn ───────────────────────
fn cpu_kmeans_blas(data: &[f32], n: usize, d: usize, k: usize, max_iter: usize, centroids: &mut [f32]) {
    let tolerance = 1e-4;
    let mut labels = vec![0u32; n];
    for _iter in 0..max_iter {
        let dists = blas_pairwise_distances(data, centroids, n, d, k);
        let mut changed = false;
        for i in 0..n {
            let base = i * k;
            let mut best = 0u32; let mut best_d = dists[base];
            for j in 1..k { if dists[base + j] < best_d { best_d = dists[base + j]; best = j as u32; } }
            if labels[i] != best { labels[i] = best; changed = true; }
        }
        if !changed { break; }
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0u64; k];
        for i in 0..n { let l = labels[i] as usize; counts[l] += 1; for dim in 0..d { sums[l * d + dim] += data[i * d + dim]; } }
        let mut max_shift = 0.0;
        for c in 0..k { if counts[c] > 0 { let inv = 1.0 / counts[c] as f32; for dim in 0..d { let old = centroids[c * d + dim]; let new = sums[c * d + dim] * inv; centroids[c * d + dim] = new; let shift = (new - old).abs(); if shift > max_shift { max_shift = shift; } } } }
        if max_shift < tolerance { break; }
    }
}

fn bench_kmeans_fit(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("No Metal device available");

    // flashlib shapes grouped by source benchmark
    // vs_cuml/standard : max_iter=15
    // vs_cuml/broad    : max_iter=3
    // vs_cuml/heavy    : max_iter=2..5
    struct Shape { label: &'static str, n: usize, d: usize, k: usize, max_iter: usize }
    let shapes: &[Shape] = &[
        // ── existing (keep for continuity) ──
        Shape { label: "N=10K_D=2_K=8",       n: 10_000,     d: 2,   k: 8,     max_iter: 15 },
        Shape { label: "N=50K_D=2_K=8",        n: 50_000,     d: 2,   k: 8,     max_iter: 15 },
        Shape { label: "N=10K_D=64_K=16",      n: 10_000,     d: 64,  k: 16,    max_iter: 15 },
        Shape { label: "N=10K_D=128_K=32",     n: 10_000,     d: 128, k: 32,    max_iter: 15 },

        // ── flashlib standard (vs_cuml/kmeans.py) ──
        Shape { label: "N=100K_D=32_K=256",    n: 100_000,    d: 32,  k: 256,   max_iter: 15 },
        Shape { label: "N=200K_D=64_K=512",    n: 200_000,    d: 64,  k: 512,   max_iter: 12 },
        Shape { label: "N=500K_D=64_K=1024",   n: 500_000,    d: 64,  k: 1024,  max_iter: 10 },

        // ── flashlib broad headline-win (vs_cuml/broad/kmeans.py) ──
        Shape { label: "N=1M_D=32_K=16",       n: 1_000_000,  d: 32,  k: 16,    max_iter: 3 },
        Shape { label: "N=1M_D=32_K=64",       n: 1_000_000,  d: 32,  k: 64,    max_iter: 3 },
        Shape { label: "N=3M_D=32_K=16",       n: 3_000_000,  d: 32,  k: 16,    max_iter: 3 },
        Shape { label: "N=3M_D=32_K=64",       n: 3_000_000,  d: 32,  k: 64,    max_iter: 3 },

        // ── flashlib broad moderate ──
        Shape { label: "N=10M_D=16_K=64",      n: 10_000_000, d: 16,  k: 64,    max_iter: 3 },
        Shape { label: "N=10M_D=32_K=16",      n: 10_000_000, d: 32,  k: 16,    max_iter: 3 },
        Shape { label: "N=10M_D=32_K=64",      n: 10_000_000, d: 32,  k: 64,    max_iter: 3 },
        Shape { label: "N=10M_D=64_K=64",      n: 10_000_000, d: 64,  k: 64,    max_iter: 3 },
        Shape { label: "N=10M_D=64_K=256",     n: 10_000_000, d: 64,  k: 256,   max_iter: 3 },

        // ── flashlib broad high-D ──
        Shape { label: "N=300K_D=128_K=1000",  n: 300_000,    d: 128, k: 1000,  max_iter: 3 },
        Shape { label: "N=1M_D=128_K=1000",    n: 1_000_000,  d: 128, k: 1000,  max_iter: 3 },
        Shape { label: "N=300K_D=256_K=1000",  n: 300_000,    d: 256, k: 1000,  max_iter: 3 },
    ];

    println!("\n  ╔══════════════════════════════════════════════════════════════╗");
    println!("  ║  KMeans fit — GPU (Metal) vs CPU (BLAS) vs sklearn          ║");
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!("  {:<24} {:>10} {:>10} {:>10} {:>10}", "shape", "metal(ms)", "blas-cpu(ms)", "sklearn(ms)", "speedup");

    let mut group = c.benchmark_group("kmeans_fit");
    group.sample_size(15);

    for s in shapes {
        let (data, centers) = generate_data(s.n, s.d, s.k, 42);
        let init: Vec<f32> = (0..s.k * s.d).map(|i| centers[i]).collect();

        let label = s.label;
        let n = s.n; let d = s.d; let k = s.k; let max_iter = s.max_iter;
        let init_for_metal = init.clone();
        group.bench_with_input(BenchmarkId::new(label, "metal"), &data, |b, data| {
            b.iter(|| {
                let mut km = KMeans::new(KMeansConfig { k, max_iterations: max_iter, tolerance: 1e-4, seed: 42, init_centroids: Some(init_for_metal.clone()) });
                km.fit(black_box(&ctx), black_box(data), black_box(n), black_box(d)).expect("metal fit failed");
            });
        });

        let init_for_blas = init.clone();
        group.bench_with_input(BenchmarkId::new(label, "blas-cpu"), &data, |b, data| {
            b.iter(|| {
                let mut c = init_for_blas.clone();
                cpu_kmeans_blas(black_box(data), black_box(n), black_box(d), black_box(k), black_box(max_iter), &mut c);
                black_box(());
            });
        });

        // Skip sklearn for shapes where piping 2GB+ through stdin is unreliable
        let nbytes = n * d * 4;
        let sklearn_ms = if nbytes > 250_000_000 {
            eprintln!("  sklearn {}: SKIPPED ({} MB data — pipe too large)", label, nbytes >> 20);
            0.0
        } else {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                time_sklearn_kmeans(&data, n, d, k, max_iter, 1e-4, Some(&init))
            }))
            .unwrap_or_else(|_| { eprintln!("  sklearn {}: FAILED", label); 0.0 })
        };
        if sklearn_ms > 0.0 {
            eprintln!("  sklearn {}: {:.3} ms", label, sklearn_ms);
        }
    }
    group.finish();
}

criterion_group!(benches, bench_kmeans_assign_vs_blas, bench_kmeans_fit);
criterion_main!(benches);
