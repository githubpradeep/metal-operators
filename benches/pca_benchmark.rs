// PCA benchmark — flashlib-style methodology.
//
// Compares:
//   - Metal GPU: mean→center→transpose→matmul(GPU) + eigh(CPU: Jacobi/Accelerate)
//   - CPU: same algorithm fully on CPU (naive Gram + Jacobi/Accelerate)
//   - sklearn: via Python subprocess
//
// Shapes from flashlib's vs_cuml/pca.py + broad/pca.py, scaled to Mac.

use metal_operators::metal::MetalContext;
use metal_operators::pca::{PCA, PCAConfig};
use std::io::Write;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// Accelerate LAPACK eigh
// ═══════════════════════════════════════════════════════════════════

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn ssyevd_(
        jobz: *const u8, uplo: *const u8, n: *const i32,
        a: *mut f32, lda: *const i32, w: *mut f32,
        work: *mut f32, lwork: *const i32,
        iwork: *mut i32, liwork: *const i32,
        info: *mut i32,
    );
}

fn eigh_accelerate(a: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
    let n = m as i32;
    let lda = n;
    let mut a_mat = a.to_vec();
    let mut w = vec![0.0f32; m];
    let mut info: i32 = 0;
    let jobz: u8 = b'V';
    let uplo: u8 = b'U';
    let mut lwork: i32 = -1;
    let mut liwork: i32 = -1;
    let mut work_size: f32 = 0.0;
    let mut iwork_size: i32 = 0;
    unsafe {
        ssyevd_(&jobz, &uplo, &n, a_mat.as_mut_ptr(), &lda, w.as_mut_ptr(),
                &mut work_size, &lwork, &mut iwork_size, &liwork, &mut info);
    }
    lwork = work_size as i32;
    liwork = iwork_size;
    let mut work = vec![0.0f32; lwork as usize];
    let mut iwork = vec![0i32; liwork as usize];
    unsafe {
        ssyevd_(&jobz, &uplo, &n, a_mat.as_mut_ptr(), &lda, w.as_mut_ptr(),
                work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &liwork, &mut info);
    }
    (w, a_mat)
}

// ═══════════════════════════════════════════════════════════════════
// CPU PCA reference
// ═══════════════════════════════════════════════════════════════════

fn pca_fit_cpu(data: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let means: Vec<f32> = (0..d)
        .map(|col| data.iter().skip(col).step_by(d).sum::<f32>() / n as f32)
        .collect();
    let centered: Vec<f32> = data.iter().enumerate()
        .map(|(i, &x)| x - means[i % d])
        .collect();

    let transposed = n < d;
    let gm = if transposed { n } else { d };

    let gram = if transposed {
        let mut g = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0f32;
                for kk in 0..d { s += centered[i * d + kk] * centered[j * d + kk]; }
                g[i * n + j] = s / n as f32;
                g[j * n + i] = s / n as f32;
            }
        }
        g
    } else {
        let mut g = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut s = 0.0f32;
                for kk in 0..n { s += centered[kk * d + i] * centered[kk * d + j]; }
                g[i * d + j] = s / n as f32;
                g[j * d + i] = s / n as f32;
            }
        }
        g
    };

    let (eigvals, eigvecs) = if gm <= 128 {
        // Jacobi
        let tol = 1e-8f32;
        let ms = 15;
        let mut a = gram;
        let mut v = vec![0.0f32; gm * gm];
        for i in 0..gm { v[i * gm + i] = 1.0; }
        for _ in 0..ms {
            let mut cv = true;
            for p in 0..gm {
                for q in (p + 1)..gm {
                    let apq = a[p * gm + q];
                    let app = a[p * gm + p];
                    let aqq = a[q * gm + q];
                    if apq.abs() <= tol * (app.abs() + aqq.abs()) * 0.5 { continue; }
                    cv = false;
                    let tau = (aqq - app) / (2.0 * apq);
                    let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) }
                            else { 1.0 / (tau - (1.0 + tau * tau).sqrt()) };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let an = c*c*app + s*s*aqq - 2.0*c*s*apq;
                    let aqn = s*s*app + c*c*aqq + 2.0*c*s*apq;
                    a[p*gm+p] = an; a[q*gm+q] = aqn; a[p*gm+q] = 0.0; a[q*gm+p] = 0.0;
                    for r in 0..gm {
                        if r != p && r != q {
                            let apr = a[p*gm+r]; let aqr = a[q*gm+r];
                            a[p*gm+r] = c*apr - s*aqr; a[r*gm+p] = a[p*gm+r];
                            a[q*gm+r] = s*apr + c*aqr; a[r*gm+q] = a[q*gm+r];
                        }
                        let vrp = v[r*gm+p]; let vrq = v[r*gm+q];
                        v[r*gm+p] = c*vrp - s*vrq; v[r*gm+q] = s*vrp + c*vrq;
                    }
                }
            }
            if cv { break; }
        }
        let mut ev = vec![0.0f32; gm];
        for i in 0..gm { ev[i] = a[i*gm+i]; }
        let mut idx: Vec<usize> = (0..gm).collect();
        idx.sort_by(|&i, &j| ev[i].partial_cmp(&ev[j]).unwrap());
        let sv: Vec<f32> = idx.iter().map(|&i| ev[i]).collect();
        let mut sv2 = vec![0.0f32; gm*gm];
        for (nc, &oc) in idx.iter().enumerate() {
            for r in 0..gm { sv2[r*gm+nc] = v[r*gm+oc]; }
        }
        (sv, sv2)
    } else {
        eigh_accelerate(&gram, gm)
    };

    let m = gm;
    let ka = k.min(m);
    let si = m - ka;
    let mut tv = vec![0.0f32; ka];
    let mut tvec = vec![0.0f32; d * ka];
    if transposed {
        for j in 0..ka {
            let sc = si + j;
            tv[j] = eigvals[sc];
            let is = 1.0 / (eigvals[sc].max(1e-30) * n as f32).sqrt();
            for i in 0..d {
                let mut s = 0.0;
                for r in 0..n { s += centered[r*d+i] * eigvecs[r*gm+sc]; }
                tvec[i*ka+j] = s * is;
            }
        }
    } else {
        for j in 0..ka {
            let sc = si + j;
            tv[j] = eigvals[sc];
            for i in 0..d { tvec[i*ka+j] = eigvecs[i*gm+sc]; }
        }
    }
    let mut comps = vec![0.0f32; ka * d];
    for j in 0..ka { let fj = ka-1-j; for i in 0..d { comps[j*d+i] = tvec[i*ka+fj]; } }
    comps
}

// ═══════════════════════════════════════════════════════════════════
// Data + sklearn
// ═══════════════════════════════════════════════════════════════════

fn gen_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n*d).map(|_| rng.f32() * 10.0 - 5.0).collect()
}

fn sklearn_ms(data: &[f32], n: usize, d: usize, k: usize) -> f64 {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benches").join("sklearn_pca.py");
    let bl = (n * d * 4) as i32;
    let db = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n * d * 4) };
    let mut inp = Vec::new();
    writeln!(inp, "{} {} {} {}", n, d, k, bl).unwrap();
    inp.extend_from_slice(db);
    let mut ch = std::process::Command::new("python3").arg(script.to_str().unwrap())
        .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null()).spawn().expect("python3?");
    ch.stdin.take().unwrap().write_all(&inp).unwrap();
    let o = ch.wait_with_output().unwrap();
    assert!(o.status.success(), "sklearn fail: {:?}", String::from_utf8_lossy(&o.stderr));
    let mut r = [0u8; 4]; r.copy_from_slice(&o.stdout[..4]);
    f32::from_le_bytes(r) as f64
}

fn time_one(label: &str, metal_ms: f64, cpu_ms: f64, sk_ms: f64) {
    let vc = if cpu_ms > 0.0 { metal_ms / cpu_ms } else { 0.0 };
    let vs = if sk_ms > 0.0 { metal_ms / sk_ms } else { 0.0 };
    let spd = format!("{:.1}x/CPU {:.1}x/skl", 1.0/vc.max(0.001), 1.0/vs.max(0.001));
    println!("  {:<28} {:>8.2} {:>8.2} {:>8.2}  {}", label, metal_ms, cpu_ms, sk_ms, spd);
}

// ═══════════════════════════════════════════════════════════════════
// Shapes — flashlib-inspired, scaled to Mac hardware
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let ctx = MetalContext::new().expect("No Metal device");

    // Warm up PCA once (compiles pipelines)
    let warm = gen_data(100, 10, 0);
    let mut p = PCA::new(PCAConfig { n_components: 2 });
    p.fit(&ctx, &warm, 100, 10).unwrap();
    pca_fit_cpu(&warm, 100, 10, 2);

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  PCA fit — Metal GPU vs CPU (Accelerate) vs sklearn (scikit-learn)    ║");
    println!("  ║  Shapes reference: flashlib vs_cuml/pca.py + broad/pca.py              ║");
    println!("  ╚══════════════════════════════════════════════════════════════════════════╝");
    println!("  {:<28} {:>8} {:>8} {:>8}  {}", "shape", "metal", "cpu", "sklearn", "speedup");

    // ── Standard regime (vs_cuml/pca.py) ──────────────────────────
    // tall:   N >> D → cov path: X^T @ X / N  (Gram dim = D)
    // wide:   D >> N → gram path: X @ X^T / N (Gram dim = N)

    let shapes: &[(&str, usize, usize, usize)] = &[
        // Standard (flashlib vs_cuml/pca.py)
        ("tall   N=100K D=128 K=32", 100_000, 128, 32),
        ("wide   N=2K   D=8K  K=32",   2_000, 8_000, 32),
        // Broad tall (flashlib broad/pca.py)
        ("tall   N=100K D=128 K=16", 100_000, 128, 16),
        ("tall   N=100K D=128 K=64", 100_000, 128, 64),
        // Broad square-ish
        ("sq     N=1K   D=512 K=32",   1_000, 512, 32),
        ("sq     N=5K   D=1K  K=32",   5_000, 1_024, 32),
        // Broad wide
        ("wide   N=500  D=4K  K=32",     500, 4_000, 32),
        ("wide   N=500  D=8K  K=32",     500, 8_000, 32),
        ("wide   N=500  D=16K K=32",     500, 16_000, 32),
    ];

    for &(label, n, d, k) in shapes {
        let data = gen_data(n, d, 42);

        // Warm up with shape
        let mut pw = PCA::new(PCAConfig { n_components: k });
        pw.fit(&ctx, &data, n, d).unwrap();
        pca_fit_cpu(&data, n, d, k);

        // GPU Metal
        let niter = if n * d > 1_000_000 { 3usize } else { 10 };
        let t0 = Instant::now();
        for _ in 0..niter {
            let mut pca = PCA::new(PCAConfig { n_components: k });
            pca.fit(&ctx, &data, n, d).unwrap();
        }
        let metal = t0.elapsed().as_secs_f64() * 1000.0 / niter as f64;

        // CPU
        let cpu = if n * d > 5_000_000 {
            0.0
        } else {
            let niter_c = if n * d > 500_000 { 3usize } else { 10 };
            let t0 = Instant::now();
            for _ in 0..niter_c { pca_fit_cpu(&data, n, d, k); }
            t0.elapsed().as_secs_f64() * 1000.0 / niter_c as f64
        };

        // sklearn
        let sk = if n * d * 4 > 200_000_000 {
            0.0
        } else {
            sklearn_ms(&data, n, d, k)
        };

        time_one(label, metal, cpu, sk);
    }

    println!();
    println!("  Notes:");
    println!("  - 'cpu = 0.00' means skipped (N*D > 5M, naive Gram loop is too slow)");
    println!("  - 'sklearn = 0.00' means skipped (data > 200 MB for pipe)");
    println!();
}
