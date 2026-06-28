use metal::CompileOptions;
use metal_operators::metal::MetalContext;
use std::time::Instant;

fn time_kernel(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    points: &metal::Buffer,
    centroids: &metal::Buffer,
    assignments: &metal::Buffer,
    min_dists: &metal::Buffer,
    norms_x: &metal::Buffer,
    norms_c: &metal::Buffer,
    n: u32,
    k: u32,
    d: u32,
    tg_size: u32,
    shared_size: u64,
    iterations: usize,
) -> f64 {
    let tg_mtl = metal::MTLSize { width: tg_size as u64, height: 1, depth: 1 };
    let groups = metal::MTLSize {
        width: ((n as u64) + tg_size as u64 - 1) / tg_size as u64,
        height: 1,
        depth: 1,
    };

    for _ in 0..3 {
        let cmd_buffer = ctx.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(points), 0);
        encoder.set_buffer(1, Some(centroids), 0);
        encoder.set_buffer(2, Some(assignments), 0);
        encoder.set_buffer(3, Some(min_dists), 0);
        encoder.set_buffer(4, Some(norms_x), 0);
        encoder.set_buffer(5, Some(norms_c), 0);
        encoder.set_bytes(6, 4, std::ptr::from_ref(&n).cast());
        encoder.set_bytes(7, 4, std::ptr::from_ref(&k).cast());
        encoder.set_bytes(8, 4, std::ptr::from_ref(&d).cast());
        encoder.set_bytes(9, 4, std::ptr::from_ref(&tg_size).cast());
        if shared_size > 0 { encoder.set_threadgroup_memory_length(0, shared_size); }
        encoder.dispatch_thread_groups(groups, tg_mtl);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let cmd_buffer = ctx.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(points), 0);
        encoder.set_buffer(1, Some(centroids), 0);
        encoder.set_buffer(2, Some(assignments), 0);
        encoder.set_buffer(3, Some(min_dists), 0);
        encoder.set_buffer(4, Some(norms_x), 0);
        encoder.set_buffer(5, Some(norms_c), 0);
        encoder.set_bytes(6, 4, std::ptr::from_ref(&n).cast());
        encoder.set_bytes(7, 4, std::ptr::from_ref(&k).cast());
        encoder.set_bytes(8, 4, std::ptr::from_ref(&d).cast());
        encoder.set_bytes(9, 4, std::ptr::from_ref(&tg_size).cast());
        if shared_size > 0 { encoder.set_threadgroup_memory_length(0, shared_size); }
        encoder.dispatch_thread_groups(groups, tg_mtl);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

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

fn compute_norms(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n).map(|i| {
        let base = i * d;
        let mut s = 0.0;
        for j in 0..d { s += x[base + j] * x[base + j]; }
        s
    }).collect()
}

fn compile_kernel(ctx: &MetalContext, src: &str, kernel_name: &str) -> metal::ComputePipelineState {
    let options = CompileOptions::new();
    let library = ctx.device.new_library_with_source(src, &options)
        .map_err(|e| format!("Shader compile failed: {}", e)).unwrap();
    let function = library.get_function(kernel_name, None)
        .map_err(|e| format!("Fn {} not found: {}", kernel_name, e)).unwrap();
    ctx.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

#[test]
fn test_compare_ptile() {
    let ctx = MetalContext::new().expect("No Metal device");
    let lib_src = include_str!("../shaders/kmeans_profile.metal");

    #[rustfmt::skip]
    let shapes: &[(usize, usize, usize, &str)] = &[
        (10_000,    32,   16, "N=10K_D=32_K=16"),
        (10_000,    64,   16, "N=10K_D=64_K=16"),
        (10_000,    128,  32, "N=10K_D=128_K=32"),
        (100_000,   32,  256, "N=100K_D=32_K=256"),
        (1_000_000, 32,   16, "N=1M_D=32_K=16"),
        (3_000_000, 32,   16, "N=3M_D=32_K=16"),
        (10_000_000,32,   16, "N=10M_D=32_K=16"),
    ];

    eprintln!("\n  PTILE/CTILE comparison (release):");
    eprintln!("  {:<26} {:>9} {:>9} {:>9} {:>8} {:>8}", "shape", "P8 (ms)", "P16 (ms)", "C16 (ms)", "P8/C16", "P16/C16");
    eprintln!("  {}", "-".repeat(70));

    for &(n, d, k, label) in shapes {
        if d > 128 { continue; } // PTILE=16 may not fit shared memory for larger D

        let (data, centers) = generate_data(n, d, k, 42);
        let norms_x = compute_norms(&data, n, d);
        let norms_c = compute_norms(&centers, k, d);
        let buf_points = ctx.new_buffer(&data);
        let buf_cent = ctx.new_buffer(&centers);
        let buf_assign = ctx.new_buffer_uninitialized((n * 4) as u64);
        let buf_dists = ctx.new_buffer_uninitialized((n * 4) as u64);
        let buf_nx = ctx.new_buffer(&norms_x);
        let buf_nc = ctx.new_buffer(&norms_c);
        let iterations = if n <= 100_000 { 100 } else { if n <= 1_000_000 { 20 } else { 5 } };

        // PTILE=8
        const CTILE: u32 = 8;
        let num_tiles = (d as u32 + CTILE - 1) / CTILE;
        let shared_p8 = (8 * d as u32 + d as u32 * CTILE + num_tiles * CTILE * CTILE + 8 * 2) as u64 * 4;
        let pipeline_p8 = compile_kernel(&ctx, lib_src, "kmeans_assign_profile");
        let t_p8 = time_kernel(
            &ctx, &pipeline_p8, &buf_points, &buf_cent, &buf_assign, &buf_dists,
            &buf_nx, &buf_nc, n as u32, k as u32, d as u32, 128, shared_p8, iterations,
        );

        // PTILE=16
        let sh16 = (16 * d as u32 + d as u32 * 8 + num_tiles * 16 * 8 + 16 * 2) as u64 * 4;
        let pipeline_p16 = compile_kernel(&ctx, lib_src, "kmeans_assign_p16");
        let t_p16 = time_kernel(
            &ctx, &pipeline_p16, &buf_points, &buf_cent, &buf_assign, &buf_dists,
            &buf_nx, &buf_nc, n as u32, k as u32, d as u32, 128, sh16, iterations,
        );

        // CTILE=16
        let dim_tiles = (d as u32 + 7) / 8;
        let sh_c16 = (8 * d as u32 + d as u32 * 16 + dim_tiles * 8 * 16 + 8 * 2) as u64 * 4;
        let pipeline_c16 = compile_kernel(&ctx, lib_src, "kmeans_assign_c16");
        let t_c16 = time_kernel(
            &ctx, &pipeline_c16, &buf_points, &buf_cent, &buf_assign, &buf_dists,
            &buf_nx, &buf_nc, n as u32, k as u32, d as u32, 128, sh_c16, iterations,
        );

        eprintln!("  {:<26} {:>9.3}  {:>9.3}  {:>9.3}  {:>7.2}x  {:>7.2}x",
            label, t_p8, t_p16, t_c16, t_p8 / t_c16, t_p16 / t_c16);
    }
}
