use metal::*;
use metal_operators::metal::MetalContext;

const SHADER_SRC: &str = include_str!("../shaders/kmeans.metal");

fn compile_pipeline(ctx: &MetalContext, name: &str) -> ComputePipelineState {
    let lib = ctx.device.new_library_with_source(SHADER_SRC, &CompileOptions::new()).unwrap();
    let func = lib.get_function(name, None).unwrap();
    ctx.device.new_compute_pipeline_state_with_function(&func).unwrap()
}

fn set_uint(enc: &ComputeCommandEncoderRef, idx: u64, val: u32) {
    let v = val;
    enc.set_bytes(idx, 4, &v as *const u32 as *const std::ffi::c_void);
}

fn generate_blobs(n: usize, d: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
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

fn cpu_assign(data: &[f32], centroids: &[f32], n: usize, d: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut labels = vec![0u32; n];
    let mut dists = vec![0.0f32; n];
    for i in 0..n {
        let mut best_d = f32::INFINITY;
        let mut best = 0u32;
        for j in 0..k {
            let mut d2 = 0.0;
            for dim in 0..d {
                let diff = data[i*d+dim] - centroids[j*d+dim];
                d2 += diff * diff;
            }
            if d2 < best_d { best_d = d2; best = j as u32; }
        }
        labels[i] = best;
        dists[i] = best_d;
    }
    (labels, dists)
}

#[test]
fn test_splitd_small_d() {
    let ctx = MetalContext::new().expect("metal");
    let pipeline = compile_pipeline(&ctx, "kmeans_assign_splitd");

    let (n, d, k) = (128usize, 2usize, 4usize);

    let centroids: Vec<f32> = vec![1.0, 2.0, 10.0, 20.0, -5.0, 5.0, 0.0, 0.0];
    let data: Vec<f32> = (0..n * d)
        .map(|i| {
            let pt = i / d;
            let cluster = pt % k;
            let base = cluster * d;
            centroids[base + i % d] + (pt as f32 * 0.1 - 0.5) * 0.01
        })
        .collect();

    let norms_x: Vec<f32> = data.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();
    let norms_c: Vec<f32> = centroids.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();

    let buf_pts = ctx.new_buffer(&data);
    let buf_ct = ctx.new_buffer(&centroids);
    let buf_nx = ctx.new_buffer(&norms_x);
    let buf_nc = ctx.new_buffer(&norms_c);
    let buf_assign = ctx.new_buffer_uninitialized((n as u64) * 4);
    let buf_dist = ctx.new_buffer_uninitialized((n as u64) * 4);

    let groups = MTLSize { width: 1, height: 1, depth: 1 }; // 1 group for 128 points
    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_pts), 0);
    enc.set_buffer(1, Some(&buf_ct), 0);
    enc.set_buffer(2, Some(&buf_assign), 0);
    enc.set_buffer(3, Some(&buf_dist), 0);
    enc.set_buffer(4, Some(&buf_nx), 0);
    enc.set_buffer(5, Some(&buf_nc), 0);
    set_uint(&enc, 6, n as u32);
    set_uint(&enc, 7, k as u32);
    set_uint(&enc, 8, d as u32);
    set_uint(&enc, 9, 128);

    let shared_bytes = 256u64 * 4; // BD * CTILE = 32 * 8
    enc.set_threadgroup_memory_length(0, shared_bytes);
    enc.dispatch_thread_groups(groups, threads);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let labels: Vec<u32> = ctx.read_buffer(&buf_assign, n);
    let distances: Vec<f32> = ctx.read_buffer(&buf_dist, n);

    let (cpu_labels, cpu_dists) = cpu_assign(&data, &centroids, n, d, k);

    for i in 0..n {
        assert_eq!(labels[i], cpu_labels[i],
            "Point {}: metal label={} dist={} cpu label={} dist={}",
            i, labels[i], distances[i], cpu_labels[i], cpu_dists[i]);
    }
}

#[test]
fn test_splitd_large_d() {
    let ctx = MetalContext::new().expect("metal");
    let pipeline = compile_pipeline(&ctx, "kmeans_assign_splitd");

    let (n, d, k) = (256usize, 320usize, 8usize);
    let (data, centroids) = generate_blobs(n, d, k, 42);

    let norms_x: Vec<f32> = data.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();
    let norms_c: Vec<f32> = centroids.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();

    let buf_pts = ctx.new_buffer(&data);
    let buf_ct = ctx.new_buffer(&centroids);
    let buf_nx = ctx.new_buffer(&norms_x);
    let buf_nc = ctx.new_buffer(&norms_c);
    let buf_assign = ctx.new_buffer_uninitialized((n as u64) * 4);
    let buf_dist = ctx.new_buffer_uninitialized((n as u64) * 4);

    let groups = MTLSize { width: 2, height: 1, depth: 1 }; // ceil(256/128)
    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_pts), 0);
    enc.set_buffer(1, Some(&buf_ct), 0);
    enc.set_buffer(2, Some(&buf_assign), 0);
    enc.set_buffer(3, Some(&buf_dist), 0);
    enc.set_buffer(4, Some(&buf_nx), 0);
    enc.set_buffer(5, Some(&buf_nc), 0);
    set_uint(&enc, 6, n as u32);
    set_uint(&enc, 7, k as u32);
    set_uint(&enc, 8, d as u32);
    set_uint(&enc, 9, 128);

    let shared_bytes = 256u64 * 4;
    enc.set_threadgroup_memory_length(0, shared_bytes);
    enc.dispatch_thread_groups(groups, threads);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let labels: Vec<u32> = ctx.read_buffer(&buf_assign, n);
    let distances: Vec<f32> = ctx.read_buffer(&buf_dist, n);

    let (cpu_labels, cpu_dists) = cpu_assign(&data, &centroids, n, d, k);

    for i in 0..n {
        assert_eq!(labels[i], cpu_labels[i],
            "Point {}: metal label={} dist={:.4} cpu label={} dist={:.4}",
            i, labels[i], distances[i], cpu_labels[i], cpu_dists[i]);
    }
}
