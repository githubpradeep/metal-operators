// Reproduce the exact failing scenario from test_simdgroup_multitile
// but call the kernel directly (bypass KMeans fit())
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

#[test]
fn test_simdgroup_direct_16() {
    let ctx = MetalContext::new().expect("metal");
    let pipeline = compile_pipeline(&ctx, "kmeans_assign_simdgroup");

    let (n, d, k) = (16usize, 8usize, 16usize);

    // Same data as test_simdgroup_multitile
    let mut centroids = vec![0.0f32; k * d];
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        centroids[i * d] = angle.cos() * 5.0 + if i >= 8 { 100.0 } else { 0.0 };
        centroids[i * d + 1] = angle.sin() * 5.0;
    }

    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % k;
        data[i * d] = centroids[cluster * d];
        data[i * d + 1] = centroids[cluster * d + 1];
        for dim in 2..d {
            data[i * d + dim] = (i as f32 * 0.1 + dim as f32 * 0.01).fract() - 0.5;
        }
    }

    let norms_x: Vec<f32> = data.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();
    let norms_c: Vec<f32> = centroids.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();

    eprintln!("norms_x: {:?}", &norms_x);
    eprintln!("norms_c: {:?}", &norms_c);

    let buf_pts = ctx.new_buffer(&data);
    let buf_ct = ctx.new_buffer(&centroids);
    let buf_nx = ctx.new_buffer(&norms_x);
    let buf_nc = ctx.new_buffer(&norms_c);
    let buf_assign = ctx.new_buffer_uninitialized((n as u64) * 4);
    let buf_dist = ctx.new_buffer_uninitialized((n as u64) * 4);

    let groups = MTLSize { width: 2, height: 1, depth: 1 }; // 2 threadgroups for 16 points
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

    let shared_bytes = ((8 + 32) * d + 8 * 32 + 8 + 8) as u64 * 4;
    enc.set_threadgroup_memory_length(0, shared_bytes);
    enc.dispatch_thread_groups(groups, threads);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let labels: Vec<u32> = ctx.read_buffer(&buf_assign, n);
    let distances: Vec<f32> = ctx.read_buffer(&buf_dist, n);

    eprintln!("Labels: {:?}", &labels);
    eprintln!("Distances: {:?}", &distances);

    // CPU reference
    let mut cpu_labels = vec![0u32; n];
    let mut cpu_dists = vec![0.0f32; n];
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
        cpu_labels[i] = best;
        cpu_dists[i] = best_d;
    }
    eprintln!("CPU labels: {:?}", &cpu_labels);
    eprintln!("CPU distances: {:?}", &cpu_dists);

    for i in 0..n {
        assert_eq!(labels[i], cpu_labels[i],
            "Point {}: metal label={} dist={} cpu label={} dist={}",
            i, labels[i], distances[i], cpu_labels[i], cpu_dists[i]);
    }
    eprintln!("ALL CORRECT");
}
