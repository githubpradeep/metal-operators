// Test: swap centroid layout so sg 0 processes centroids 8..15 and sg 1 processes 0..7
// If the bug follows sg 1 (swap), it's a simdgroup issue.
// If the bug stays with columns 8..15, it's a memory/layout issue.
use metal::*;
use metal_operators::metal::MetalContext;
use std::ffi::c_void;

#[test]
fn test_simdgroup_swap_centroids() {
    let ctx = MetalContext::new().expect("metal");

    // Same debug kernel but we'll swap which centroids go to which columns
    let debug_src = "
#include <metal_stdlib>
using namespace metal;

kernel void debug_swap(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device float* out_dots [[buffer(2)]],
    device const float* norms_X [[buffer(3)]],
    device const float* norms_C [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    constant uint& tg_size [[buffer(8)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint PTILE = 8;
    constexpr uint CTILE = 32;

    uint p_start = gid * PTILE;
    if (p_start >= n) return;

    threadgroup float* sh_pts  = shared;
    threadgroup float* sh_cent = shared + PTILE * d;
    threadgroup float* sh_dots = shared + PTILE * d + d * CTILE;

    uint total_pt = PTILE * d;
    for (uint i = lid; i < total_pt; i += tg_size) {
        uint po = i / d, pd = i % d;
        uint gi = p_start + po;
        sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
    }

    // Loop over centroid tiles
    for (uint c_base = 0; c_base < k; c_base += CTILE) {
        uint c_tile = min(CTILE, k - c_base);
        uint total_load = c_tile * d;
        for (uint i = lid; i < total_load; i += tg_size) {
            uint co = i % c_tile;
            uint dim = i / c_tile;
            sh_cent[dim * CTILE + co] = centroids[(c_base + co) * d + dim];
        }
        // zero unused columns (16..31)
        uint zero_cnt = d * (CTILE - c_tile);
        for (uint i = lid; i < zero_cnt; i += tg_size) {
            uint dd = i / (CTILE - c_tile);
            uint co = c_tile + i % (CTILE - c_tile);
            sh_cent[dd * CTILE + co] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint c_sg = simd_gid * 8;
        if (c_sg < CTILE) {
            simdgroup_float8x8 acc = {};
            for (uint dd = 0; dd < d; dd += 8) {
                simdgroup_float8x8 A, B;
                simdgroup_load(A, sh_pts, d, dd, 0);
                simdgroup_load(B, sh_cent, CTILE, c_sg, dd);
                simdgroup_multiply(acc, A, B);
            }
            simdgroup_store(acc, sh_dots, CTILE, c_sg, 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dump: for each point, store dot with each centroid
        if (lid < PTILE) {
            for (uint c = 0; c < c_tile; c++) {
                out_dots[(p_start + lid) * k + (c_base + c)] = sh_dots[lid * CTILE + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
";
    let debug_lib = ctx.device.new_library_with_source(debug_src, &CompileOptions::new()).unwrap();
    let func = debug_lib.get_function("debug_swap", None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();

    let (n, d, k) = (8usize, 8usize, 16usize);

    // SWAPPED centroids: put centroids 0..7 in columns 8..15 and centroids 8..15 in columns 0..7
    // We do this by storing centroids as: centroid[0] = true centroid 8, centroid[1] = true centroid 9, etc.
    let mut centroids = vec![0.0f32; k * d];
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        centroids[i * d] = angle.cos() * 5.0 + if i >= 8 { 100.0 } else { 0.0 };
        centroids[i * d + 1] = angle.sin() * 5.0;
    }

    // Build a REORDERED centroid array where:
    // centroids_in[0..7] = actual centroids 8..15 (far side)
    // centroids_in[8..15] = actual centroids 0..7 (origin side)
    let mut centroids_swapped = vec![0.0f32; k * d];
    for i in 0..k {
        let src = if i < 8 { i + 8 } else { i - 8 };
        for dim in 0..d {
            centroids_swapped[i * d + dim] = centroids[src * d + dim];
        }
    }

    // Points unchanged (same order)
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % k;
        data[i * d] = centroids[cluster * d];
        data[i * d + 1] = centroids[cluster * d + 1];
        for dim in 2..d {
            data[i * d + dim] = (i as f32 * 0.1 + dim as f32 * 0.01).fract() - 0.5;
        }
    }

    // norms_C based on SWAPPED centroids (since kernel reads norms_C[c_base + c])
    // For c_base=0: kernel accesses norms_C[0..15] which should be the norm of the centroid at position [c] in the swapped array.
    let norms_c_swapped: Vec<f32> = centroids_swapped.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();
    let norms_x: Vec<f32> = data.chunks(d).map(|r| r.iter().map(|x| x*x).sum()).collect();

    eprintln!("=== SWAPPED centroids layout ===");
    eprintln!("centroids_swapped[0..7] = actual centroids 8..15 (far side)");
    eprintln!("centroids_swapped[8..15] = actual centroids 0..7 (origin side)");
    eprintln!("norms_c_swapped = {:?}", &norms_c_swapped);

    let dots_buf = ctx.device.new_buffer((n * k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_pts = ctx.new_buffer(&data);
    let buf_ct = ctx.new_buffer(&centroids_swapped);
    let buf_nx = ctx.new_buffer(&norms_x);
    let buf_nc = ctx.new_buffer(&norms_c_swapped);

    let groups = MTLSize { width: 1, height: 1, depth: 1 };
    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_pts), 0);
    enc.set_buffer(1, Some(&buf_ct), 0);
    enc.set_buffer(2, Some(&dots_buf), 0);
    enc.set_buffer(3, Some(&buf_nx), 0);
    enc.set_buffer(4, Some(&buf_nc), 0);
    let set_uint = |e: &ComputeCommandEncoderRef, i: u64, v: u32| e.set_bytes(i, 4, &v as *const u32 as *const c_void);
    set_uint(&enc, 5, n as u32);
    set_uint(&enc, 6, k as u32);
    set_uint(&enc, 7, d as u32);
    set_uint(&enc, 8, 128);

    let shared_bytes = ((8 + 32) * d + 8 * 32 + 8 + 8) as u64 * 4;
    enc.set_threadgroup_memory_length(0, shared_bytes);
    enc.dispatch_thread_groups(groups, threads);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let dots: Vec<f32> = ctx.read_buffer(&dots_buf, n * k);

    eprintln!("\n=== Dots from GPU (swapped layout) ===");
    for i in 0..n {
        eprintln!("Point {} (true cluster {}):", i, i);
        for c in 0..k {
            let dot = dots[i * k + c];
            let dist = norms_x[i] + norms_c_swapped[c] - 2.0 * dot;
            eprintln!("  swapped_pos[{}] (actual centroid {}) dot={:.4} dist={:.4}", c, if c < 8 { c+8 } else { c-8 }, dot, dist);
        }
    }

    // CPU reference using original (non-swapped) centroids
    eprintln!("\n=== CPU reference ===");
    for i in 0..n {
        let mut best_d = f32::INFINITY;
        let mut best = 0u32;
        for c in 0..k {
            let mut d2 = 0.0f32;
            for dim in 0..d {
                let diff = data[i*d+dim] - centroids[c*d+dim];
                d2 += diff * diff;
            }
            eprintln!("  actual_centroid[{}] dist={:.4}", c, d2);
            if d2 < best_d { best_d = d2; best = c as u32; }
        }
        eprintln!("  => actual label={}", best);
    }
}
