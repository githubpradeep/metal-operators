use metal::*;
use metal_operators::metal::MetalContext;
use std::ffi::c_void;

#[test]
fn test_row_first_layout() {
    let ctx = MetalContext::new().expect("metal");

    let src = "
#include <metal_stdlib>
using namespace metal;

kernel void
test_kernel(device const float *points    [[buffer(0)]],
            device const float *centroids [[buffer(1)]],
            device float       *out_dd0   [[buffer(2)]],
            device float       *out_dd1   [[buffer(3)]],
            device float       *out_dd2   [[buffer(4)]],
            device float       *out_dd3   [[buffer(5)]],
            device float       *out_sum   [[buffer(6)]],
            constant uint      &n         [[buffer(7)]],
            constant uint      &k_all     [[buffer(8)]],
            constant uint      &d         [[buffer(9)]],
            constant uint      &tg_size   [[buffer(10)]],
            threadgroup float  *shared    [[threadgroup(0)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_index_in_threadgroup]],
            uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    constexpr uint PTILE = 8;
    constexpr uint CTILE = 8;

    uint p_start = gid * PTILE;
    if (p_start >= n) return;

    threadgroup float *sh_pts  = shared;
    threadgroup float *sh_cent = shared + PTILE * d;
    // Store each tile's result at different offset within sh_dots
    uint max_tiles = (d + CTILE - 1) / CTILE;
    threadgroup float *sh_dots = shared + PTILE * d + d * CTILE;

    // Load points standard: sh_pts[po * d + pd]
    for (uint i = lid; i < PTILE * d; i += tg_size) {
        uint po = i / d;
        uint pd = i % d;
        uint gi = p_start + po;
        sh_pts[po * d + pd] = (gi < n) ? points[gi * d + pd] : 0.0f;
    }

    // Load centroids transposed: sh_cent[dim * CTILE + co]
    for (uint i = lid; i < d * CTILE; i += tg_size) {
        uint co = i % CTILE;
        uint dim = i / CTILE;
        sh_cent[dim * CTILE + co] = centroids[co * d + dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        uint num_tiles = (d + CTILE - 1) / CTILE;

        for (uint dd = 0; dd < num_tiles; dd++) {
            simdgroup_float8x8 A, B, tile = {};
            simdgroup_load(A, sh_pts + dd * CTILE, d, 0, 0);
            simdgroup_load(B, sh_cent + dd * CTILE * CTILE, CTILE, 0, 0);
            // tile = A * B (replaces, not accumulate on this GPU)
            simdgroup_multiply(tile, A, B);
            // Store tile to its slot in shared memory
            simdgroup_store(tile, sh_dots + dd * CTILE * CTILE, CTILE, 0, 0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread sums tiles for its point
        if (lid < PTILE) {
            for (uint c = 0; c < CTILE; c++) {
                float sum = 0.0f;
                for (uint dd = 0; dd < num_tiles; dd++) {
                    sum += sh_dots[dd * CTILE * CTILE + lid * CTILE + c];
                }
                out_sum[(p_start + lid) * k_all + c] = sum;
            }
        }
    }
}
";
    let lib = ctx.device.new_library_with_source(src, &CompileOptions::new()).unwrap();
    let func = lib.get_function("test_kernel", None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();

    let (n, d, k) = (8usize, 32usize, 8usize);

    let mut rng = fastrand::Rng::with_seed(42);
    let mut centers = vec![0.0f32; k * d];
    for i in 0..k {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / k as f32;
        for dim in 0..d {
            centers[i * d + dim] = if dim == 0 { angle.cos() * 5.0 }
                else if dim == 1 { angle.sin() * 5.0 }
                else { rng.f32() * 4.0 - 2.0 };
        }
    }
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        for dim in 0..d {
            data[i * d + dim] = centers[i * d + dim] + (rng.f32() - 0.5) * 1.5;
        }
    }

    let num_tiles = d / k;
    let mut expected_tiles = vec![vec![0.0f32; n * k]; num_tiles];
    let mut expected_total = vec![0.0f32; n * k];
    for p in 0..n {
        for c in 0..k {
            for tile in 0..num_tiles {
                let mut dot = 0.0;
                for dim in tile*8..(tile+1)*8 {
                    dot += data[p * d + dim] * centers[c * d + dim];
                }
                expected_tiles[tile][p * k + c] = dot;
            }
            let mut total = 0.0;
            for dim in 0..d {
                total += data[p * d + dim] * centers[c * d + dim];
            }
            expected_total[p * k + c] = total;
        }
    }

    let b_dd = (0..num_tiles).map(|_| ctx.new_buffer_uninitialized((n * k * 4) as u64)).collect::<Vec<_>>();
    let b_sum = ctx.new_buffer_uninitialized((n * k * 4) as u64);
    let pbuf = ctx.new_buffer(&data);
    let cbuf = ctx.new_buffer(&centers);

    const PTILE: usize = 8;
    const CTILE: usize = 8;
    let groups = MTLSize { width: 1, height: 1, depth: 1 };
    let threads = MTLSize { width: 128, height: 1, depth: 1 };
    let max_tiles = (d + CTILE - 1) / CTILE;
    let shared = (PTILE * d + d * CTILE + max_tiles * CTILE * CTILE) as u64 * 4;
    let set_uint = |e: &ComputeCommandEncoderRef, i: u64, v: u32| e.set_bytes(i, 4, &v as *const u32 as *const c_void);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&pbuf), 0);
    enc.set_buffer(1, Some(&cbuf), 0);
    for (i, b) in b_dd.iter().enumerate() {
        enc.set_buffer((2 + i) as u64, Some(b), 0);
    }
    enc.set_buffer(6, Some(&b_sum), 0);
    set_uint(&enc, 7, n as u32);
    set_uint(&enc, 8, k as u32);
    set_uint(&enc, 9, d as u32);
    set_uint(&enc, 10, 128);
    enc.set_threadgroup_memory_length(0, shared);
    enc.dispatch_thread_groups(groups, threads);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    for tile in 0..num_tiles {
        let out: Vec<f32> = ctx.read_buffer(&b_dd[tile], n * k);
        eprintln!("\nTile dd={} (dims {}..{}):", tile, tile*8, (tile+1)*8-1);
        let mut ok = 0;
        for p in 0..n {
            for c in 0..k {
                let got = out[p * k + c];
                let exp = expected_tiles[tile][p * k + c];
                let flag = if (got - exp).abs() < 0.1 { "OK" } else { "WRONG" };
                if flag == "OK" { ok += 1; }
                if flag != "OK" {
                    eprintln!("  p{} c{}: got={:.4} exp={:.4} {}", p, c, got, exp, flag);
                }
            }
        }
        eprintln!("Tile {} OK: {}/{}", tile, ok, n * k);
    }

    let final_out: Vec<f32> = ctx.read_buffer(&b_sum, n * k);
    eprintln!("\nFinal accumulated:");
    let mut ok = 0;
    for p in 0..n {
        for c in 0..k {
            let got = final_out[p * k + c];
            let exp = expected_total[p * k + c];
            let flag = if (got - exp).abs() < 0.1 { "OK" } else { "WRONG" };
            if flag == "OK" { ok += 1; }
            if flag != "OK" {
                eprintln!("  p{} c{}: got={:.4} exp={:.4} {}", p, c, got, exp, flag);
            }
        }
    }
    eprintln!("Final OK: {}/{}", ok, n * k);
}
