use metal::*;
use metal_operators::metal::MetalContext;

const SHADER: &str = "\
#include <metal_stdlib>
using namespace metal;

kernel void test_multisg(
    device const float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    simdgroup_float8x8 mat;
    simdgroup_load(mat, inp + simd_gid * 64, 8, 0, 0);
    simdgroup_store(mat, out + simd_gid * 64, 8, 0, 0);
}
";

#[test]
fn test_multisg_device_read() {
    let ctx = MetalContext::new().expect("metal");

    let lib = ctx.device.new_library_with_source(SHADER, &CompileOptions::new())
        .expect("compile test shader");
    let func = lib.get_function("test_multisg", None)
        .expect("function not found");
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func)
        .expect("pipeline");

    let mut inp = vec![0.0f32; 256];
    for t in 0..4 {
        for i in 0..8 {
            for j in 0..8 {
                inp[t * 64 + i * 8 + j] = (t * 100 + i * 10 + j) as f32;
            }
        }
    }

    let buf_inp = ctx.new_buffer(&inp);
    let buf_out = ctx.new_buffer_uninitialized(256 * 4);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_inp), 0);
    enc.set_buffer(1, Some(&buf_out), 0);
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 },
                               MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd.commit();
    let start = std::time::Instant::now();
    cmd.wait_until_completed();
    let elapsed = start.elapsed();

    let out: Vec<f32> = ctx.read_buffer(&buf_out, 256);
    for sg in 0..4 {
        let base = sg * 64;
        for i in 0..8 {
            for j in 0..8 {
                let got = out[base + i * 8 + j];
                let exp = inp[sg * 64 + i * 8 + j];
                assert!((got - exp).abs() < 1e-4,
                    "sg={} [{}][{}]: got {:.0} expected {:.0}", sg, i, j, got, exp);
            }
        }
    }
    eprintln!("PASS: 4 simdgroups loaded from device memory in {:?}", elapsed);
}

#[test]
fn test_multisg_shared() {
    let ctx = MetalContext::new().expect("metal");

    // Same kernel but using threadgroup memory
    let shader = "\
#include <metal_stdlib>
using namespace metal;

kernel void test_multisg_shared(
    device float* out [[buffer(0)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    // Fill shared memory: each thread fills 2 elements
    uint n = lid * 2;
    sh[n] = (float)(n);
    sh[n + 1] = (float)(n + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup loads a different 8×8 tile from shared memory
    // sg0: rows 0-7, stride 8 (sh[0..63])
    // sg1: rows 0-7, from sh[64..127]
    // sg2: rows 0-7, from sh[128..191]
    // sg3: rows 0-7, from sh[192..255]
    simdgroup_float8x8 mat;
    simdgroup_load(mat, sh + simd_gid * 64, 8, 0, 0);
    simdgroup_store(mat, out + simd_gid * 64, 8, 0, 0);
}
";

    let lib = ctx.device.new_library_with_source(shader, &CompileOptions::new())
        .expect("compile");
    let func = lib.get_function("test_multisg_shared", None)
        .expect("function");
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func)
        .expect("pipeline");

    let buf_out = ctx.new_buffer_uninitialized(256 * 4);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_out), 0);
    enc.set_threadgroup_memory_length(0, 256 * 4);
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 },
                               MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out: Vec<f32> = ctx.read_buffer(&buf_out, 256);

    // Check each simdgroup loaded correctly from its portion of shared memory
    for sg in 0..4 {
        let base = sg * 64;
        for i in 0..8 {
            for j in 0..8 {
                let got = out[base + i * 8 + j];
                let exp = (sg * 64 + i * 8 + j) as f32;
                assert!((got - exp).abs() < 1e-4,
                    "sg={} [{}][{}]: got {:.0} expected {:.0}", sg, i, j, got, exp);
            }
        }
    }
    eprintln!("PASS: 4 simdgroups loaded from shared memory");
}
