use metal::*;
use metal_operators::metal::MetalContext;
use std::ptr;

#[test]
fn test_knn_debug8() {
    let ctx = MetalContext::new().expect("Metal");

    let source = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void knn_test8(
        device float* out               [[buffer(0)]],
        constant uint& nq               [[buffer(1)]],
        constant uint& nc               [[buffer(2)]],
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;

        uint count = 0;
        for (uint c = 0; c < nc; c++) {
            count++;
        }
        out[qid] = (float)count;
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "knn_test8").expect("compile");
    let out = ctx.new_buffer_uninitialized((2 * 4) as u64);

    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&out), 0);
    // INDIVIDUAL set_bytes — NOT using a single set_bytes for multiple params
    let nq_val: u32 = 2;
    let nc_val: u32 = 10;
    enc.set_bytes(1, 4, ptr::from_ref(&nq_val).cast());
    enc.set_bytes(2, 4, ptr::from_ref(&nc_val).cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let vals: Vec<f32> = ctx.read_buffer(&out, 2);
    eprintln!("counts: {:?}", vals);
    assert_eq!(vals[0] as u32, 10);
}
