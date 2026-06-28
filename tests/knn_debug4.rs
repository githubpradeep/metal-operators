use metal::*;
use metal_operators::metal::MetalContext;

#[test]
fn test_knn_debug4() {
    let ctx = MetalContext::new().expect("Metal");

    let source = r#"
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void test_arr(
        device float* out [[buffer(0)]],
        constant uint& nq [[buffer(1)]],
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;
        
        float arr[4];
        arr[0] = 1.0f;
        arr[1] = 2.0f;
        arr[2] = 3.0f;
        arr[3] = 4.0f;
        
        // Write using array indexing
        out[qid * 4 + 0] = arr[0];
        out[qid * 4 + 1] = arr[1];
        out[qid * 4 + 2] = arr[2];
        out[qid * 4 + 3] = arr[3];
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "test_arr").expect("compile");
    let out = ctx.new_buffer_uninitialized((2 * 4 * 4) as u64);
    let nq: u32 = 2;
    
    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&out), 0);
    enc.set_bytes(1, 4, std::ptr::from_ref(&nq).cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let vals: Vec<f32> = ctx.read_buffer(&out, 8);
    eprintln!("arr test: {:?}", vals);
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[4], 1.0);
}
