use metal::*;
use metal_operators::metal::MetalContext;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

#[test]
fn test_knn_debug7() {
    let ctx = MetalContext::new().expect("Metal");

    let (nq, nc, d, k) = (2usize, 10usize, 3usize, 2usize);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);

    // Simplest possible loop — just count corpus points
    let source = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void knn_test7(
        device float* out               [[buffer(0)]],
        constant uint& nq               [[buffer(1)]],
        constant uint& nc               [[buffer(2)]],
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;

        // Just count how many corpus points we process
        uint count = 0;
        for (uint c = 0; c < nc; c++) {
            count++;
        }
        out[qid] = (float)count;
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "knn_test7").expect("compile");
    
    let out = ctx.new_buffer_uninitialized((nq * 4) as u64);

    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&out), 0);
    let params: [u32; 2] = [nq as u32, nc as u32];
    enc.set_bytes(1, 8, params.as_ptr().cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let vals: Vec<f32> = ctx.read_buffer(&out, nq);
    eprintln!("counts: {:?}", vals);
    assert_eq!(vals[0] as u32, nc as u32, "count should be {}", nc);
}
