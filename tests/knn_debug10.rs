use metal::*;
use metal_operators::metal::MetalContext;
use std::ptr;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

#[test]
fn test_knn_debug10() {
    let ctx = MetalContext::new().expect("Metal");

    let (nq, nc, d) = (2usize, 10usize, 3usize);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);

    let source = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void knn_test10(
        device const float* queries    [[buffer(0)]],
        device const float* corpus      [[buffer(1)]],
        device float* out               [[buffer(2)]],
        constant uint& nq               [[buffer(3)]],
        constant uint& nc               [[buffer(4)]],
        constant uint& d                [[buffer(5)]],
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;

        float sum = 0.0;
        for (uint c = 0; c < nc; c++) {
            for (uint dd = 0; dd < d; dd++) {
                sum += queries[qid * d + dd] * corpus[c * d + dd];
            }
        }
        out[qid] = sum;
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "knn_test10").expect("compile");

    let qbuf = ctx.new_buffer(&queries);
    let cbuf = ctx.new_buffer(&corpus);
    let out = ctx.new_buffer_uninitialized((nq * 4) as u64);

    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&qbuf), 0);
    enc.set_buffer(1, Some(&cbuf), 0);
    enc.set_buffer(2, Some(&out), 0);
    let nq_v = nq as u32; let nc_v = nc as u32; let d_v = d as u32;
    enc.set_bytes(3, 4, ptr::from_ref(&nq_v).cast());
    enc.set_bytes(4, 4, ptr::from_ref(&nc_v).cast());
    enc.set_bytes(5, 4, ptr::from_ref(&d_v).cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let vals: Vec<f32> = ctx.read_buffer(&out, nq);
    eprintln!("sums: {:?}", vals);

    // Expected sum for qid=0: sum over all corpus points of dot products
    let mut expected = 0.0f32;
    for c in 0..nc {
        for dd in 0..d {
            expected += queries[0*d+dd] * corpus[c*d+dd];
        }
    }
    eprintln!("expected: {}", expected);
    assert!((vals[0] - expected).abs() < 1e-2, "sum mismatch: {} vs {}", vals[0], expected);
}
