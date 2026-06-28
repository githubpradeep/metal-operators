use metal::*;
use metal_operators::metal::MetalContext;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

#[test]
fn test_knn_debug6() {
    let ctx = MetalContext::new().expect("Metal");

    let (nq, nc, d, k) = (2usize, 10usize, 3usize, 2usize);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);

    let source = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void knn_test6(
        device const float* queries    [[buffer(0)]],
        device const float* corpus      [[buffer(1)]],
        device float* out_scores        [[buffer(2)]],
        constant uint& nq               [[buffer(3)]],
        constant uint& nc               [[buffer(4)]],
        constant uint& d                [[buffer(5)]],
        constant uint& k                [[buffer(6)]],
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;

        float best_score = INFINITY;
        uint best_idx = 0;

        for (uint c = 0; c < nc; c++) {
            float dot = 0.0;
            for (uint dd = 0; dd < d; dd++) {
                dot += queries[qid * d + dd] * corpus[c * d + dd];
            }
            // simple min check (not true L2, just testing)
            if (dot < best_score) {
                best_score = dot;
                best_idx = c;
            }
        }

        out_scores[qid * k] = best_score;
        out_scores[qid * k + 1] = (float)best_idx;
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "knn_test6").expect("compile");
    
    let query_buf = ctx.new_buffer(&queries);
    let corpus_buf = ctx.new_buffer(&corpus);
    let out_scores = ctx.new_buffer_uninitialized((nq * k * 4) as u64);

    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&query_buf), 0);
    enc.set_buffer(1, Some(&corpus_buf), 0);
    enc.set_buffer(2, Some(&out_scores), 0);
    let params: [u32; 4] = [nq as u32, nc as u32, d as u32, k as u32];
    enc.set_bytes(3, 16, params.as_ptr().cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let scores: Vec<f32> = ctx.read_buffer(&out_scores, nq * k);
    eprintln!("scores: {:?}", scores);
    
    // Best dot for query 0 (should be the minimum dot product)
    let mut min_dot = f32::INFINITY;
    let mut min_idx = 0u32;
    for c in 0..nc {
        let dot = queries[0*d+0]*corpus[c*d+0] + queries[0*d+1]*corpus[c*d+1] + queries[0*d+2]*corpus[c*d+2];
        if dot < min_dot { min_dot = dot; min_idx = c as u32; }
    }
    eprintln!("expected: best_dot={} idx={}", min_dot, min_idx);
    assert!((scores[0] - min_dot).abs() < 1e-4, "best dot mismatch: {} vs {}", scores[0], min_dot);
}
