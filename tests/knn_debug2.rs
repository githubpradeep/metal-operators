use metal::*;
use metal_operators::metal::MetalContext;
use std::time::Instant;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

fn compute_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n).map(|i| data[i*d..(i+1)*d].iter().map(|x| x*x).sum()).collect()
}

#[test]
fn test_knn_debug2() {
    let ctx = MetalContext::new().expect("Metal");

    let (nq, nc, d, k) = (2usize, 10usize, 3usize, 2usize);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);
    let norms_q = compute_norms(&queries, nq, d);
    let norms_c = compute_norms(&corpus, nc, d);

    // Minimal kernel: just compute one dot product and write it out
    let source = r#"
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void test_min(
        device const float* queries    [[buffer(0)]],
        device const float* corpus      [[buffer(1)]],
        device float* out_scores        [[buffer(2)]],
        device uint* out_indices        [[buffer(3)]],
        device const float* norms_Q     [[buffer(4)]],
        device const float* norms_C     [[buffer(5)]],
        constant uint& nq               [[buffer(6)]],
        constant uint& nc               [[buffer(7)]],
        constant uint& d                [[buffer(8)]],
        constant uint& k                [[buffer(9)]],
        constant uint& num_splits       [[buffer(10)]],
        constant uint& m_per_split      [[buffer(11)]],
        threadgroup float* shared       [[threadgroup(0)]],
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;
        
        // Just compute dot(q0, c0) and write it out
        float dot = queries[qid * d + 0] * corpus[0 * d + 0]
                  + queries[qid * d + 1] * corpus[0 * d + 1]
                  + queries[qid * d + 2] * corpus[0 * d + 2];
        out_scores[qid * k + 0] = dot;
        out_scores[qid * k + 1] = 42.0f;
        out_indices[qid * k + 0] = 123u;
        out_indices[qid * k + 1] = 456u;
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "test_min").expect("compile");
    
    let query_buf = ctx.new_buffer(&queries);
    let corpus_buf = ctx.new_buffer(&corpus);
    let norms_q_buf = ctx.new_buffer(&norms_q);
    let norms_c_buf = ctx.new_buffer(&norms_c);
    
    let out_scores = ctx.new_buffer_uninitialized((nq * k * 4) as u64);
    let out_indices = ctx.new_buffer_uninitialized((nq * k * 4) as u64);

    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&query_buf), 0);
    enc.set_buffer(1, Some(&corpus_buf), 0);
    enc.set_buffer(2, Some(&out_scores), 0);
    enc.set_buffer(3, Some(&out_indices), 0);
    enc.set_buffer(4, Some(&norms_q_buf), 0);
    enc.set_buffer(5, Some(&norms_c_buf), 0);
    let params: [u32; 6] = [nq as u32, nc as u32, d as u32, k as u32, 1, nc as u32];
    enc.set_bytes(6, 24, params.as_ptr().cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let scores: Vec<f32> = ctx.read_buffer(&out_scores, nq * k);
    let indices: Vec<u32> = ctx.read_buffer(&out_indices, nq * k);
    
    eprintln!("scores: {:?}", scores);
    eprintln!("indices: {:?}", indices);

    // Verify dot product against expected
    let expected_dot = queries[0]*corpus[0] + queries[1]*corpus[1] + queries[2]*corpus[2];
    eprintln!("expected dot(q0,c0): {}", expected_dot);
    assert!((scores[0] - expected_dot).abs() < 1e-4, "dot product mismatch: {} vs {}", scores[0], expected_dot);
}
