use metal::*;
use metal_operators::metal::MetalContext;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

fn compute_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n).map(|i| data[i*d..(i+1)*d].iter().map(|x| x*x).sum()).collect()
}

#[test]
fn test_knn_debug5() {
    let ctx = MetalContext::new().expect("Metal");

    let (nq, nc, d, k) = (2usize, 10usize, 3usize, 2usize);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);
    let norms_q = compute_norms(&queries, nq, d);
    let norms_c = compute_norms(&corpus, nc, d);

    let source = format!(r#"
    #include <metal_stdlib>
    using namespace metal;
    
    void heap_insert(thread float* dists, thread uint* idxs, uint k, float d, uint idx) {{
        if (d >= dists[k - 1]) return;
        uint pos = k - 1;
        while (pos > 0 && d < dists[pos - 1]) {{
            dists[pos] = dists[pos - 1];
            idxs[pos]  = idxs[pos - 1];
            pos--;
        }}
        dists[pos] = d;
        idxs[pos]  = idx;
    }}

    kernel void knn_test5(
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
        uint3 gid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]
    ) {{
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;

        // Load 1st query value to verify
        float val = queries[qid * d + 0];
        
        // Simple heap test
        float hd[3];
        uint  hi[3];
        hd[0] = INFINITY; hi[0] = 0;
        hd[1] = INFINITY; hi[1] = 0;
        
        // Process first 3 corpus points
        for (uint c = 0; c < min((uint)3, nc); c++) {{
            float dot = 0.0;
            for (uint dd = 0; dd < d; dd++) {{
                dot += queries[qid * d + dd] * corpus[c * d + dd];
            }}
            float score = norms_C[c] - 2.0f * dot;
            heap_insert(hd, hi, k, score, c);
        }}

        uint ob = qid * k;
        out_scores[ob + 0] = hd[0];
        out_scores[ob + 1] = hd[1];
        out_indices[ob + 0] = hi[0];
        out_indices[ob + 1] = hi[1];
        
        // Also write query[0] as marker
        out_scores[ob + 0] = val;
    }}
    "#);

    let pipeline = ctx.compile_kernel(&source, "knn_test5").expect("compile");
    
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
    let params: [u32; 4] = [nq as u32, nc as u32, d as u32, k as u32];
    enc.set_bytes(6, 16, params.as_ptr().cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let scores: Vec<f32> = ctx.read_buffer(&out_scores, nq * k);
    let indices: Vec<u32> = ctx.read_buffer(&out_indices, nq * k);
    
    eprintln!("scores: {:?}", scores);
    eprintln!("indices: {:?}", indices);
    eprintln!("expected q[0]: {}", queries[0]);
    eprintln!("expected q[1]: {}", queries[3]);
    
    // scores[0] should be queries[0] (val marker overrides hd[0])
    assert!((scores[0] - queries[0]).abs() < 1e-4, "val mismatch: {} vs {}", scores[0], queries[0]);
}
