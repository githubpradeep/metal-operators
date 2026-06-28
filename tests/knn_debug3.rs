use metal::*;
use metal_operators::metal::MetalContext;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n * d).map(|_| rng.f32() * 10.0).collect()
}

fn compute_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n).map(|i| data[i*d..(i+1)*d].iter().map(|x| x*x).sum()).collect()
}

fn cpu_knn(corpus: &[f32], nc: usize, d: usize, queries: &[f32], nq: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut distances = vec![0.0f32; nq * k];
    let mut indices = vec![0u32; nq * k];
    for q in 0..nq {
        let mut pairs: Vec<(f32, u32)> = (0..nc as u32)
            .map(|c| {
                let mut dist = 0.0;
                for dim in 0..d {
                    let diff = queries[q * d + dim] - corpus[c as usize * d + dim];
                    dist += diff * diff;
                }
                (dist, c)
            })
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for j in 0..k {
            distances[q * k + j] = pairs[j].0;
            indices[q * k + j] = pairs[j].1;
        }
    }
    (distances, indices)
}

#[test]
fn test_knn_debug_direct() {
    let ctx = MetalContext::new().expect("Metal");

    let (nq, nc, d, k) = (2usize, 10usize, 3usize, 2usize);
    let corpus = make_data(nc, d, 42);
    let queries = make_data(nq, d, 99);
    let norms_q = compute_norms(&queries, nq, d);
    let norms_c = compute_norms(&corpus, nc, d);

    // Kernel: direct device reads, no shared memory, no barriers
    let source = r#"
    #include <metal_stdlib>
    using namespace metal;
    
    void heap_insert(thread float* dists, thread uint* idxs, uint k, float d, uint idx) {
        if (d >= dists[k - 1]) return;
        uint pos = k - 1;
        while (pos > 0 && d < dists[pos - 1]) {
            dists[pos] = dists[pos - 1];
            idxs[pos]  = idxs[pos - 1];
            pos--;
        }
        dists[pos] = d;
        idxs[pos]  = idx;
    }

    kernel void knn_direct(
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
    ) {
        uint qid = gid.x * 128 + lid;
        if (qid >= nq) return;

        // Load query into registers
        float my_q[4];
        for (uint dd = 0; dd < d; dd++) {
            my_q[dd] = queries[qid * d + dd];
        }

        // Heap in registers
        float hd[4];
        uint  hi[4];
        for (uint j = 0; j < k; j++) { hd[j] = INFINITY; hi[j] = 0; }

        // Scan all corpus points — DIRECT device reads (no shared memory)
        for (uint c = 0; c < nc; c++) {
            float dot = 0.0;
            for (uint dd = 0; dd < d; dd++) {
                dot += my_q[dd] * corpus[c * d + dd];
            }
            float score = norms_C[c] - 2.0f * dot;
            heap_insert(hd, hi, k, score, c);
        }

        // Write results
        uint ob = qid * k;
        for (uint j = 0; j < k; j++) {
            out_scores[ob + j] = hd[j];
            out_indices[ob + j] = hi[j];
        }
    }
    "#;

    let pipeline = ctx.compile_kernel(source, "knn_direct").expect("compile");
    
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
    enc.set_bytes(6, 4, std::ptr::from_ref(&(nq as u32)).cast());
    enc.set_bytes(7, 4, std::ptr::from_ref(&(nc as u32)).cast());
    enc.set_bytes(8, 4, std::ptr::from_ref(&(d as u32)).cast());
    enc.set_bytes(9, 4, std::ptr::from_ref(&(k as u32)).cast());
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let gpu_scores: Vec<f32> = ctx.read_buffer(&out_scores, nq * k);
    let gpu_idx: Vec<u32> = ctx.read_buffer(&out_indices, nq * k);
    let (cpu_d, cpu_i) = cpu_knn(&corpus, nc, d, &queries, nq, k);
    
    eprintln!("GPU scores: {:?}", gpu_scores);
    eprintln!("GPU idxs: {:?}", gpu_idx);
    eprintln!("CPU dists: {:?}", cpu_d);
    eprintln!("CPU idxs: {:?}", cpu_i);

    // Test: add true L2 norms_q to get true distances
    let mut gpu_d = gpu_scores.clone();
    for q in 0..nq {
        for j in 0..k {
            gpu_d[q*k+j] = norms_q[q] + gpu_scores[q*k+j];
        }
    }
    eprintln!("GPU true dists: {:?}", gpu_d);

    for q in 0..nq {
        for j in 0..k {
            assert_eq!(gpu_idx[q*k+j], cpu_i[q*k+j], "idx q={} j={}: {:?} vs {:?}", q, j, gpu_idx, cpu_i);
            assert!((gpu_d[q*k+j] - cpu_d[q*k+j]).abs() < 1e-3, "dist q={} j={}: {} vs {}", q, j, gpu_d[q*k+j], cpu_d[q*k+j]);
        }
    }
}
