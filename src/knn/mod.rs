use crate::metal::MetalContext;
use metal::*;
use std::cell::RefCell;

const SHADER_SRC: &str = include_str!("../../shaders/knn.metal");

pub struct KNNConfig {
    pub k: usize,
}

impl Default for KNNConfig {
    fn default() -> Self {
        Self { k: 5 }
    }
}

enum KernelVariant {
    Dense,
    Splitm,
    Naive,
}

struct Scratch {
    query: Option<Buffer>,
    norms_q: Option<Buffer>,
    out_score: Option<Buffer>,
    out_idx: Option<Buffer>,
}

pub struct KNN {
    config: KNNConfig,
    corpus: Vec<f32>,
    n_corpus: usize,
    d: usize,
    norms_c: Vec<f32>,
    pipeline: Option<ComputePipelineState>,
    variant: KernelVariant,
    corpus_buf: Option<Buffer>,
    norms_c_buf: Option<Buffer>,
    scratch: RefCell<Scratch>,
}

impl KNN {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            corpus: Vec::new(),
            n_corpus: 0,
            d: 0,
            norms_c: Vec::new(),
            pipeline: None,
            variant: KernelVariant::Dense,
            corpus_buf: None,
            norms_c_buf: None,
            scratch: RefCell::new(Scratch {
                query: None,
                norms_q: None,
                out_score: None,
                out_idx: None,
            }),
        }
    }

    pub fn fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()> {
        anyhow::ensure!(d > 0 && n > 0, "Invalid parameters: n={}, d={}", n, d);
        anyhow::ensure!(data.len() == n * d, "Data length mismatch: expected {}, got {}", n * d, data.len());

        self.corpus = data.to_vec();
        self.n_corpus = n;
        self.d = d;
        self.norms_c = compute_norms(data, n, d);

        self.corpus_buf = Some(ctx.new_buffer(&self.corpus));
        self.norms_c_buf = Some(ctx.new_buffer(&self.norms_c));

        self.variant = if d < 32 && self.config.k <= 64 {
            KernelVariant::Dense
        } else if d >= 8 && d % 8 == 0 && self.config.k <= 64 {
            KernelVariant::Splitm
        } else {
            KernelVariant::Naive
        };
        let kernel_name = match self.variant {
            KernelVariant::Dense => "knn_assign_dense",
            KernelVariant::Splitm => "knn_assign_splitm",
            KernelVariant::Naive => "knn_assign_naive",
        };
        self.pipeline = Some(ctx.compile_kernel(SHADER_SRC, kernel_name)?);
        let _gather = ctx.compile_kernel(SHADER_SRC, "knn_gather_l2")?;

        Ok(())
    }

    pub fn kneighbors(&self, ctx: &MetalContext, queries: &[f32], nq: usize) -> anyhow::Result<(Vec<f32>, Vec<u32>)> {
        let k = self.config.k;
        anyhow::ensure!(self.n_corpus > 0, "KNN has not been fitted");
        anyhow::ensure!(queries.len() == nq * self.d, "Query length mismatch");

        let splits = 1usize;
        let norms_q = compute_norms(queries, nq, self.d);

        let corpus_buf = self.corpus_buf.as_ref().unwrap();
        let norms_c_buf = self.norms_c_buf.as_ref().unwrap();

        let nqk = nq * k;
        let mut s = self.scratch.borrow_mut();

        // query buffer
        let qbytes = (queries.len() * 4) as u64;
        let query_buf = if let Some(buf) = &s.query {
            if buf.length() >= qbytes {
                unsafe { std::ptr::copy_nonoverlapping(queries.as_ptr(), buf.contents() as *mut f32, queries.len()); }
                buf.clone()
            } else {
                let buf = ctx.new_buffer(queries);
                s.query = Some(buf.clone());
                buf
            }
        } else {
            let buf = ctx.new_buffer(queries);
            s.query = Some(buf.clone());
            buf
        };

        // norms_q buffer
        let nqbytes = (norms_q.len() * 4) as u64;
        let norms_q_buf = if let Some(buf) = &s.norms_q {
            if buf.length() >= nqbytes {
                unsafe { std::ptr::copy_nonoverlapping(norms_q.as_ptr(), buf.contents() as *mut f32, norms_q.len()); }
                buf.clone()
            } else {
                let buf = ctx.new_buffer(&norms_q);
                s.norms_q = Some(buf.clone());
                buf
            }
        } else {
            let buf = ctx.new_buffer(&norms_q);
            s.norms_q = Some(buf.clone());
            buf
        };

        // output buffers
        let out_bytes = (nqk * 4) as u64;
        let out_score_buf = match &s.out_score {
            Some(buf) if buf.length() >= out_bytes => buf.clone(),
            _ => {
                let buf = ctx.new_buffer_uninitialized(out_bytes);
                s.out_score = Some(buf.clone());
                buf
            }
        };
        let out_idx_buf = match &s.out_idx {
            Some(buf) if buf.length() >= out_bytes => buf.clone(),
            _ => {
                let buf = ctx.new_buffer_uninitialized(out_bytes);
                s.out_idx = Some(buf.clone());
                buf
            }
        };
        drop(s);

        let pipeline = self.pipeline.as_ref().unwrap();

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);

        enc.set_buffer(0, Some(&query_buf), 0);
        enc.set_buffer(1, Some(&corpus_buf), 0);
        enc.set_buffer(2, Some(&out_score_buf), 0);
        enc.set_buffer(3, Some(&out_idx_buf), 0);
        enc.set_buffer(4, Some(&norms_q_buf), 0);
        enc.set_buffer(5, Some(&norms_c_buf), 0);
        set_u32(&enc, 6, nq as u32);
        set_u32(&enc, 7, self.n_corpus as u32);
        set_u32(&enc, 8, self.d as u32);
        set_u32(&enc, 9, k as u32);

        match self.variant {
            KernelVariant::Dense => {
                let nq_blocks = (nq as u64 + 127) / 128;
                let groups = MTLSize { width: nq_blocks, height: 1, depth: 1 };
                let tg = MTLSize { width: 128, height: 1, depth: 1 };
                enc.dispatch_thread_groups(groups, tg);
            }
            KernelVariant::Splitm => {
                set_u32(&enc, 10, splits as u32);
                set_u32(&enc, 11, self.n_corpus as u32);
                let num_tiles = (self.d + 7) / 8;
                let shared_floats = 16 * self.d
                    + self.d * 8
                    + num_tiles * 16 * 8
                    + 16 * k
                    + 16 * k;
                let shared_bytes = (shared_floats as u64) * 4;
                if shared_bytes > 0 {
                    enc.set_threadgroup_memory_length(0, shared_bytes);
                }
                let nq_blocks = (nq as u64 + 15) / 16;
                let groups = MTLSize { width: splits as u64, height: nq_blocks, depth: 1 };
                let tg = MTLSize { width: 128, height: 1, depth: 1 };
                enc.dispatch_thread_groups(groups, tg);
            }
            KernelVariant::Naive => {
                set_u32(&enc, 10, splits as u32);
                set_u32(&enc, 11, self.n_corpus as u32);
                let tg_per_block = 256u64;
                let nq_blocks = (nq as u64 + tg_per_block - 1) / tg_per_block;
                let groups = MTLSize { width: nq_blocks, height: 1, depth: 1 };
                let tg = MTLSize { width: tg_per_block, height: 1, depth: 1 };
                enc.dispatch_thread_groups(groups, tg);
            }
        }

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let split_scores: Vec<f32> = ctx.read_buffer(&out_score_buf, nqk);
        let split_indices: Vec<u32> = ctx.read_buffer(&out_idx_buf, nqk);
        let mut out_d = vec![0.0f32; nqk];
        let mut out_i = vec![0u32; nqk];
        for q in 0..nq {
            for j in 0..k {
                let idx = split_indices[q * k + j];
                if idx < self.n_corpus as u32 {
                    out_d[q * k + j] = norms_q[q] + split_scores[q * k + j];
                    out_i[q * k + j] = idx;
                } else {
                    out_d[q * k + j] = f32::INFINITY;
                    out_i[q * k + j] = 0;
                }
            }
        }

        Ok((out_d, out_i))
    }
}

fn compute_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| data[i * d..(i + 1) * d].iter().map(|x| x * x).sum::<f32>())
        .collect()
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
}
