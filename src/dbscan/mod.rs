//! DBSCAN clustering with GPU-accelerated ε-neighborhood computation.
//!
//! `fit` computes each point's exact ε-neighborhood on the GPU with the
//! `dbscan_count` / `dbscan_gather` kernels, which reuse the same squared-L2
//! distance approach as the KNN distance kernels (`shaders/knn.metal`) but
//! enumerate *every* neighbor within `eps` (no per-query `k` cap). Core-point
//! detection, mutual-core connectivity and border/noise labelling are then
//! resolved on the CPU with a union-find (deterministic, ordered labels).
//!
//! Semantics (scikit-learn compatible): a point is a **core** point when it
//! has at least `min_samples` points (including itself) within distance
//! `eps`. Two core points are in the same cluster when they are mutually
//! within `eps`; any point within `eps` of a core point joins that core's
//! cluster. Points reachable by no core point are labelled `-1` (noise).

use crate::metal::MetalContext;
use metal::*;

const SHADER_SRC: &str = include_str!("../../shaders/dbscan.metal");

pub struct DBSCANConfig {
    pub eps: f32,
    pub min_samples: usize,
}

impl Default for DBSCANConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
        }
    }
}

pub struct DBSCAN {
    config: DBSCANConfig,
    n: usize,
    d: usize,
    labels: Vec<isize>,
    n_clusters: usize,
}

impl DBSCAN {
    pub fn new(config: DBSCANConfig) -> Self {
        Self {
            config,
            n: 0,
            d: 0,
            labels: Vec::new(),
            n_clusters: 0,
        }
    }

    pub fn labels(&self) -> &[isize] {
        &self.labels
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(n > 0 && d > 0, "Invalid parameters: n={}, d={}", n, d);
        anyhow::ensure!(
            data.len() == n * d,
            "Data length mismatch: expected {}, got {}",
            n * d,
            data.len()
        );
        let eps = self.config.eps;
        let min_samples = self.config.min_samples;
        anyhow::ensure!(eps > 0.0, "eps must be > 0, got {}", eps);
        anyhow::ensure!(
            min_samples >= 1,
            "min_samples must be >= 1, got {}",
            min_samples
        );

        self.n = n;
        self.d = d;

        let data_buf = ctx.new_buffer(data);
        let eps_sq = eps * eps;

        // Pass A: per-point counts of other points within eps.
        let count_pipeline = ctx.compile_kernel(SHADER_SRC, "dbscan_count")?;
        let counts_buf = ctx.new_buffer_uninitialized((n * 4) as u64);

        {
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&count_pipeline);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(&counts_buf), 0);
            set_u32(&enc, 2, n as u32);
            set_u32(&enc, 3, d as u32);
            set_f32(&enc, 4, eps_sq);
            let groups = MTLSize {
                width: (n as u64 + 255) / 256,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(groups, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let counts: Vec<u32> = ctx.read_buffer(&counts_buf, n);
        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        let total = offsets[n] as usize;

        // Pass B: gather compact neighbor lists (CSR).
        let gather_pipeline = ctx.compile_kernel(SHADER_SRC, "dbscan_gather")?;
        let offsets_buf = ctx.new_buffer(&offsets);
        let nbr_buf = ctx.new_buffer_uninitialized((total.max(1) * 4) as u64);

        {
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&gather_pipeline);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(&offsets_buf), 0);
            enc.set_buffer(2, Some(&nbr_buf), 0);
            set_u32(&enc, 3, n as u32);
            set_u32(&enc, 4, d as u32);
            set_f32(&enc, 5, eps_sq);
            let groups = MTLSize {
                width: (n as u64 + 255) / 256,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(groups, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let neighbors: Vec<u32> = if total > 0 {
            ctx.read_buffer(&nbr_buf, total)
        } else {
            Vec::new()
        };

        // Core detection: counts (others within eps) + 1 (self) >= min_samples.
        let core: Vec<bool> = counts
            .iter()
            .map(|&c| (c as usize) + 1 >= min_samples)
            .collect();

        // Union-find over core points connected by ε-adjacency (exact graph).
        let mut uf = UnionFind::new(n);
        for p in 0..n {
            if !core[p] {
                continue;
            }
            for i in offsets[p] as usize..offsets[p + 1] as usize {
                let q = neighbors[i] as usize;
                if core[q] {
                    uf.union(p, q);
                }
            }
        }

        // Assign cluster ids to core components in first-appearance order.
        let mut labels: Vec<isize> = vec![-1; n];
        let mut cluster_of_root: Vec<isize> = vec![-1; n];
        let mut next = 0isize;
        for p in 0..n {
            if core[p] {
                let r = uf.find(p);
                if cluster_of_root[r] == -1 {
                    cluster_of_root[r] = next;
                    next += 1;
                }
            }
        }
        for p in 0..n {
            if core[p] {
                labels[p] = cluster_of_root[uf.find(p)];
            }
        }

        // Border points: join the first core neighbor's cluster.
        for p in 0..n {
            if core[p] {
                continue;
            }
            for i in offsets[p] as usize..offsets[p + 1] as usize {
                let q = neighbors[i] as usize;
                if core[q] && labels[q] >= 0 {
                    labels[p] = labels[q];
                    break;
                }
            }
        }

        self.labels = labels;
        self.n_clusters = next as usize;

        Ok(())
    }
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
}

fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    let len = std::mem::size_of::<f32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
}
